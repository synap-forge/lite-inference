#include "core/litert/llm_engine.h"

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "c/engine.h"
#include "nlohmann/json.hpp"

namespace lite_inference {
namespace {

using json = nlohmann::json;

// Build the JSON object for a single ChatMessage in LiteRT-LM's native format:
//   path-based:  {"type":"image","path":"/abs/path"}
//   inline data: {"type":"image","blob":"<base64>"}
// The conversation API's ModelDataProcessor (Gemma4DataProcessor) reads these,
// runs StbImagePreprocessor on the raw bytes, and injects the image tokens.
std::string BuildSingleMessageJson(const ChatMessage& m) {
  json content = json::array();
  for (const auto& att : m.media) {
    const char* type_str = (att.type == MediaType::kImage) ? "image" : "audio";
    if (!att.path.empty()) {
      content.push_back({{"type", type_str}, {"path", att.path}});
    } else {
      static const char kB64[] =
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      std::string b64;
      b64.reserve(((att.data.size() + 2) / 3) * 4);
      for (size_t i = 0; i < att.data.size(); i += 3) {
        uint32_t v = att.data[i] << 16;
        if (i + 1 < att.data.size()) v |= att.data[i + 1] << 8;
        if (i + 2 < att.data.size()) v |= att.data[i + 2];
        b64 += kB64[(v >> 18) & 63];
        b64 += kB64[(v >> 12) & 63];
        b64 += (i + 1 < att.data.size()) ? kB64[(v >> 6) & 63] : '=';
        b64 += (i + 2 < att.data.size()) ? kB64[v & 63] : '=';
      }
      content.push_back({{"type", type_str}, {"blob", b64}});
    }
  }
  if (!m.content.empty())
    content.push_back({{"type", "text"}, {"text", m.content}});
  return json{{"role", m.role}, {"content", content}}.dump();
}

// Returns a new optional_args with visual_token_budget set, or nullptr if 0.
LiteRtLmConversationOptionalArgs* MakeOptionalArgs(int visual_token_budget) {
  if (visual_token_budget <= 0) return nullptr;
  LiteRtLmConversationOptionalArgs* args =
      litert_lm_conversation_optional_args_create();
  litert_lm_conversation_optional_args_set_visual_token_budget(
      args, visual_token_budget);
  return args;
}

LiteRtLmConversationConfig* MakeConversationConfig(
    const GenerationParams& params) {
  LiteRtLmSamplerParams sampler{};
  sampler.temperature = params.temperature;
  sampler.top_p       = params.top_p;
  sampler.top_k       = params.top_k;

  LiteRtLmSessionConfig* session_cfg = litert_lm_session_config_create();
  litert_lm_session_config_set_sampler_params(session_cfg, &sampler);
  if (params.max_tokens > 0)
    litert_lm_session_config_set_max_output_tokens(session_cfg, params.max_tokens);
  LiteRtLmConversationConfig* conv_cfg = litert_lm_conversation_config_create();
  litert_lm_conversation_config_set_session_config(conv_cfg, session_cfg);
  litert_lm_session_config_delete(session_cfg);
  return conv_cfg;
}

// Extract text from a JSON message chunk (conversation API response format).
std::string ExtractTextFromChunk(const char* chunk) {
  if (!chunk || !*chunk) return {};
  try {
    auto j = nlohmann::json::parse(chunk);
    std::string out;
    for (const auto& part : j.value("content", nlohmann::json::array()))
      if (part.value("type", "") == "text") out += part.value("text", "");
    return out;
  } catch (...) {
    return chunk;
  }
}

}  // namespace

struct LlmEngine::Impl {
  LiteRtLmEngine* engine = nullptr;
  bool multimodal        = false;
};

LlmEngine::~LlmEngine() {
  if (impl_ && impl_->engine)
    litert_lm_engine_delete(impl_->engine);
}

std::unique_ptr<LlmEngine> LlmEngine::Create(const std::string& model_path,
                                             const std::string& model_id,
                                             const EngineOptions& opts,
                                             std::string& error_out) {
  auto self = std::unique_ptr<LlmEngine>(new LlmEngine());
  self->impl_ = std::make_unique<Impl>();
  self->impl_->multimodal = opts.multimodal;

  const bool want_gpu = (opts.backend == "gpu" || opts.backend == "GPU");

  // vision_backend / audio_backend are only set when --multimodal is on.
  // Keeping them nullptr skips Gemma4DataProcessor init entirely, which
  // eliminates the per-request ~100 ms multimodal setup overhead.
  const char* vision_be = opts.multimodal ? "cpu" : nullptr;
  const char* audio_be  = opts.multimodal ? "cpu" : nullptr;

  auto try_create = [&](const char* backend_str) -> LiteRtLmEngine* {
    LiteRtLmEngineSettings* settings =
        litert_lm_engine_settings_create(model_path.c_str(), backend_str,
                                         vision_be, audio_be);
    if (vision_be)
      litert_lm_engine_settings_set_max_num_images(settings, 1);
    if (opts.mtp)
      litert_lm_engine_settings_set_enable_speculative_decoding(settings, true);
    LiteRtLmEngine* eng = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    return eng;
  };

  if (want_gpu) {
    fprintf(stderr, "Initializing GPU (Metal) engine%s%s...\n",
            opts.multimodal ? " +multimodal" : "",
            opts.mtp        ? " +mtp"        : "");
    self->impl_->engine = try_create("gpu");
    if (self->impl_->engine) {
      self->active_backend_ = "gpu";
    } else if (opts.force_gpu) {
      error_out = "GPU engine init failed and --force_gpu is set; refusing CPU fallback";
      return nullptr;
    } else {
      fprintf(stderr, "GPU init failed, falling back to CPU.\n");
    }
  }

  if (!self->impl_->engine) {
    self->impl_->engine = try_create("cpu");
    if (!self->impl_->engine) {
      error_out = "Failed to create LiteRT-LM engine (tried cpu)";
      return nullptr;
    }
    self->active_backend_ = "cpu";
  }

  if (!model_id.empty()) {
    self->model_id_ = model_id;
  } else {
    auto slash = model_path.find_last_of("/\\");
    self->model_id_ = (slash == std::string::npos)
                          ? model_path
                          : model_path.substr(slash + 1);
  }

  fprintf(stderr, "Engine ready: backend=%s multimodal=%s mtp=%s\n",
          self->active_backend_.c_str(),
          opts.multimodal ? "on" : "off",
          opts.mtp ? "on" : "off");
  return self;
}

void LlmEngine::Warmup() {
  fprintf(stderr, "Warming up engine (first-request cold-start elimination)...\n");
  GenerationParams params;
  params.max_tokens = 1;
  params.temperature = 0.0f;
  std::string err;
  Generate({ChatMessage{"user", "hi"}}, params, err);
  fprintf(stderr, "Warmup complete.\n");
}

std::string LlmEngine::Generate(const std::vector<ChatMessage>& messages,
                                const GenerationParams& params,
                                std::string& error_out) {
  LiteRtLmConversationConfig* conv_cfg = MakeConversationConfig(params);
  LiteRtLmConversation* conv =
      litert_lm_conversation_create(impl_->engine, conv_cfg);
  litert_lm_conversation_config_delete(conv_cfg);
  if (!conv) { error_out = "Failed to create conversation"; return {}; }

  LiteRtLmConversationOptionalArgs* opt_args =
      impl_->multimodal ? MakeOptionalArgs(params.visual_token_budget) : nullptr;

  // Prefill all messages except the last via send_message (responses discarded),
  // then send the last message to trigger decode.
  for (size_t i = 0; i + 1 < messages.size(); ++i) {
    std::string prefill_json = BuildSingleMessageJson(messages[i]);
    LiteRtLmJsonResponse* r =
        litert_lm_conversation_send_message(conv, prefill_json.c_str(), nullptr, opt_args);
    if (r) litert_lm_json_response_delete(r);
  }

  std::string msg_json = BuildSingleMessageJson(messages.back());
  LiteRtLmJsonResponse* resp =
      litert_lm_conversation_send_message(conv, msg_json.c_str(), nullptr, opt_args);
  if (opt_args) litert_lm_conversation_optional_args_delete(opt_args);

  std::string result;
  if (!resp) {
    error_out = "send_message returned null";
  } else {
    const char* raw = litert_lm_json_response_get_string(resp);
    if (raw) result = ExtractTextFromChunk(raw);
    litert_lm_json_response_delete(resp);
  }
  litert_lm_conversation_delete(conv);
  return result;
}

bool LlmEngine::GenerateStream(const std::vector<ChatMessage>& messages,
                               const GenerationParams& params,
                               const TokenCallback& on_token,
                               std::string& error_out) {
  struct CallbackState {
    const TokenCallback* on_token;
    std::string error;
    bool cancelled = false;
    bool done      = false;
    std::mutex mtx;
    std::condition_variable cv;
  } state{&on_token};

  auto callback = [](void* data, const char* chunk, bool is_final,
                     const char* error_msg) {
    auto* s = static_cast<CallbackState*>(data);
    if (error_msg) {
      std::lock_guard<std::mutex> lk(s->mtx);
      s->error = error_msg;
      s->done  = true;
      s->cv.notify_all();
      return;
    }
    if (chunk && *chunk && !s->cancelled) {
      std::string text = ExtractTextFromChunk(chunk);
      if (!text.empty() && !(*s->on_token)(text))
        s->cancelled = true;
    }
    if (is_final) {
      std::lock_guard<std::mutex> lk(s->mtx);
      s->done = true;
      s->cv.notify_all();
    }
  };

  LiteRtLmConversationConfig* conv_cfg = MakeConversationConfig(params);
  LiteRtLmConversation* conv =
      litert_lm_conversation_create(impl_->engine, conv_cfg);
  litert_lm_conversation_config_delete(conv_cfg);
  if (!conv) { error_out = "Failed to create conversation"; return false; }

  LiteRtLmConversationOptionalArgs* opt_args =
      impl_->multimodal ? MakeOptionalArgs(params.visual_token_budget) : nullptr;

  // Prefill all messages except the last, then stream the final decode.
  for (size_t i = 0; i + 1 < messages.size(); ++i) {
    std::string prefill_json = BuildSingleMessageJson(messages[i]);
    LiteRtLmJsonResponse* r =
        litert_lm_conversation_send_message(conv, prefill_json.c_str(), nullptr, opt_args);
    if (r) litert_lm_json_response_delete(r);
  }

  std::string msg_json = BuildSingleMessageJson(messages.back());
  int rc = litert_lm_conversation_send_message_stream(
      conv, msg_json.c_str(), nullptr, opt_args, callback, &state);
  if (opt_args) litert_lm_conversation_optional_args_delete(opt_args);
  if (rc != 0) {
    error_out = "send_message_stream failed with code " + std::to_string(rc);
    litert_lm_conversation_delete(conv);
    return false;
  }

  std::unique_lock<std::mutex> lk(state.mtx);
  state.cv.wait(lk, [&] { return state.done; });
  litert_lm_conversation_delete(conv);

  if (!state.error.empty()) { error_out = state.error; return false; }
  return true;
}

}  // namespace lite_inference
