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

// Returns true if the message contains at least one image or audio attachment.
bool HasMedia(const ChatMessage& m) { return !m.media.empty(); }

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

// Returns the number of tokens in `text` according to the engine's tokenizer.
// Returns 0 on failure (safe to use as an undercount — callers won't trim more
// than necessary).
size_t CountTokens(LiteRtLmEngine* engine, const std::string& text) {
  if (text.empty()) return 0;
  LiteRtLmTokenizeResult* r = litert_lm_engine_tokenize(engine, text.c_str());
  if (!r) return 0;
  size_t n = litert_lm_tokenize_result_get_num_tokens(r);
  litert_lm_tokenize_result_delete(r);
  return n;
}

// Trims `messages` so the total token count of all text content fits within
// `limit` tokens, preserving:
//   - messages[0] if it is a "system" message (always kept)
//   - messages.back() (the current user turn — always kept)
// Drops the oldest non-system turns first.
// Returns the number of messages dropped (0 = no trim needed).
int TrimToContextLimit(LiteRtLmEngine* engine,
                       std::vector<ChatMessage>& messages,
                       size_t limit) {
  if (messages.size() < 2) return 0;  // nothing droppable

  // Leave ~20% headroom for image/audio tokens not counted here.
  const size_t effective_limit = limit * 4 / 5;

  auto token_count = [&](const ChatMessage& m) -> size_t {
    return CountTokens(engine, m.content);
  };

  size_t total = 0;
  for (const auto& m : messages) total += token_count(m);
  if (total <= effective_limit) return 0;

  // Keep messages[0] if it is a system prompt; always keep messages.back().
  const size_t first_droppable = (!messages.empty() && messages[0].role == "system") ? 1 : 0;

  int dropped = 0;
  // Erase oldest droppable turns one at a time; stop when only the last
  // message (current user turn) would remain after first_droppable.
  while (total > effective_limit && first_droppable + 1 < messages.size()) {
    total -= token_count(messages[first_droppable]);
    messages.erase(messages.begin() + static_cast<ptrdiff_t>(first_droppable));
    ++dropped;
  }
  return dropped;
}

}  // namespace

struct LlmEngine::Impl {
  LiteRtLmEngine* engine = nullptr;
  bool multimodal        = false;
  bool mtp               = false;
  size_t max_num_tokens  = 0;  // parsed from ekv#### in model filename; 0 = unknown
};

LlmEngine::~LlmEngine() {
  if (impl_ && impl_->engine)
    litert_lm_engine_delete(impl_->engine);
}

bool            LlmEngine::multimodal()     const { return impl_ && impl_->multimodal; }
bool            LlmEngine::mtp()            const { return impl_ && impl_->mtp; }
size_t          LlmEngine::context_length() const { return impl_ ? impl_->max_num_tokens : 0; }
LiteRtLmEngine* LlmEngine::raw_engine()     const { return impl_ ? impl_->engine : nullptr; }

std::unique_ptr<LlmEngine> LlmEngine::Create(const std::string& model_path,
                                             const std::string& model_id,
                                             const EngineOptions& opts,
                                             std::string& error_out) {
  auto self = std::unique_ptr<LlmEngine>(new LlmEngine());
  self->impl_ = std::make_unique<Impl>();
  self->impl_->multimodal = opts.multimodal;
  self->impl_->mtp        = opts.mtp;

  // "auto" tries GPU first and silently falls back to CPU. "gpu" also falls
  // back unless --force_gpu is set. "cpu" skips the GPU attempt entirely.
  const bool is_auto  = (opts.backend == "auto" || opts.backend == "AUTO" ||
                         opts.backend.empty());
  const bool want_gpu = is_auto ||
                        (opts.backend == "gpu" || opts.backend == "GPU");

  // vision/audio sub-backends are only wired up when multimodal is requested.
  // Keeping them nullptr skips Gemma4DataProcessor init entirely, which
  // eliminates the per-request ~100 ms multimodal setup overhead. The fallback
  // path in create_with_fallback() may also retry with multimodal disabled if
  // the bundle lacks vision/audio sections.
  auto try_create = [&](const char* backend_str, bool with_mtp,
                        bool with_multimodal) -> LiteRtLmEngine* {
    const char* v_be = with_multimodal ? "cpu" : nullptr;
    const char* a_be = with_multimodal ? "cpu" : nullptr;
    LiteRtLmEngineSettings* settings =
        litert_lm_engine_settings_create(model_path.c_str(), backend_str,
                                         v_be, a_be);
    if (v_be)
      litert_lm_engine_settings_set_max_num_images(settings, 1);
    if (with_mtp)
      litert_lm_engine_settings_set_enable_speculative_decoding(settings, true);
    if (!opts.cache_dir.empty())
      litert_lm_engine_settings_set_cache_dir(settings, opts.cache_dir.c_str());
    if (!opts.dispatch_lib_dir.empty())
      litert_lm_engine_settings_set_litert_dispatch_lib_dir(
          settings, opts.dispatch_lib_dir.c_str());
    LiteRtLmEngine* eng = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    return eng;
  };

  // Attempts engine creation, falling back by disabling bundle-specific
  // features the model may not support. Order: full → no-mtp → no-features.
  // MTP requires a tf_lite_mtp_drafter section in the .litertlm; multimodal
  // requires vision/audio encoders. Non-Gemma bundles often lack these.
  auto create_with_fallback = [&](const char* backend_str) -> LiteRtLmEngine* {
    bool want_mtp        = opts.mtp;
    bool want_multimodal = opts.multimodal;

    LiteRtLmEngine* eng = try_create(backend_str, want_mtp, want_multimodal);
    if (eng) return eng;

    if (want_mtp) {
      fprintf(stderr, "[lite_inference] Engine init failed with +mtp; "
                      "retrying without MTP (model may lack drafter section).\n");
      eng = try_create(backend_str, false, want_multimodal);
      if (eng) {
        self->impl_->multimodal = want_multimodal;
        self->impl_->mtp        = false;
        return eng;
      }
    }
    if (want_multimodal) {
      fprintf(stderr, "[lite_inference] Engine init failed with +multimodal; "
                      "retrying as text-only (model may lack vision/audio sections).\n");
      eng = try_create(backend_str, false, false);
      if (eng) {
        self->impl_->multimodal = false;
        self->impl_->mtp        = false;
        return eng;
      }
    }
    return nullptr;
  };

  if (want_gpu) {
    fprintf(stderr, "[lite_inference] Initializing GPU (Metal) engine%s%s...\n",
            opts.multimodal ? " +multimodal" : "",
            opts.mtp        ? " +mtp"        : "");
    self->impl_->engine = create_with_fallback("gpu");
    if (self->impl_->engine) {
      fprintf(stderr, "[lite_inference] GPU engine created successfully.\n");
      self->active_backend_ = "gpu";
    } else if (opts.force_gpu) {
      error_out = "GPU (Metal) engine init failed; no CPU fallback (force_gpu=true). "
                  "Check stderr/device console for ABSL error details.";
      return nullptr;
    } else {
      fprintf(stderr, "[lite_inference] GPU init failed, falling back to CPU.\n");
    }
  }

  if (!self->impl_->engine) {
    self->impl_->engine = create_with_fallback("cpu");
    if (!self->impl_->engine) {
      error_out = "Failed to create LiteRT-LM engine (tried cpu, including "
                  "fallbacks without mtp/multimodal). The .litertlm bundle is "
                  "incompatible with this engine build.";
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

  // Determine context window size. Prefer explicit opts.context_length;
  // fall back to parsing ekv#### from the model filename.
  if (opts.context_length > 0) {
    self->impl_->max_num_tokens = opts.context_length;
  } else {
    auto pos = model_path.find("ekv");
    if (pos != std::string::npos) {
      size_t ekv = 0;
      for (size_t i = pos + 3;
           i < model_path.size() && std::isdigit((unsigned char)model_path[i]);
           ++i)
        ekv = ekv * 10 + (model_path[i] - '0');
      if (ekv > 0) self->impl_->max_num_tokens = ekv;
    }
  }

  fprintf(stderr, "Engine ready: backend=%s multimodal=%s mtp=%s ctx=%zu\n",
          self->active_backend_.c_str(),
          opts.multimodal ? "on" : "off",
          opts.mtp ? "on" : "off",
          self->impl_->max_num_tokens);
  return self;
}

// A minimal 64x64 gray PNG, embedded so multimodal warmup can exercise the
// vision path without a file on disk. Warming up with an image forces the
// Gemma4 vision data processor and model-type metadata to fully initialize
// before the first real request — otherwise the per-request is_gemma4() check
// can race and silently drop visual_token_budget, letting the patch grid grow
// past the encoder's safe limit and segfault.
static const unsigned char kWarmupImagePng[] = {
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
  0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x40, 0x08, 0x02, 0x00, 0x00, 0x00, 0x25, 0x0b, 0xe6,
  0x89, 0x00, 0x00, 0x00, 0x4b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0xed, 0xcf, 0x31, 0x0d, 0x00,
  0x00, 0x0c, 0x03, 0xa0, 0x4a, 0xaf, 0xf4, 0x4a, 0xd8, 0xbd, 0x04, 0x1c, 0x90, 0x3e, 0x17, 0x01,
  0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
  0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
  0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x81, 0xcb, 0x00,
  0x1f, 0x3e, 0x01, 0x69, 0xe6, 0x7d, 0x4c, 0x9c, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
  0xae, 0x42, 0x60, 0x82,
};

bool LlmEngine::Warmup(std::string& error_out) {
  fprintf(stderr, "Warming up engine (first-request cold-start elimination)...\n");
  GenerationParams params;
  params.max_tokens = 1;
  params.temperature = 0.0f;

  ChatMessage msg{"user", "hi"};
  if (impl_->multimodal) {
    msg.media.push_back({MediaType::kImage, {},
                         std::vector<uint8_t>(kWarmupImagePng,
                                              kWarmupImagePng +
                                                  sizeof(kWarmupImagePng))});
  }
  Generate({msg}, params, error_out);
  if (!error_out.empty()) {
    fprintf(stderr, "Warmup failed: %s\n", error_out.c_str());
    return false;
  }
  fprintf(stderr, "Warmup complete.\n");
  return true;
}

std::string LlmEngine::Generate(const std::vector<ChatMessage>& messages,
                                const GenerationParams& params,
                                std::string& error_out) {
  std::vector<ChatMessage> trimmed = messages;
  if (impl_->max_num_tokens > 0) {
    int dropped = TrimToContextLimit(impl_->engine, trimmed, impl_->max_num_tokens);
    if (dropped > 0)
      fprintf(stderr, "[lite_inference] Context too long: dropped %d oldest turn(s) "
                      "to fit within %zu-token limit.\n", dropped, impl_->max_num_tokens);
  }
  const std::vector<ChatMessage>& msgs = trimmed;

  LiteRtLmConversationConfig* conv_cfg = MakeConversationConfig(params);
  LiteRtLmConversation* conv =
      litert_lm_conversation_create(impl_->engine, conv_cfg);
  litert_lm_conversation_config_delete(conv_cfg);
  if (!conv) { error_out = "Failed to create conversation"; return {}; }

  // opt_args is only passed for messages that actually contain media — passing
  // it for text-only messages can trigger a null-deref inside
  // ProcessAndCombineContents when the vision encoder isn't fully initialized.
  auto make_args_for = [&](const ChatMessage& m) -> LiteRtLmConversationOptionalArgs* {
    return (impl_->multimodal && HasMedia(m))
               ? MakeOptionalArgs(params.visual_token_budget)
               : nullptr;
  };

  // Prefill all messages except the last via send_message (responses discarded),
  // then send the last message to trigger decode.
  for (size_t i = 0; i + 1 < msgs.size(); ++i) {
    std::string prefill_json = BuildSingleMessageJson(msgs[i]);
    LiteRtLmConversationOptionalArgs* args = make_args_for(msgs[i]);
    LiteRtLmJsonResponse* r =
        litert_lm_conversation_send_message(conv, prefill_json.c_str(), nullptr, args);
    if (args) litert_lm_conversation_optional_args_delete(args);
    if (r) litert_lm_json_response_delete(r);
  }

  std::string msg_json = BuildSingleMessageJson(msgs.back());
  LiteRtLmConversationOptionalArgs* opt_args = make_args_for(msgs.back());
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

  std::vector<ChatMessage> trimmed = messages;
  if (impl_->max_num_tokens > 0) {
    int dropped = TrimToContextLimit(impl_->engine, trimmed, impl_->max_num_tokens);
    if (dropped > 0)
      fprintf(stderr, "[lite_inference] Context too long: dropped %d oldest turn(s) "
                      "to fit within %zu-token limit.\n", dropped, impl_->max_num_tokens);
  }
  const std::vector<ChatMessage>& msgs = trimmed;

  LiteRtLmConversationConfig* conv_cfg = MakeConversationConfig(params);
  LiteRtLmConversation* conv =
      litert_lm_conversation_create(impl_->engine, conv_cfg);
  litert_lm_conversation_config_delete(conv_cfg);
  if (!conv) { error_out = "Failed to create conversation"; return false; }

  auto make_args_for = [&](const ChatMessage& m) -> LiteRtLmConversationOptionalArgs* {
    return (impl_->multimodal && HasMedia(m))
               ? MakeOptionalArgs(params.visual_token_budget)
               : nullptr;
  };

  // Prefill all messages except the last, then stream the final decode.
  for (size_t i = 0; i + 1 < msgs.size(); ++i) {
    std::string prefill_json = BuildSingleMessageJson(msgs[i]);
    LiteRtLmConversationOptionalArgs* args = make_args_for(msgs[i]);
    LiteRtLmJsonResponse* r =
        litert_lm_conversation_send_message(conv, prefill_json.c_str(), nullptr, args);
    if (args) litert_lm_conversation_optional_args_delete(args);
    if (r) litert_lm_json_response_delete(r);
  }

  std::string msg_json = BuildSingleMessageJson(msgs.back());
  LiteRtLmConversationOptionalArgs* opt_args = make_args_for(msgs.back());
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
