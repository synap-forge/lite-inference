#include "litert_inference.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/litert/llm_engine.h"
#include "third_party/dart/dart_api_dl.h"

// ---------------------------------------------------------------------------
// Global create-error slot (used when engine* is NULL)
// ---------------------------------------------------------------------------
namespace
{
  std::string g_create_error;
  std::mutex g_create_mutex;
} // namespace

// ---------------------------------------------------------------------------
// Internal engine wrapper
// ---------------------------------------------------------------------------
struct LiteInferenceEngine
{
  std::unique_ptr<lite_inference::EngineBase> impl;
  mutable std::string last_error;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace
{

  lite_inference::GenerationParams ToGenParams(const LiteInferenceGenParams *p)
  {
    lite_inference::GenerationParams gp;
    if (!p)
      return gp;
    gp.temperature = p->temperature > 0.0f ? p->temperature : 1.0f;
    gp.top_p = p->top_p > 0.0f ? p->top_p : 1.0f;
    gp.top_k = p->top_k;
    gp.max_tokens = p->max_tokens;
    gp.visual_token_budget = p->visual_token_budget > 0 ? p->visual_token_budget : 256;
    return gp;
  }

  std::vector<lite_inference::ChatMessage> ToMessages(
      const char *const *roles,
      const char *const *contents,
      int n)
  {
    std::vector<lite_inference::ChatMessage> msgs;
    msgs.reserve(n);
    for (int i = 0; i < n; ++i)
    {
      lite_inference::ChatMessage m;
      m.role = roles[i] ? roles[i] : "";
      m.content = contents[i] ? contents[i] : "";
      msgs.push_back(std::move(m));
    }
    return msgs;
  }

  // Like ToMessages, but also attaches media[i] (an array of media_counts[i]
  // LiteInferenceMedia) to each message. media / media_counts may be NULL.
  std::vector<lite_inference::ChatMessage> ToMessagesEx(
      const char *const *roles,
      const char *const *contents,
      const LiteInferenceMedia *const *media,
      const int *media_counts,
      int n)
  {
    std::vector<lite_inference::ChatMessage> msgs;
    msgs.reserve(n);
    for (int i = 0; i < n; ++i)
    {
      lite_inference::ChatMessage m;
      m.role = roles[i] ? roles[i] : "";
      m.content = contents[i] ? contents[i] : "";

      if (media && media_counts && media[i] && media_counts[i] > 0)
      {
        const LiteInferenceMedia *atts = media[i];
        const int count = media_counts[i];
        m.media.reserve(count);
        for (int j = 0; j < count; ++j)
        {
          lite_inference::MediaAttachment att;
          att.type = (atts[j].type == LITE_INFERENCE_MEDIA_AUDIO)
                         ? lite_inference::MediaType::kAudio
                         : lite_inference::MediaType::kImage;
          if (atts[j].path && atts[j].path[0] != '\0')
          {
            att.path = atts[j].path;
          }
          else if (atts[j].data && atts[j].data_len > 0)
          {
            att.data.assign(atts[j].data, atts[j].data + atts[j].data_len);
          }
          m.media.push_back(std::move(att));
        }
      }
      msgs.push_back(std::move(m));
    }
    return msgs;
  }

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

LiteInferenceEngine *LiteInferenceEngineCreate(const LiteInferenceOptions *opts)
{
  if (!opts || !opts->model_path)
  {
    std::lock_guard<std::mutex> lk(g_create_mutex);
    g_create_error = "LiteInferenceEngineCreate: opts or model_path is NULL";
    return nullptr;
  }

  lite_inference::EngineOptions eo;
  eo.backend = opts->backend ? opts->backend : "gpu";
  eo.force_gpu = opts->force_gpu != 0;
  eo.multimodal = opts->multimodal != 0;
  eo.mtp = opts->mtp != 0;
  if (opts->cache_dir && opts->cache_dir[0])
    eo.cache_dir = opts->cache_dir;
  if (opts->dispatch_lib_dir && opts->dispatch_lib_dir[0])
    eo.dispatch_lib_dir = opts->dispatch_lib_dir;

  // model_id: derive from filename (last path component, no extension)
  std::string model_path = opts->model_path;
  std::string model_id;
  {
    auto slash = model_path.find_last_of("/\\");
    model_id = (slash == std::string::npos) ? model_path : model_path.substr(slash + 1);
    auto dot = model_id.rfind('.');
    if (dot != std::string::npos)
      model_id.erase(dot);
  }

  auto wrapper = std::make_unique<LiteInferenceEngine>();
  std::string create_err;
  wrapper->impl = lite_inference::LlmEngine::Create(model_path, model_id, eo, create_err);
  if (!wrapper->impl)
  {
    std::lock_guard<std::mutex> lk(g_create_mutex);
    g_create_error = create_err.empty() ? "LiteInferenceEngineCreate: unknown error" : create_err;
    return nullptr;
  }
  return wrapper.release();
}

void LiteInferenceEngineDestroy(LiteInferenceEngine *engine)
{
  delete engine;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

const char *LiteInferenceModelId(const LiteInferenceEngine *engine)
{
  if (!engine || !engine->impl)
    return "";
  return engine->impl->model_id().c_str();
}

const char *LiteInferenceActiveBackend(const LiteInferenceEngine *engine)
{
  if (!engine || !engine->impl)
    return "";
  return engine->impl->active_backend().c_str();
}

void LiteInferenceWarmup(LiteInferenceEngine *engine)
{
  if (engine && engine->impl)
    engine->impl->Warmup();
}

// ---------------------------------------------------------------------------
// Inference — blocking
// ---------------------------------------------------------------------------

char *LiteInferenceGenerate(
    LiteInferenceEngine *engine,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    const LiteInferenceGenParams *params)
{
  if (!engine || !engine->impl)
    return nullptr;

  auto msgs = ToMessages(roles, contents, n_messages);
  auto gp = ToGenParams(params);

  std::string err;
  std::string result = engine->impl->Generate(msgs, gp, err);

  if (!err.empty())
  {
    engine->last_error = err;
    return nullptr;
  }
  engine->last_error.clear();

  // Caller must free with LiteInferenceFreeString().
  char *out = static_cast<char *>(std::malloc(result.size() + 1));
  if (!out)
  {
    engine->last_error = "LiteInferenceGenerate: malloc failed";
    return nullptr;
  }
  std::memcpy(out, result.c_str(), result.size() + 1);
  return out;
}

char *LiteInferenceGenerateEx(
    LiteInferenceEngine *engine,
    const char *const *roles,
    const char *const *contents,
    const LiteInferenceMedia *const *media,
    const int *media_counts,
    int n_messages,
    const LiteInferenceGenParams *params)
{
  if (!engine || !engine->impl)
    return nullptr;

  auto msgs = ToMessagesEx(roles, contents, media, media_counts, n_messages);
  auto gp = ToGenParams(params);

  std::string err;
  std::string result = engine->impl->Generate(msgs, gp, err);

  if (!err.empty())
  {
    engine->last_error = err;
    return nullptr;
  }
  engine->last_error.clear();

  char *out = static_cast<char *>(std::malloc(result.size() + 1));
  if (!out)
  {
    engine->last_error = "LiteInferenceGenerateEx: malloc failed";
    return nullptr;
  }
  std::memcpy(out, result.c_str(), result.size() + 1);
  return out;
}

// ---------------------------------------------------------------------------
// Inference — streaming
// ---------------------------------------------------------------------------

int LiteInferenceGenerateStream(
    LiteInferenceEngine *engine,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    const LiteInferenceGenParams *params,
    LiteInferenceTokenCallback callback,
    void *user_data)
{
  if (!engine || !engine->impl || !callback)
    return 0;

  auto msgs = ToMessages(roles, contents, n_messages);
  auto gp = ToGenParams(params);

  lite_inference::TokenCallback on_token =
      [callback, user_data](const std::string &delta) -> bool
  {
    return callback(delta.c_str(), user_data) != 0;
  };

  std::string err;
  bool ok = engine->impl->GenerateStream(msgs, gp, on_token, err);

  if (!err.empty())
    engine->last_error = err;
  else
    engine->last_error.clear();

  return ok ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Dart DL API init + port-based streaming
// ---------------------------------------------------------------------------

int LiteInferenceInitDartApi(void *data)
{
  return Dart_InitializeApiDL(data);
}

namespace
{
  // Post a UTF-8 string to a Dart SendPort as a Dart String. Thread-safe.
  void PostString(int64_t port, const std::string &s)
  {
    Dart_CObject obj;
    obj.type = Dart_CObject_kString;
    // Dart copies the string during the post, so a stack pointer is fine.
    obj.value.as_string = const_cast<char *>(s.c_str());
    Dart_PostCObject_DL(port, &obj);
  }
} // namespace

int LiteInferenceGenerateStreamPort(
    LiteInferenceEngine *engine,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    const LiteInferenceGenParams *params,
    int64_t send_port)
{
  if (!engine || !engine->impl)
    return 0;

  auto msgs = ToMessages(roles, contents, n_messages);
  auto gp = ToGenParams(params);

  // Posting to a port is safe from LiteRT-LM's internal engine thread, unlike
  // a Dart FFI callback. Each delta is copied into the port message by Dart.
  lite_inference::TokenCallback on_token =
      [send_port](const std::string &delta) -> bool
  {
    if (!delta.empty())
      PostString(send_port, delta);
    return true; // never cancel from here
  };

  std::string err;
  bool ok = engine->impl->GenerateStream(msgs, gp, on_token, err);

  if (!err.empty())
    engine->last_error = err;
  else
    engine->last_error.clear();

  return ok ? 1 : 0;
}

int LiteInferenceGenerateStreamPortEx(
    LiteInferenceEngine *engine,
    const char *const *roles,
    const char *const *contents,
    const LiteInferenceMedia *const *media,
    const int *media_counts,
    int n_messages,
    const LiteInferenceGenParams *params,
    int64_t send_port)
{
  if (!engine || !engine->impl)
    return 0;

  auto msgs = ToMessagesEx(roles, contents, media, media_counts, n_messages);
  auto gp = ToGenParams(params);

  lite_inference::TokenCallback on_token =
      [send_port](const std::string &delta) -> bool
  {
    if (!delta.empty())
      PostString(send_port, delta);
    return true;
  };

  std::string err;
  bool ok = engine->impl->GenerateStream(msgs, gp, on_token, err);

  if (!err.empty())
    engine->last_error = err;
  else
    engine->last_error.clear();

  return ok ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

const char *LiteInferenceGetError(const LiteInferenceEngine *engine)
{
  if (!engine)
  {
    std::lock_guard<std::mutex> lk(g_create_mutex);
    return g_create_error.c_str();
  }
  return engine->last_error.c_str();
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

void LiteInferenceFreeString(char *str)
{
  std::free(str);
}
