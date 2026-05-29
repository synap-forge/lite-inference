#include "server/rest/openai_api.h"
#include "core/litert/embedding_engine.h"

#ifndef LITERT_VERSION
#define LITERT_VERSION "dev"
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "core/engine.h"
#include "core/engine_manager.h"
#include "hub/model_resolver.h"
#include "server/rest/openapi.h"
#include "server/rest/reflect.h"

namespace {

// RFC 4648 base64 decode. Returns empty on invalid input.
std::vector<uint8_t> Base64Decode(const std::string& in) {
  static const int8_t kTable[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,-1,
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
    -1,-1,-1,-1,-1,-1,
    26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,
    -1,-1,-1,-1,-1
  };
  std::vector<uint8_t> out;
  out.reserve(in.size() * 3 / 4);
  int val = 0, bits = -8;
  for (unsigned char c : in) {
    if (c == '=' || c == '\n' || c == '\r') continue;
    int v = kTable[c];
    if (v < 0) return {};
    val = (val << 6) + v;
    bits += 6;
    if (bits >= 0) {
      out.push_back((val >> bits) & 0xff);
      bits -= 8;
    }
  }
  return out;
}

// Read a local file into a byte vector.
std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return {std::istreambuf_iterator<char>(f), {}};
}

// Fetch an http(s) URL into a byte vector via curl subprocess.
std::vector<uint8_t> FetchUrl(const std::string& url) {
  std::string tmp = "/tmp/litert_media_" + std::to_string(getpid()) + ".bin";
  std::string cmd = "curl --silent --location --fail --output " + tmp + " \"" + url + "\"";
  if (std::system(cmd.c_str()) != 0) return {};
  auto bytes = ReadFile(tmp);
  std::remove(tmp.c_str());
  return bytes;
}

// Returns the local absolute path if url is a local file reference, else "".
// Covers: /abs/path, ./rel/path, file:///abs/path
std::string LocalPath(const std::string& url) {
  if (url.substr(0, 7) == "file://") return url.substr(7);
  if (!url.empty() && (url[0] == '/' || url[0] == '.')) return url;
  return {};
}

// Resolve any URL/path/data-URI to raw bytes.
// Supports: http(s)://, file://, /abs/path, ./rel/path, data:...;base64,...
std::vector<uint8_t> DecodeDataUrl(const std::string& url) {
  if (url.substr(0, 8) == "https://" || url.substr(0, 7) == "http://")
    return FetchUrl(url);
  std::string lp = LocalPath(url);
  if (!lp.empty()) return ReadFile(lp);
  const std::string prefix = "base64,";
  auto pos = url.find(prefix);
  return Base64Decode(pos != std::string::npos ? url.substr(pos + prefix.size()) : url);
}

}  // namespace

namespace lite_inference {
namespace {

using json = nlohmann::json;

int64_t NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static std::atomic<uint64_t> g_counter{0};
std::string NewId(const char* prefix) {
  return std::string(prefix) + "-" + std::to_string(NowSeconds()) + "-" +
         std::to_string(g_counter.fetch_add(1));
}

void JsonError(httplib::Response& res, int status, const std::string& msg,
               const std::string& type) {
  res.status = status;
  res.set_content(
      json{{"error", {{"message", msg}, {"type", type}, {"code", nullptr}}}}
          .dump(),
      "application/json");
}

bool Authorized(const httplib::Request& req, const ServerOptions& opts) {
  if (opts.api_key.empty()) return true;
  auto it = req.headers.find("Authorization");
  return it != req.headers.end() &&
         it->second == "Bearer " + opts.api_key;
}

struct ParsedRequest {
  std::vector<ChatMessage> messages;
  GenerationParams params;
  bool stream = false;
};

bool ParseChat(const json& body, ParsedRequest& out, std::string& err) {
  if (!body.contains("messages") || !body["messages"].is_array()) {
    err = "'messages' must be an array";
    return false;
  }
  for (const auto& m : body["messages"]) {
    ChatMessage cm;
    cm.role = m.value("role", "user");
    if (m.contains("content")) {
      if (m["content"].is_string()) {
        cm.content = m["content"].get<std::string>();
      } else if (m["content"].is_array()) {
        for (const auto& p : m["content"]) {
          const std::string type = p.value("type", "");
          if (type == "text") {
            cm.content += p.value("text", "");
          } else if (type == "image_url" && p.contains("image_url")) {
            std::string url = p["image_url"].value("url", "");
            std::string lp = LocalPath(url);
            if (!lp.empty()) {
              cm.media.push_back({MediaType::kImage, lp, {}});
            } else {
              auto bytes = DecodeDataUrl(url);
              if (!bytes.empty())
                cm.media.push_back({MediaType::kImage, {}, std::move(bytes)});
            }
          } else if (type == "input_audio" && p.contains("input_audio")) {
            const auto& ia = p["input_audio"];
            if (ia.contains("url")) {
              std::string url = ia.value("url", "");
              std::string lp = LocalPath(url);
              if (!lp.empty()) {
                cm.media.push_back({MediaType::kAudio, lp, {}});
              } else {
                auto bytes = DecodeDataUrl(url);
                if (!bytes.empty())
                  cm.media.push_back({MediaType::kAudio, {}, std::move(bytes)});
              }
            } else {
              auto bytes = Base64Decode(ia.value("data", ""));
              if (!bytes.empty())
                cm.media.push_back({MediaType::kAudio, {}, std::move(bytes)});
            }
          }
        }
      }
    }
    out.messages.push_back(std::move(cm));
  }
  out.params.temperature         = body.value("temperature", 1.0f);
  out.params.top_p               = body.value("top_p", 1.0f);
  out.params.top_k               = body.value("top_k", 0);
  out.params.max_tokens          = body.value("max_tokens", 0);
  out.params.visual_token_budget = body.value("visual_token_budget", 256);
  out.stream                     = body.value("stream", false);
  return true;
}

// Serialises all inference calls: LiteRT-LM's conversation object is not
// thread-safe. Concurrent HTTP requests are accepted, queued here, and
// processed one at a time. The queue depth is tracked so callers can see
// their position in a 503 when the server is at capacity.
struct InferenceQueue {
  std::mutex              mtx;
  std::condition_variable cv;
  int                     active  = 0;  // 0 or 1 (currently inferring)
  int                     waiting = 0;  // requests blocked on the queue
  static constexpr int    kMaxQueue = 8;

  // RAII guard: blocks until it is this request's turn, then releases on
  // destruction so the next waiter is unblocked.
  struct Slot {
    InferenceQueue& q;
    bool            acquired = false;

    explicit Slot(InferenceQueue& q, httplib::Response& res) : q(q) {
      std::unique_lock<std::mutex> lk(q.mtx);
      if (q.waiting >= kMaxQueue) {
        res.status = 503;
        res.set_content(
            R"({"error":{"message":"Server busy — too many queued requests","type":"server_error"}})",
            "application/json");
        return;
      }
      ++q.waiting;
      q.cv.wait(lk, [&q] { return q.active == 0; });
      --q.waiting;
      q.active = 1;
      acquired = true;
    }

    ~Slot() {
      if (!acquired) return;
      std::lock_guard<std::mutex> lk(q.mtx);
      q.active = 0;
      q.cv.notify_one();
    }

    // Non-copyable, non-movable.
    Slot(const Slot&)            = delete;
    Slot& operator=(const Slot&) = delete;
  };
};

static InferenceQueue g_queue;

void HandleChatCompletions(EngineManager& manager, const ServerOptions& opts,
                           const httplib::Request& req,
                           httplib::Response& res) {
  if (!Authorized(req, opts)) {
    JsonError(res, 401, "Invalid API key", "invalid_request_error");
    return;
  }
  json body;
  try { body = json::parse(req.body); }
  catch (const std::exception& e) {
    JsonError(res, 400, std::string("Invalid JSON: ") + e.what(),
              "invalid_request_error");
    return;
  }

  // Lazy startup: if engine is null and a lazy builder is configured, init now
  // on the first incoming request. Drop the probe guard before taking the write
  // lock inside EnsureLoaded to avoid shared/exclusive lock inversion.
  // The "model" field from the request body is forwarded as the hint so the
  // lazy builder can resolve the correct repo without --hf_repo at startup.
  if (!manager.Acquire() && opts.lazy_builder) {
    const std::string model_hint = body.value("model", "");
    std::string lazy_err;
    if (!manager.EnsureLoaded(
            [&](std::string& e) { return opts.lazy_builder(model_hint, e); },
            lazy_err)) {
      JsonError(res, 503, "Model load failed: " + lazy_err, "server_error");
      return;
    }
  }

  // Pin the active engine for the entire request. While this guard is held,
  // POST /v1/models/load will block (it takes the unique lock).
  auto guard = std::make_shared<EngineManager::Guard>(manager.Acquire());
  if (!*guard) {
    JsonError(res, 503, "No model loaded", "server_error");
    return;
  }
  EngineBase& engine = guard->engine();

  // --startup_load partial defers warmup to here: run it once before the first
  // real generation. The guard keeps this engine pinned across the warmup.
  std::string warmup_err;
  if (!manager.EnsureWarmed(engine, warmup_err)) {
    JsonError(res, 503, "Engine warmup failed: " + warmup_err, "server_error");
    return;
  }

  if (body.contains("model") && body["model"].is_string()) {
    const std::string requested = body["model"].get<std::string>();
    if (requested != engine.model_id()) {
      JsonError(res, 400,
                "Model '" + requested + "' is not loaded. "
                "This server is running '" + engine.model_id() + "'. "
                "Call POST /v1/models/load to switch, or start with --hf_repo " +
                    requested,
                "invalid_request_error");
      return;
    }
  }

  ParsedRequest pr;
  std::string parse_err;
  if (!ParseChat(body, pr, parse_err)) {
    JsonError(res, 400, parse_err, "invalid_request_error");
    return;
  }

  const std::string id      = NewId("chatcmpl");
  const int64_t     created = NowSeconds();
  const std::string model   = engine.model_id();

  if (pr.stream) {
    // Acquire the slot here so we can move it into the lambda, keeping it
    // alive for the full duration of the chunked stream.
    auto slot = std::make_shared<InferenceQueue::Slot>(g_queue, res);
    if (!slot->acquired) return;

    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [guard, messages = pr.messages, params = pr.params,
         id, created, model, slot](size_t, httplib::DataSink& sink) -> bool {
          EngineBase& engine = guard->engine();
          json head = {{"id", id}, {"object", "chat.completion.chunk"},
                       {"created", created}, {"model", model},
                       {"choices", {{{"index", 0},
                                     {"delta", {{"role", "assistant"}}},
                                     {"finish_reason", nullptr}}}}};
          std::string s = "data: " + head.dump() + "\n\n";
          sink.write(s.data(), s.size());

          std::string err;
          engine.GenerateStream(
              messages, params,
              [&sink, &id, created, &model](const std::string& delta) -> bool {
                json chunk = {{"id", id}, {"object", "chat.completion.chunk"},
                              {"created", created}, {"model", model},
                              {"choices", {{{"index", 0},
                                            {"delta", {{"content", delta}}},
                                            {"finish_reason", nullptr}}}}};
                std::string out = "data: " + chunk.dump() + "\n\n";
                return sink.write(out.data(), out.size());
              },
              err);

          json tail = {{"id", id}, {"object", "chat.completion.chunk"},
                       {"created", created}, {"model", model},
                       {"choices", {{{"index", 0},
                                     {"delta", json::object()},
                                     {"finish_reason", "stop"}}}}};
          s = "data: " + tail.dump() + "\n\ndata: [DONE]\n\n";
          sink.write(s.data(), s.size());
          sink.done();
          return true;
        });
    return;
  }

  // Non-streaming: hold the slot for the duration of the synchronous call.
  InferenceQueue::Slot slot(g_queue, res);
  if (!slot.acquired) return;

  std::string err;
  std::string text = engine.Generate(pr.messages, pr.params, err);
  if (!err.empty()) {
    JsonError(res, 500, err, "internal_error");
    return;
  }

  json out = {{"id", id}, {"object", "chat.completion"},
              {"created", created}, {"model", model},
              {"choices", {{{"index", 0},
                            {"message", {{"role", "assistant"},
                                         {"content", text}}},
                            {"finish_reason", "stop"}}}},
              {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0},
                         {"total_tokens", 0}}}};
  res.set_content(out.dump(), "application/json");
}

void HandleModels(EngineManager& manager, httplib::Response& res) {
  auto guard = manager.Acquire();
  const std::string active = guard ? guard.engine().model_id() : "";

  json data = json::array();
  // Currently loaded engine first.
  if (!active.empty()) {
    data.push_back({{"id", active}, {"object", "model"},
                    {"created", NowSeconds()}, {"owned_by", "litert-lm"},
                    {"status", "loaded"}});
  }
  // Anything else sitting in the HF cache, available for /load.
  for (const auto& c : ListCachedModels()) {
    if (c.repo_id == active) continue;
    data.push_back({{"id", c.repo_id}, {"object", "model"},
                    {"created", NowSeconds()}, {"owned_by", "litert-lm"},
                    {"status", "cached"},
                    {"revision", c.revision},
                    {"filename", c.filename}});
  }
  res.set_content(json{{"object", "list"}, {"data", data}}.dump(),
                  "application/json");
}

// Parse the load/pull request body into a LoadRequest. Both endpoints share
// the same shape; pull ignores backend/mtp/multimodal but accepts them
// silently for client convenience.
bool ParseLoadRequest(const json& body, ServerOptions::LoadRequest& out,
                      std::string& err) {
  if (!body.contains("repo_id") || !body["repo_id"].is_string()) {
    err = "'repo_id' is required (string)";
    return false;
  }
  out.repo_id  = body["repo_id"].get<std::string>();
  out.filename = body.value("filename", "");
  out.revision = body.value("revision", "");
  out.backend  = body.value("backend", "");
  if (body.contains("multimodal") && body["multimodal"].is_boolean()) {
    out.has_multimodal = true;
    out.multimodal     = body["multimodal"].get<bool>();
  }
  if (body.contains("mtp") && body["mtp"].is_boolean()) {
    out.has_mtp = true;
    out.mtp     = body["mtp"].get<bool>();
  }
  if (body.contains("context_length") && body["context_length"].is_number_integer()) {
    int v = body["context_length"].get<int>();
    if (v > 0) out.context_length = static_cast<size_t>(v);
  }
  return true;
}

void HandlePull(const httplib::Request& req, httplib::Response& res) {
  json body;
  try { body = json::parse(req.body); }
  catch (const std::exception& e) {
    JsonError(res, 400, std::string("Invalid JSON: ") + e.what(),
              "invalid_request_error");
    return;
  }
  ServerOptions::LoadRequest lr;
  std::string err;
  if (!ParseLoadRequest(body, lr, err)) {
    JsonError(res, 400, err, "invalid_request_error");
    return;
  }
  ResolveOptions ro;
  ro.repo_id  = lr.repo_id;
  ro.filename = lr.filename;
  if (!lr.revision.empty()) ro.revision = lr.revision;

  std::string resolve_err;
  std::string path = ResolveModel(ro, resolve_err);
  if (path.empty()) {
    JsonError(res, 500, "Pull failed: " + resolve_err, "server_error");
    return;
  }
  json out = {{"repo_id", lr.repo_id}, {"path", path}, {"status", "ready"}};
  res.set_content(out.dump(), "application/json");
}

void HandleLoad(EngineManager& manager, const ServerOptions& opts,
                const httplib::Request& req, httplib::Response& res) {
  if (!opts.build_engine) {
    JsonError(res, 500, "Server has no engine factory configured",
              "server_error");
    return;
  }
  json body;
  try { body = json::parse(req.body); }
  catch (const std::exception& e) {
    JsonError(res, 400, std::string("Invalid JSON: ") + e.what(),
              "invalid_request_error");
    return;
  }
  ServerOptions::LoadRequest lr;
  std::string err;
  if (!ParseLoadRequest(body, lr, err)) {
    JsonError(res, 400, err, "invalid_request_error");
    return;
  }

  // Drain in-flight inference before flipping engines. The shared_mutex in
  // EngineManager would already serialize the swap, but draining here keeps
  // the queue's invariant simple: no Generate() calls run while an engine
  // pointer is being replaced underneath them.
  {
    std::unique_lock<std::mutex> lk(g_queue.mtx);
    g_queue.cv.wait(lk, [] { return g_queue.active == 0; });
  }

  std::string build_err;
  bool ok = manager.Swap(
      [&](std::string& e) { return opts.build_engine(lr, e); }, build_err);
  if (!ok) {
    JsonError(res, 500, "Load failed: " + build_err, "server_error");
    return;
  }

  auto guard = manager.Acquire();
  json out = {{"status", "ok"},
              {"model", guard ? guard.engine().model_id() : lr.repo_id},
              {"backend", guard ? guard.engine().active_backend() : ""}};
  res.set_content(out.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// DTOs used solely for OpenAPI schema generation. Handlers continue to parse
// nlohmann::json directly — these structs exist so Body<T>()/Response<T>()
// can derive a JSON Schema via reflection.
// ---------------------------------------------------------------------------

struct ChatMessageDto {
  std::string                role;
  std::string                content;  // string or content-part array; described as string here
  LITERT_REFLECT(ChatMessageDto, role, content)
};

struct ChatCompletionRequestDto {
  std::string                                  model;
  std::vector<ChatMessageDto>                  messages;
  std::optional<float>                         temperature;
  std::optional<float>                         top_p;
  std::optional<int>                           top_k;
  std::optional<int>                           max_tokens;
  std::optional<int>                           visual_token_budget;
  std::optional<bool>                          stream;
  LITERT_REFLECT(ChatCompletionRequestDto,
                 model, messages, temperature, top_p, top_k,
                 max_tokens, visual_token_budget, stream)
};

struct ChatChoiceDto {
  int                  index = 0;
  ChatMessageDto       message;
  std::string          finish_reason;
  LITERT_REFLECT(ChatChoiceDto, index, message, finish_reason)
};

struct UsageDto {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  LITERT_REFLECT(UsageDto, prompt_tokens, completion_tokens, total_tokens)
};

struct ChatCompletionResponseDto {
  std::string                  id;
  std::string                  object;
  int                          created = 0;
  std::string                  model;
  std::vector<ChatChoiceDto>   choices;
  UsageDto                     usage;
  LITERT_REFLECT(ChatCompletionResponseDto,
                 id, object, created, model, choices, usage)
};

struct ModelEntryDto {
  std::string id;
  std::string object;
  int         created = 0;
  std::string owned_by;
  std::string status;
  std::optional<std::string> revision;
  std::optional<std::string> filename;
  LITERT_REFLECT(ModelEntryDto, id, object, created, owned_by, status, revision, filename)
};

struct ModelsListResponseDto {
  std::string                 object;
  std::vector<ModelEntryDto>  data;
  LITERT_REFLECT(ModelsListResponseDto, object, data)
};

struct LoadRequestDto {
  std::string                 repo_id;
  std::optional<std::string>  filename;
  std::optional<std::string>  revision;
  std::optional<std::string>  backend;
  std::optional<bool>         multimodal;
  std::optional<bool>         mtp;
  LITERT_REFLECT(LoadRequestDto, repo_id, filename, revision, backend, multimodal, mtp)
};

struct PullResponseDto {
  std::string repo_id;
  std::string path;
  std::string status;
  LITERT_REFLECT(PullResponseDto, repo_id, path, status)
};

struct LoadResponseDto {
  std::string status;
  std::string model;
  std::string backend;
  LITERT_REFLECT(LoadResponseDto, status, model, backend)
};

struct QueueDto {
  int active = 0;
  int waiting = 0;
  int capacity = 0;
  LITERT_REFLECT(QueueDto, active, waiting, capacity)
};

struct EngineInfoDto {
  std::optional<std::string> model;
  std::optional<std::string> backend;
  std::optional<bool>        multimodal;
  std::optional<bool>        mtp;
  LITERT_REFLECT(EngineInfoDto, model, backend, multimodal, mtp)
};

struct HealthResponseDto {
  std::string    status;
  std::string    version;
  EngineInfoDto  engine;
  QueueDto       queue;
  LITERT_REFLECT(HealthResponseDto, status, version, engine, queue)
};

struct ErrorObjectDto {
  std::string                 message;
  std::string                 type;
  std::optional<std::string>  code;
  LITERT_REFLECT(ErrorObjectDto, message, type, code)
};

struct ErrorResponseDto {
  ErrorObjectDto error;
  LITERT_REFLECT(ErrorResponseDto, error)
};

struct EmbeddingRequestDto {
  std::string                 model;
  std::string                 input;   // string or array; described as string here
  std::optional<std::string>  encoding_format;
  LITERT_REFLECT(EmbeddingRequestDto, model, input, encoding_format)
};

struct EmbeddingObjectDto {
  std::string        object;  // "embedding"
  int                index = 0;
  std::vector<float> embedding;
  LITERT_REFLECT(EmbeddingObjectDto, object, index, embedding)
};

struct EmbeddingUsageDto {
  int prompt_tokens = 0;
  int total_tokens  = 0;
  LITERT_REFLECT(EmbeddingUsageDto, prompt_tokens, total_tokens)
};

struct EmbeddingResponseDto {
  std::string                    object;  // "list"
  std::vector<EmbeddingObjectDto> data;
  std::string                    model;
  EmbeddingUsageDto              usage;
  LITERT_REFLECT(EmbeddingResponseDto, object, data, model, usage)
};

void HandleEmbeddings(const ServerOptions& opts,
                      const httplib::Request& req,
                      httplib::Response& res) {
  if (!opts.embedding_engine) {
    JsonError(res, 501, "No embedding model loaded. "
                        "Start the server with --embed_repo to enable /v1/embeddings.",
              "not_implemented");
    return;
  }

  json body;
  try { body = json::parse(req.body); }
  catch (const std::exception& e) {
    JsonError(res, 400, std::string("Invalid JSON: ") + e.what(),
              "invalid_request_error");
    return;
  }

  // Collect input texts: either a single string or an array of strings.
  std::vector<std::string> inputs;
  if (!body.contains("input")) {
    JsonError(res, 400, "'input' is required", "invalid_request_error");
    return;
  }
  if (body["input"].is_string()) {
    inputs.push_back(body["input"].get<std::string>());
  } else if (body["input"].is_array()) {
    for (const auto& item : body["input"]) {
      if (!item.is_string()) {
        JsonError(res, 400, "'input' array must contain strings only",
                  "invalid_request_error");
        return;
      }
      inputs.push_back(item.get<std::string>());
    }
  } else {
    JsonError(res, 400, "'input' must be a string or array of strings",
              "invalid_request_error");
    return;
  }

  EmbeddingEngine& emb = *opts.embedding_engine;
  json data = json::array();
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string err;
    std::vector<float> vec = emb.Embed(inputs[i], err);
    if (!err.empty()) {
      JsonError(res, 500, "Embedding failed: " + err, "internal_error");
      return;
    }
    data.push_back({{"object", "embedding"},
                    {"index", static_cast<int>(i)},
                    {"embedding", vec}});
  }

  json out = {{"object", "list"},
              {"data", data},
              {"model", emb.model_id()},
              {"usage", {{"prompt_tokens", 0}, {"total_tokens", 0}}}};
  res.set_content(out.dump(), "application/json");
}

}  // namespace

int RunServer(EngineManager& manager, const ServerOptions& options) {
  httplib::Server svr;
  namespace oa = lite_inference::openapi;

  oa::Route(svr, "GET", "/health")
      .Summary("Server and engine health, queue depth, version.")
      .Tag("system")
      .Response<HealthResponseDto>(200)
      .Handler([&manager](const httplib::Request&, httplib::Response& res) {
        auto guard = manager.Acquire();
        json engine_info = json::object();
        if (guard) {
          engine_info["model"]          = guard.engine().model_id();
          engine_info["backend"]        = guard.engine().active_backend();
          engine_info["multimodal"]     = guard.engine().multimodal();
          engine_info["mtp"]            = guard.engine().mtp();
          engine_info["context_length"] = guard.engine().context_length();
        }
        std::lock_guard<std::mutex> lk(g_queue.mtx);
        json out = {{"status", "ok"},
                    {"version", LITERT_VERSION},
                    {"engine", engine_info},
                    {"queue", {{"active",  g_queue.active},
                               {"waiting", g_queue.waiting},
                               {"capacity", InferenceQueue::kMaxQueue}}}};
        res.set_content(out.dump(), "application/json");
      });

  oa::Route(svr, "GET", "/v1/models")
      .Summary("List the active model plus anything cached locally.")
      .Tag("models")
      .Response<ModelsListResponseDto>(200)
      .Handler([&manager](const httplib::Request&, httplib::Response& res) {
        HandleModels(manager, res);
      });

  oa::Route(svr, "POST", "/v1/models/pull")
      .Summary("Resolve and download a model into the local cache.")
      .Tag("models")
      .Body<LoadRequestDto>()
      .Response<PullResponseDto>(200)
      .Response<ErrorResponseDto>(400, "Invalid request")
      .Response<ErrorResponseDto>(401, "Invalid API key")
      .Response<ErrorResponseDto>(500, "Resolve failed")
      .Handler([&options](const httplib::Request& req, httplib::Response& res) {
        if (!Authorized(req, options)) {
          JsonError(res, 401, "Invalid API key", "invalid_request_error");
          return;
        }
        HandlePull(req, res);
      });

  oa::Route(svr, "POST", "/v1/models/load")
      .Summary("Swap the active engine to a different model.")
      .Tag("models")
      .Body<LoadRequestDto>()
      .Response<LoadResponseDto>(200)
      .Response<ErrorResponseDto>(400, "Invalid request")
      .Response<ErrorResponseDto>(401, "Invalid API key")
      .Response<ErrorResponseDto>(500, "Load failed")
      .Handler([&manager, &options](const httplib::Request& req,
                                    httplib::Response& res) {
        if (!Authorized(req, options)) {
          JsonError(res, 401, "Invalid API key", "invalid_request_error");
          return;
        }
        HandleLoad(manager, options, req, res);
      });

  oa::Route(svr, "POST", "/v1/embeddings")
      .Summary("OpenAI-compatible embeddings. Requires --embed_repo at startup.")
      .Tag("embeddings")
      .Body<EmbeddingRequestDto>()
      .Response<EmbeddingResponseDto>(200)
      .Response<ErrorResponseDto>(400, "Invalid request")
      .Response<ErrorResponseDto>(401, "Invalid API key")
      .Response<ErrorResponseDto>(501, "No embedding model loaded")
      .Handler([&options](const httplib::Request& req, httplib::Response& res) {
        if (!Authorized(req, options)) {
          JsonError(res, 401, "Invalid API key", "invalid_request_error");
          return;
        }
        HandleEmbeddings(options, req, res);
      });

  oa::Route(svr, "POST", "/v1/chat/completions")
      .Summary("OpenAI-compatible chat completions. Set stream=true for SSE.")
      .Tag("chat")
      .Body<ChatCompletionRequestDto>()
      .Response<ChatCompletionResponseDto>(200, "Non-streaming completion")
      .RawResponse(200, "Server-sent events when stream=true; each event is "
                        "'data: <ChatCompletionChunk>\\n\\n', terminated by "
                        "'data: [DONE]\\n\\n'.",
                   "text/event-stream")
      .Response<ErrorResponseDto>(400, "Invalid request")
      .Response<ErrorResponseDto>(401, "Invalid API key")
      .Response<ErrorResponseDto>(503, "Server busy or no model loaded")
      .Handler([&manager, &options](const httplib::Request& req,
                                    httplib::Response& res) {
        HandleChatCompletions(manager, options, req, res);
      });

  // OpenAPI document. Built lazily on first request so all routes above are
  // registered by the time we serialize.
  svr.Get("/openapi.json",
          [](const httplib::Request&, httplib::Response& res) {
            auto spec = oa::BuildSpec("LiteRT Inference Server", LITERT_VERSION);
            res.set_content(spec.dump(2), "application/json");
          });

  // Swagger UI loaded from CDN. Tiny static HTML, no binary bloat.
  svr.Get("/docs", [](const httplib::Request&, httplib::Response& res) {
    static const char* kHtml = R"HTML(<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LiteRT Inference API</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({ url: '/openapi.json', dom_id: '#swagger-ui' });
  </script>
</body>
</html>)HTML";
    res.set_content(kHtml, "text/html");
  });

  svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    fprintf(stderr, "%s %s -> %d\n",
            req.method.c_str(), req.path.c_str(), res.status);
  });

  {
    auto guard = manager.Acquire();
    fprintf(stderr, "Serving on http://%s:%d  model=%s  backend=%s\n",
            options.host.c_str(), options.port,
            guard ? guard.engine().model_id().c_str() : "(none)",
            guard ? guard.engine().active_backend().c_str() : "(none)");
  }

  if (!svr.listen(options.host, options.port)) {
    fprintf(stderr, "Failed to bind %s:%d\n",
            options.host.c_str(), options.port);
    return 1;
  }
  return 0;
}

}  // namespace lite_inference
