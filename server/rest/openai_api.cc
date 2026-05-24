#include "server/rest/openai_api.h"

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

void HandleChatCompletions(EngineBase& engine, const ServerOptions& opts,
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

  if (body.contains("model") && body["model"].is_string()) {
    const std::string requested = body["model"].get<std::string>();
    if (requested != engine.model_id()) {
      JsonError(res, 400,
                "Model '" + requested + "' is not loaded. "
                "This server is running '" + engine.model_id() + "'. "
                "Start the server with --hf_repo " + requested,
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
        [&engine, messages = pr.messages, params = pr.params,
         id, created, model, slot](size_t, httplib::DataSink& sink) -> bool {
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

void HandleModels(EngineBase& engine, httplib::Response& res) {
  json out = {{"object", "list"},
              {"data", {{{"id", engine.model_id()}, {"object", "model"},
                         {"created", NowSeconds()}, {"owned_by", "litert-lm"}}}}};
  res.set_content(out.dump(), "application/json");
}

}  // namespace

int RunServer(EngineBase& engine, const ServerOptions& options) {
  httplib::Server svr;

  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_queue.mtx);
    json out = {{"status", "ok"},
                {"queue", {{"active",  g_queue.active},
                           {"waiting", g_queue.waiting},
                           {"capacity", InferenceQueue::kMaxQueue}}}};
    res.set_content(out.dump(), "application/json");
  });

  svr.Get("/v1/models",
          [&engine](const httplib::Request&, httplib::Response& res) {
            HandleModels(engine, res);
          });

  svr.Post("/v1/chat/completions",
           [&engine, &options](const httplib::Request& req,
                               httplib::Response& res) {
             HandleChatCompletions(engine, options, req, res);
           });

  svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    fprintf(stderr, "%s %s -> %d\n",
            req.method.c_str(), req.path.c_str(), res.status);
  });

  fprintf(stderr, "Serving on http://%s:%d  model=%s  backend=%s\n",
          options.host.c_str(), options.port,
          engine.model_id().c_str(), engine.active_backend().c_str());

  if (!svr.listen(options.host, options.port)) {
    fprintf(stderr, "Failed to bind %s:%d\n",
            options.host.c_str(), options.port);
    return 1;
  }
  return 0;
}

}  // namespace lite_inference
