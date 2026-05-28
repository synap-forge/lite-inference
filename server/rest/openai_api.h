#pragma once
#include <functional>
#include <memory>
#include <string>

#include "core/engine.h"
#include "core/engine_manager.h"
#include "core/litert/embedding_engine.h"

namespace lite_inference {

struct ServerOptions {
  std::string host    = "0.0.0.0";
  int         port    = 8080;
  std::string api_key;

  // Factory used by POST /v1/models/load to build a new engine for a given
  // repo_id/filename. Receives optional per-request overrides via the JSON
  // body (parsed by the server). The factory is responsible for resolving the
  // model (downloading if needed), constructing the engine, and warming it up.
  // Return nullptr on failure with `error_out` set.
  struct LoadRequest {
    std::string repo_id;
    std::string filename;     // optional
    std::string revision;     // optional; "" means default
    // Engine option overrides; empty/absent means inherit server defaults.
    std::string backend;      // "cpu" | "gpu" | ""
    bool        has_multimodal = false;
    bool        multimodal     = false;
    bool        has_mtp        = false;
    bool        mtp            = false;
    size_t      context_length = 0;  // 0 = auto-detect from ekv#### in filename
  };
  using EngineFactory =
      std::function<std::unique_ptr<EngineBase>(const LoadRequest& req,
                                                std::string& error_out)>;
  EngineFactory build_engine;

  // Set when --startup_load=none. Called once on the first inference request
  // to lazily resolve, create, and warm up the engine. `model_hint` is the
  // value of the "model" field from the request body (may be empty).
  // Null means eager startup.
  using LazyBuilder = std::function<std::unique_ptr<EngineBase>(
      const std::string& model_hint, std::string& error_out)>;
  LazyBuilder lazy_builder;

  // Optional embedding engine for POST /v1/embeddings.
  // Null means the endpoint returns 501.
  EmbeddingEngine* embedding_engine = nullptr;
};

int RunServer(EngineManager& manager, const ServerOptions& options);

}  // namespace lite_inference
