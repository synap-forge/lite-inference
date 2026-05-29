#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "core/engine_manager.h"
#include "core/litert/embedding_engine.h"
#include "core/litert/llm_engine.h"
#include "hub/model_resolver.h"
#include "server/rest/openai_api.h"

#ifndef LITERT_VERSION
#define LITERT_VERSION "dev"
#endif

static void Usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "\n"
    "Model resolution:\n"
    "  --model_path PATH      Explicit .litertlm/.tflite file (overrides HF resolution)\n"
    "  --hf_repo  REPO        HF repo id (default: litert-community/gemma-4-E2B-it-litert-lm)\n"
    "  --hf_file  FILE        File within repo (default: autodetect *.litertlm / *.tflite)\n"
    "  --hf_revision REV      HF revision/branch (default: main)\n"
    "\n"
    "Engine:\n"
    "  --backend  BACKEND     auto (default) | gpu | cpu\n"
    "                         'auto' tries GPU and falls back to CPU on failure\n"
    "  --force_gpu            Hard-fail if GPU init fails; no CPU fallback\n"
    "  --multimodal           Enable vision + audio sub-backends (cpu). Off by default\n"
    "                         to avoid per-request Gemma4DataProcessor overhead\n"
    "  --mtp [DRAFT_TOKENS]   Enable multi-token prediction (speculative decoding)\n"
    "                         Default draft tokens: 3\n"
    "  --context_length N     KV-cache context window size in tokens.\n"
    "                         0 (default): auto-detect from ekv#### in model filename.\n"
    "                         Set explicitly for models without ekv in their filename.\n"
    "  --startup_load MODE    full (default) | partial | none\n"
    "\n"
    "Embeddings (optional):\n"
    "  --embed_repo  REPO     HF repo for embedding .tflite (e.g. litert-community/embeddinggemma-300m)\n"
    "  --embed_file  FILE     Specific .tflite file within the repo (default: autodetect)\n"
    "  --embed_seq_len N      Sequence length the embedding model was built for (default: 512)\n"
    "                         full:    load + warmup at startup (lowest first-request latency)\n"
    "                         partial: load at startup, skip warmup (first request pays GPU init)\n"
    "                         none:    defer everything to first request (fastest startup, slow first call)\n"
    "\n"
    "Server:\n"
    "  --host     HOST        Bind host (default: 0.0.0.0)\n"
    "  --port     PORT        Listen port (default: 8080)\n"
    "  --api_key  KEY         Require Authorization: Bearer KEY\n"
    "\n"
    "Env vars: HF_HOME (cache root), HF_TOKEN (private repos)\n"
    "\n"
    "  --version, -V          Print version and exit\n",
    prog);
}

static const char* GetArg(int argc, char** argv, const char* flag,
                           const char* fallback = "") {
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return argv[i + 1];
  return fallback;
}

static bool HasFlag(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return true;
  return false;
}

// Returns the value after `flag` if it looks like an integer, else `fallback`.
static int GetOptionalInt(int argc, char** argv, const char* flag, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (strcmp(argv[i], flag) == 0) {
      char* end = nullptr;
      long v = strtol(argv[i + 1], &end, 10);
      if (end != argv[i + 1] && *end == '\0') return static_cast<int>(v);
    }
  }
  return fallback;
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      Usage(argv[0]); return 0;
    }
    if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-V") == 0) {
      printf("%s\n", LITERT_VERSION);
      return 0;
    }
  }

  const std::string startup_load = GetArg(argc, argv, "--startup_load", "full");
  if (startup_load != "full" && startup_load != "partial" && startup_load != "none") {
    fprintf(stderr, "Invalid --startup_load value '%s'. Use: full | partial | none\n",
            startup_load.c_str());
    return 1;
  }

  lite_inference::EngineOptions engine_opts;
  engine_opts.backend        = GetArg(argc, argv, "--backend", "auto");
  engine_opts.force_gpu      = HasFlag(argc, argv, "--force_gpu");
  engine_opts.multimodal     = HasFlag(argc, argv, "--multimodal");
  engine_opts.mtp            = HasFlag(argc, argv, "--mtp");
  engine_opts.context_length = static_cast<size_t>(
      GetOptionalInt(argc, argv, "--context_length", 0));

  std::unique_ptr<lite_inference::LlmEngine> engine;

  if (startup_load != "none") {
    lite_inference::ResolveOptions resolve;
    resolve.model_path = GetArg(argc, argv, "--model_path");
    resolve.repo_id    = GetArg(argc, argv, "--hf_repo",
                                 "litert-community/gemma-4-E2B-it-litert-lm");
    resolve.filename   = GetArg(argc, argv, "--hf_file");
    resolve.revision   = GetArg(argc, argv, "--hf_revision", "main");

    std::string resolve_err;
    std::string model_path = lite_inference::ResolveModel(resolve, resolve_err);
    if (model_path.empty()) {
      fprintf(stderr, "Error resolving model: %s\n", resolve_err.c_str());
      return 2;
    }
    const std::string model_id = resolve.model_path.empty() ? resolve.repo_id : "";

    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    std::string engine_err;
    engine = lite_inference::LlmEngine::Create(
        model_path, model_id, engine_opts, engine_err);
    if (!engine) {
      fprintf(stderr, "Engine init failed: %s\n", engine_err.c_str());
      return 1;
    }

    if (startup_load == "full") {
      std::string warmup_err;
      if (!engine->Warmup(warmup_err)) {
        fprintf(stderr, "Engine warmup failed: %s\n", warmup_err.c_str());
        return 1;
      }
    } else {
      fprintf(stderr, "Model loaded (warmup deferred to first request)\n");
    }
  } else {
    fprintf(stderr, "Startup load deferred — model will load on first request\n");
  }

  lite_inference::EngineManager manager(std::move(engine));

  lite_inference::ServerOptions opts;
  opts.host    = GetArg(argc, argv, "--host",    "0.0.0.0");
  opts.port    = std::atoi(GetArg(argc, argv, "--port", "8080"));
  opts.api_key = GetArg(argc, argv, "--api_key");

  // Factory used by POST /v1/models/load to hot-swap models.
  //
  // Inheritance: backend + cache/dispatch paths carry over from startup so
  // GPU selection and prebuilt-lib lookup work identically across loads.
  // mtp and multimodal do NOT inherit — they are Gemma-family bundle features
  // (the .litertlm needs MTP drafter / vision encoder sections), so applying
  // them to a fresh model that wasn't built with those sections causes
  // engine init to fail with RET_CHECK_NE(embedding_lookup, nullptr) etc.
  // Callers opt in per request: {"mtp": true} or {"multimodal": true}.
  const lite_inference::EngineOptions startup_opts = engine_opts;
  opts.build_engine =
      [startup_opts](const lite_inference::ServerOptions::LoadRequest& req,
                     std::string& err) -> std::unique_ptr<lite_inference::EngineBase> {
        lite_inference::ResolveOptions ro;
        ro.repo_id  = req.repo_id;
        ro.filename = req.filename;
        if (!req.revision.empty()) ro.revision = req.revision;

        std::string path = lite_inference::ResolveModel(ro, err);
        if (path.empty()) return nullptr;

        lite_inference::EngineOptions eo;
        eo.backend          = startup_opts.backend;
        eo.force_gpu        = startup_opts.force_gpu;
        eo.cache_dir        = startup_opts.cache_dir;
        eo.dispatch_lib_dir = startup_opts.dispatch_lib_dir;
        // Per-request opt-ins for bundle-specific features.
        eo.mtp            = req.has_mtp        ? req.mtp        : false;
        eo.multimodal     = req.has_multimodal ? req.multimodal : false;
        eo.context_length = req.context_length;  // 0 = auto-detect from filename
        if (!req.backend.empty()) eo.backend = req.backend;

        auto next = lite_inference::LlmEngine::Create(path, req.repo_id, eo, err);
        if (!next) return nullptr;
        if (!next->Warmup(err)) {
          // Engine constructed but inference is broken (e.g. GPU sampler
          // unavailable). Drop the engine — caller surfaces `err` as 500.
          return nullptr;
        }
        return next;
      };

  if (startup_load == "none") {
    // Capture the raw CLI args; resolution happens on the first request.
    // model_hint (the "model" field from the request body) takes priority;
    // --hf_repo / --model_path are used as fallbacks if the request omits it.
    const std::string default_repo     = GetArg(argc, argv, "--hf_repo", "");
    const std::string default_path     = GetArg(argc, argv, "--model_path", "");
    const std::string hf_file          = GetArg(argc, argv, "--hf_file", "");
    const std::string hf_revision      = GetArg(argc, argv, "--hf_revision", "main");
    const lite_inference::EngineOptions lazy_opts = engine_opts;

    opts.lazy_builder = [default_repo, default_path, hf_file, hf_revision, lazy_opts]
        (const std::string& model_hint, std::string& err)
        -> std::unique_ptr<lite_inference::EngineBase> {
      lite_inference::ResolveOptions ro;
      ro.model_path = default_path;
      ro.repo_id    = !model_hint.empty() ? model_hint : default_repo;
      ro.filename   = hf_file;
      ro.revision   = hf_revision;

      if (ro.repo_id.empty() && ro.model_path.empty()) {
        err = "No model specified. Provide 'model' in the request body, "
              "or pass --hf_repo / --model_path at startup.";
        return nullptr;
      }

      fprintf(stderr, "Lazy-loading model on first request: %s\n",
              ro.repo_id.empty() ? ro.model_path.c_str() : ro.repo_id.c_str());
      std::string resolve_err;
      std::string path = lite_inference::ResolveModel(ro, resolve_err);
      if (path.empty()) { err = resolve_err; return nullptr; }

      const std::string model_id = ro.model_path.empty() ? ro.repo_id : "";
      auto e = lite_inference::LlmEngine::Create(path, model_id, lazy_opts, err);
      if (!e) return nullptr;
      if (!e->Warmup(err)) return nullptr;
      return e;
    };
  }

  // Optional embedding engine.
  std::unique_ptr<lite_inference::EmbeddingEngine> embed_engine;
  const std::string embed_repo = GetArg(argc, argv, "--embed_repo");
  if (!std::string(embed_repo).empty()) {
    if (startup_load == "none") {
      fprintf(stderr, "Error: --embed_repo requires the LLM engine to be loaded "
                      "(cannot use with --startup_load=none alone).\n");
      return 1;
    }
    lite_inference::ResolveOptions embed_resolve;
    embed_resolve.repo_id  = embed_repo;
    embed_resolve.filename = GetArg(argc, argv, "--embed_file");
    embed_resolve.revision = GetArg(argc, argv, "--hf_revision", "main");

    std::string embed_err;
    std::string embed_path = lite_inference::ResolveModel(embed_resolve, embed_err);
    if (embed_path.empty()) {
      fprintf(stderr, "Error resolving embedding model: %s\n", embed_err.c_str());
      return 2;
    }

    int embed_seq_len = GetOptionalInt(argc, argv, "--embed_seq_len", 512);
    fprintf(stderr, "Loading embedding model: %s\n", embed_path.c_str());
    embed_engine = lite_inference::EmbeddingEngine::Create(
        embed_path, embed_repo, embed_seq_len, engine->raw_engine(), embed_err);
    if (!embed_engine) {
      fprintf(stderr, "Embedding engine init failed: %s\n", embed_err.c_str());
      return 1;
    }
    opts.embedding_engine = embed_engine.get();
  }

  return lite_inference::RunServer(manager, opts);
}
