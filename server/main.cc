#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

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
    "  --backend  BACKEND     gpu (default) | cpu\n"
    "  --force_gpu            Hard-fail if GPU init fails; no CPU fallback\n"
    "  --multimodal           Enable vision + audio sub-backends (cpu). Off by default\n"
    "                         to avoid per-request Gemma4DataProcessor overhead\n"
    "  --mtp [DRAFT_TOKENS]   Enable multi-token prediction (speculative decoding)\n"
    "                         Default draft tokens: 3\n"
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

  lite_inference::ResolveOptions resolve;
  resolve.model_path  = GetArg(argc, argv, "--model_path");
  resolve.repo_id     = GetArg(argc, argv, "--hf_repo",
                                "litert-community/gemma-4-E2B-it-litert-lm");
  resolve.filename    = GetArg(argc, argv, "--hf_file");
  resolve.revision    = GetArg(argc, argv, "--hf_revision", "main");

  std::string resolve_err;
  std::string model_path = lite_inference::ResolveModel(resolve, resolve_err);
  if (model_path.empty()) {
    fprintf(stderr, "Error resolving model: %s\n", resolve_err.c_str());
    return 2;
  }

  fprintf(stderr, "Loading model: %s\n", model_path.c_str());

  const std::string model_id = resolve.model_path.empty() ? resolve.repo_id : "";

  lite_inference::EngineOptions engine_opts;
  engine_opts.backend    = GetArg(argc, argv, "--backend", "gpu");
  engine_opts.force_gpu  = HasFlag(argc, argv, "--force_gpu");
  engine_opts.multimodal = HasFlag(argc, argv, "--multimodal");
  engine_opts.mtp        = HasFlag(argc, argv, "--mtp");

  std::string engine_err;
  auto engine = lite_inference::LlmEngine::Create(
      model_path, model_id, engine_opts, engine_err);
  if (!engine) {
    fprintf(stderr, "Engine init failed: %s\n", engine_err.c_str());
    return 1;
  }

  engine->Warmup();

  lite_inference::ServerOptions opts;
  opts.host    = GetArg(argc, argv, "--host",    "0.0.0.0");
  opts.port    = std::atoi(GetArg(argc, argv, "--port", "8080"));
  opts.api_key = GetArg(argc, argv, "--api_key");

  return lite_inference::RunServer(*engine, opts);
}
