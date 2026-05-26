# lite-inference — Design

A cross-platform C++ server that exposes an OpenAI-compatible REST API backed by
Google's LiteRT-LM on-device inference engine. The process loads one model at
startup, serves concurrent requests against it, and supports hot-swapping the
model at runtime.

## High-level architecture

```
                       ┌──────────────────────────────────────────────┐
                       │                  Clients                     │
                       │   curl · OpenAI SDK · benchmarks/benchmark.py│
                       └──────────────────────┬───────────────────────┘
                                              │  HTTP / SSE
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          server/main.cc  (process entry)                      │
│  1. Parse CLI flags  ─►  ResolveOptions / EngineOptions / ServerOptions       │
│  2. Resolve model    ─►  hub::ResolveModel()                                  │
│  3. Build engine     ─►  LlmEngine::Create() + Warmup()                       │
│  4. Wrap in          ─►  EngineManager(engine)                                │
│  5. Provide          ─►  build_engine factory (for /v1/models/load hot-swap)  │
│  6. RunServer(manager, opts)                                                  │
└─────────────────────────────────────┬─────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        ▼                             ▼                              ▼
┌────────────────────┐   ┌──────────────────────────────┐  ┌──────────────────────┐
│ hub/                │   │ server/rest/openai_api.cc   │  │ core/engine_manager  │
│ model_resolver.cc   │   │  (cpp-httplib + nlohmann)   │  │  shared_mutex guard  │
│                     │   │                              │  │   Acquire() ─► Guard │
│ • $HF_HOME lookup   │   │ Routes:                      │  │   Swap(builder)      │
│ • HTTPS download    │   │  GET  /health                │  │                      │
│ • ListCachedModels  │   │  GET  /v1/models             │  │ Concurrent reads,    │
│                     │   │  POST /v1/chat/completions   │  │ exclusive swap.      │
└──────────┬──────────┘   │  POST /v1/models/load  ──────┼─►│ build_engine(req)    │
           │              │                              │  └──────────┬───────────┘
           │ path         │ • Auth: Bearer api_key       │             │
           │              │ • Parse OpenAI JSON          │             │ Guard
           │              │ • Multimodal: image/audio    │             ▼
           │              │   (file/http/base64/data:)   │  ┌──────────────────────┐
           │              │ • Non-stream → Generate()    │  │  EngineBase (iface)  │
           │              │ • Stream     → SSE deltas    │  │  Generate / Stream / │
           │              │                via Token-    │  │  Warmup / model_id / │
           │              │                Callback      │  │  active_backend      │
           │              └──────────────┬───────────────┘  └──────────┬───────────┘
           │                             │                             │
           │                             └──── ChatMessage[] ──────────┤
           │                                   GenerationParams        │
           ▼                                                            ▼
┌─────────────────────┐                                ┌──────────────────────────────┐
│  HF cache on disk   │                                │ core/litert/llm_engine.cc    │
│  $HF_HOME/hub/...   │                                │  (concrete LiteRT-LM impl)   │
│  snapshots/<rev>/   │  ──── model file ────────►     │                              │
│   *.litertlm        │                                │ • litert::lm::Engine load    │
└─────────────────────┘                                │ • GPU-first, CPU fallback    │
                                                       │   (unless --force_gpu)       │
                                                       │ • Conversation per request   │
                                                       │ • Multimodal subbackends     │
                                                       │ • MTP / speculative decode   │
                                                       │ • webgpu_sampler_shim        │
                                                       └──────────────┬───────────────┘
                                                                      │
                                                                      ▼
                                                       ┌──────────────────────────────┐
                                                       │   LiteRT-LM runtime (submod) │
                                                       │   Metal / OpenCL / Vulkan /  │
                                                       │   CPU · TFLite delegates     │
                                                       └──────────────────────────────┘
```

## Layers

| Layer | Files | Responsibility |
|------|-------|----------------|
| Entry | [server/main.cc](server/main.cc) | CLI parsing, model resolution, engine construction, server boot, hot-swap factory |
| Transport | [server/rest/openai_api.cc](server/rest/openai_api.cc), [server/rest/openai_api.h](server/rest/openai_api.h) | OpenAI-compatible HTTP routes, SSE streaming, auth, multimodal input parsing |
| Concurrency | [core/engine_manager.h](core/engine_manager.h) | `shared_mutex`-guarded current engine, RAII `Guard`, atomic `Swap` |
| Engine API | [core/engine.h](core/engine.h) | `EngineBase` abstraction (`Generate` / `GenerateStream` / `Warmup`) + DTOs |
| LiteRT impl | [core/litert/llm_engine.h](core/litert/llm_engine.h), [core/litert/llm_engine.cc](core/litert/llm_engine.cc) | Wraps `litert::lm::Engine` + `Conversation`; GPU-first with CPU fallback |
| Model hub | [hub/model_resolver.h](hub/model_resolver.h), [hub/model_resolver.cc](hub/model_resolver.cc) | `$HF_HOME` cache lookup and HTTPS download from huggingface.co |
| Runtime | [LiteRT-LM/](LiteRT-LM/) (submodule) | TFLite + GPU delegates (Metal / OpenCL / Vulkan) |

## Request lifecycle (chat completions)

```
client POST /v1/chat/completions
   │
   ▼
openai_api.cc :  auth ─► parse JSON ─► build ChatMessage[] (+ media)
   │                                              │
   │                                              └─ download/decode image/audio
   ▼
EngineManager::Acquire()  ── shared_lock ──►  Guard{ EngineBase* }
   │
   ├── stream=false ──►  engine.Generate(...)              ──►  JSON response
   │
   └── stream=true  ──►  engine.GenerateStream(.., on_token)
                              │
                              └─►  SSE  "data: {delta}\n\n"  per token
                                   "data: [DONE]\n\n"
```

Multiple requests run concurrently: each holds a shared lock on the manager for
the duration of its inference call. The lock is released when the `Guard` goes
out of scope.

## Model resolution

On startup (and on every `/v1/models/load`) the resolver runs in this order:

1. `--model_path` / explicit `model_path` override → use as-is.
2. Look in `$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/` for a cached
   `*.litertlm` (or the explicit `filename`).
3. Otherwise download from `https://huggingface.co/<repo>/resolve/<rev>/<file>`
   into the HF cache layout, honoring `HF_TOKEN` for gated repos.

The engine itself only ever sees a fully-materialized local path.

## Hot-swap (`POST /v1/models/load`)

```
request {repo_id, filename?, revision?, backend?, multimodal?, mtp?}
   │
   ▼
openai_api → opts.build_engine(req)        (closure built in main.cc)
   │              │
   │              ├─ hub::ResolveModel()   (download if missing)
   │              ├─ LlmEngine::Create()   (apply per-request overrides on top
   │              │                         of startup EngineOptions)
   │              └─ engine->Warmup()
   ▼
EngineManager::Swap(builder)
   │   unique_lock (waits for in-flight readers to drop their Guards)
   │   replace engine_ on success; old engine destroyed when write lock releases
   ▼
new model serves subsequent requests; failure leaves the old engine intact
```

## Engine internals (LiteRT impl)

- Loads the `.litertlm` file once at `Create()`; long-lived `litert::lm::Engine`.
- GPU-first: attempts the configured GPU delegate; on init failure falls back
  to CPU unless `--force_gpu` is set (then hard-fails).
- Per-request `Conversation` object built from `ChatMessage[]`; system/user/
  assistant roles preserved.
- Optional multimodal sub-backends (vision + audio) are off by default to avoid
  per-request `Gemma4DataProcessor` overhead; enabled with `--multimodal`.
- Optional multi-token prediction (`--mtp`) enables speculative decoding with a
  configurable draft-token count.
- `Warmup()` runs a minimal inference at startup to trigger lazy GPU shader /
  sampler initialization so the first real request is fast.

## Concurrency model

- **Reads** (`Generate` / `GenerateStream`) take a shared lock via
  `EngineManager::Acquire()`. Many can run in parallel; LiteRT-LM serializes
  internally where needed.
- **Writes** (`Swap`) take an exclusive lock. The swap blocks until all
  outstanding `Guard`s are released, then atomically replaces the engine. The
  old engine is destroyed when the unique lock goes out of scope.
- The `Guard` holds a raw pointer plus its shared lock, so the engine pointer
  is stable for the guard's lifetime even if a swap is queued.

## Key design points

- **One resident engine, swappable.** Avoids the memory cost of multi-model
  hosting on-device while still allowing runtime model changes.
- **Pluggable backend** through `EngineBase` ([core/engine.h:37](core/engine.h#L37));
  LiteRT is today's only implementation, MLX is a future slot under `core/`.
- **Model resolution decoupled from inference.** The engine just gets a path;
  caching and download live in `hub/`.
- **GPU-first with graceful CPU fallback** unless `--force_gpu` is set
  ([core/litert/llm_engine.h:11](core/litert/llm_engine.h#L11)).
- **OpenAI wire compatibility isolated** in `server/rest/openai_api.cc`, so the
  core engine API stays free of OpenAI-specific shapes.
- **Multimodal normalization at the edge.** Image/audio inputs (path, `file://`,
  http(s), base64, `data:` URI) are decoded in the REST layer into
  `MediaAttachment` before reaching the engine.


curl -X POST http://localhost:8080/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"repo_id":"litert-community/gemma-4-E4B-it-litert-lm"}'