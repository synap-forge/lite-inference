# lite-inference

A cross-platform C++ server that exposes an **OpenAI-compatible REST API** backed
by Google's [**LiteRT-LM**](https://ai.google.dev/edge/litert-lm/cpp) on-device
inference engine.

- **Multi-platform**: macOS, Windows, Linux (and Android via cross-compile).
- **OpenAI Spec**: `POST /v1/chat/completions` (streaming + non-streaming),
  `GET /v1/models`, `GET /health`.
- **GPU first**: requests the GPU backend and transparently falls back to CPU if
  GPU initialization fails.
- **Model loaded at startup**: the `.litertlm` model is loaded once when the
  process boots, so the first request is fast.
- **Hugging Face aware**: resolves the model from `$HF_HOME` using the standard
  HF cache layout, and **downloads it from huggingface.co** into that cache if
  it isn't present yet.

---

## Architecture

```
client ──HTTP──> server/openai_api.cc   (cpp-httplib + nlohmann/json)
                        │
                        ▼
                 server/llm_engine.cc    (wraps litert::lm::Engine + Conversation)
                        │
                        ▼
                   LiteRT-LM runtime      (GPU delegate / CPU)
```

| File | Responsibility |
|------|----------------|
| `server/main.cc` | CLI flags, resolve + load model at startup, start server |
| `server/model_resolver.{h,cc}` | HF cache lookup (`$HF_HOME`) + native HTTPS download |
| `server/llm_engine.{h,cc}` | LiteRT-LM wrapper: model load, GPU-first, generate/stream |
| `server/openai_api.{h,cc}` | OpenAI-compatible routes, SSE streaming, auth |

---

## Prerequisites

- **Git** with submodule support (`git submodule update --init --recursive`)
- **Git LFS** (`git lfs install`) — LiteRT-LM ships prebuilt GPU binaries via LFS.
- **Bazelisk** (`brew install bazelisk`) — auto-manages Bazel 7.6.1.
- Platform toolchain:
  - **macOS**: `xcode-select --install`, `brew install openssl cmake`
  - **Linux**: clang + standard build tools, `apt install libssl-dev cmake`
  - **Windows**: Visual Studio 2022, Python 3.13, Java (`JAVA_HOME`), long paths enabled

---

## Repository layout

LiteRT-LM is included as a **git submodule** at `LiteRT-LM/` (pinned to `v0.12.0`).
After cloning this repo, initialise it once:

```bash
git clone https://github.com/your-org/lite-inference.git
cd lite-inference
git submodule update --init --recursive
git lfs pull --include="LiteRT-LM/**"
```

---

## How it builds

LiteRT-LM manages all its transitive deps (abseil, TFLite, GPU delegates…) from
its own root `WORKSPACE`. The server lives **inside** the LiteRT-LM source tree
as a package (`lite_inference_server/`), so it references LiteRT-LM targets
directly with no extra dep-wrangling.

`setup.sh` handles this:
1. Uses `LiteRT-LM/` from the submodule (or `$LITERT_DIR` override).
2. Copies `server/` into the clone as `lite_inference_server/`.
3. Patches LiteRT-LM's `WORKSPACE` with `cpp-httplib` and `nlohmann/json`
   (real `sha256` values fetched at setup time, not placeholders).

---

## Model resolution

On startup the server resolves the model in this order:

1. `--model_path` if given (explicit override).
2. The Hugging Face cache: `$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/`.
3. Otherwise it **downloads** from `huggingface.co/<repo>` into that cache.

Default repo: `litert-community/gemma-4-E2B-it-litert-lm`.

```bash
export HF_HOME=/Volumes/Sources/AI/huggingface   # or wherever your HF cache is
export HF_TOKEN=hf_xxx                            # only for gated/private repos
```

---

## Build & run

The recommended path builds the engine shared library with Bazel, then builds
the HTTP server with CMake (which links against that `.so`).

### 0. Clone with the submodule

```bash
git clone --recurse-submodules <this-repo>
# or, if already cloned:
git submodule update --init --recursive
```

> The `LiteRT-LM/` submodule points at the fork
> `git@github.com:aminnasiri/LiteRT-LM.git` (branch `lite-inference-embeddings`),
> which carries the embedding C API used by `/v1/embeddings`.

### 1. Run setup once

```bash
cd /path/to/lite-inference
./setup.sh
# Uses LiteRT-LM/ submodule by default. Override with:
#   LITERT_DIR=/my/path ./setup.sh
```

### 2. Build the engine shared library (Bazel)

```bash
cd LiteRT-LM
bazelisk build //lite_inference_server:libLiteRtLmEngine.so

# Stage it where CMake expects a stable copy (macOS arm64 shown):
cp bazel-bin/lite_inference_server/libLiteRtLmEngine.so \
   prebuilt/macos_arm64/libLiteRtLmEngine.so
cd ..
```

> CMake prefers `LiteRT-LM/prebuilt/<platform>/libLiteRtLmEngine.so` if present,
> otherwise falls back to `LiteRT-LM/bazel-bin/.../libLiteRtLmEngine.so`.

### 3. Build the server (CMake)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENSSL_ROOT_DIR=$(brew --prefix openssl) -Wno-dev

cmake --build build -j$(sysctl -n hw.ncpu)
```

The build step copies the required runtime `.dylib`/`.so` files next to
`build/litert_server` and fixes their install names, so the binary is
self-contained — no `DYLD_LIBRARY_PATH` needed.

### 4. Run

```bash
export HF_HOME=/Volumes/Sources/AI/huggingface
export HF_HUB_OFFLINE=1   # optional: skip network checks when models are cached

# Chat only — GPU (Metal) first, CPU fallback:
./build/litert_server \
  --backend auto \
  --port 8080 \
  --hf_repo litert-community/gemma-4-E4B-it-litert-lm

# Chat + embeddings (verified working on macOS arm64, CPU):
./build/litert_server \
  --backend cpu \
  --port 8091 \
  --startup_load partial \
  --hf_repo litert-community/gemma-4-E4B-it-litert-lm \
  --embed_repo litert-community/embeddinggemma-300m \
  --embed_seq_len 512
```

Smoke test:

```bash
curl -s -X POST http://localhost:8091/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"Embedded it","model":"litert-community/embeddinggemma-300m"}' | jq
```

### Pure-Bazel alternative (no embeddings wrapper)

The standalone Bazel server target also exists, but it does **not** go through
the CMake `/v1/embeddings` wiring:

```bash
cd LiteRT-LM
bazelisk build //lite_inference_server:litert_server              # CPU
bazelisk build //lite_inference_server:litert_server --config=gpu # GPU
bazel-bin/lite_inference_server/litert_server --backend=gpu --port=8080
```

### Windows (PowerShell)

```powershell
$Env:HF_HOME = "C:\path\to\huggingface"
.\setup.sh    # run from Git Bash, or do steps manually
cd LiteRT-LM
bazelisk build //lite_inference_server:litert_server --config=windows
bazel-bin\lite_inference_server\litert_server.exe --backend=gpu --port=8080
```

### Android (cross-compile from Linux/macOS)

```bash
export ANDROID_NDK_HOME=/path/to/ndk   # r28b or newer
cd LiteRT-LM
bazelisk build --config=android_arm64 //lite_inference_server:litert_server
```

---

## CLI flags

Run `./build/litert_server --help` for the authoritative list.

**Model resolution**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | *(empty)* | Explicit `.litertlm`/`.tflite` path; overrides HF resolution |
| `--hf_repo` | `litert-community/gemma-4-E2B-it-litert-lm` | HF repo to resolve/download |
| `--hf_file` | *(autodetect)* | File within the repo, e.g. `gemma-4-E2B-it.litertlm` |
| `--hf_revision` | `main` | Repo revision/branch |

**Engine**

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | `auto` (try GPU, fall back to CPU) \| `gpu` \| `cpu` |
| `--force_gpu` | off | Hard-fail if GPU init fails; no CPU fallback |
| `--multimodal` | off | Enable vision + audio sub-backends (CPU) |
| `--mtp [N]` | off / 3 | Multi-token prediction (speculative decoding); optional draft-token count |
| `--context_length N` | `0` | KV-cache window in tokens. `0` = auto-detect from `ekv####` in filename |
| `--startup_load` | `full` | `full` (load + warmup) \| `partial` (load, no warmup) \| `none` (defer to first request) |

**Embeddings** *(optional — enables `/v1/embeddings`)*

| Flag | Default | Description |
|------|---------|-------------|
| `--embed_repo` | *(empty)* | HF repo for the embedding `.tflite` (e.g. `litert-community/embeddinggemma-300m`). Setting this enables `/v1/embeddings` |
| `--embed_file` | *(autodetect)* | Specific `.tflite` within the repo |
| `--embed_seq_len` | `512` | Sequence length the model was built for; must match the `seqNNNN` in the filename (e.g. `512` → `...seq512...tflite`) |

**Server**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind interface |
| `--port` | `8080` | Listen port |
| `--api_key` | *(empty)* | If set, require `Authorization: Bearer <key>` |

Env vars: `HF_HOME` (cache root), `HF_TOKEN` (gated/private repos), `HF_HUB_OFFLINE=1` (skip network checks when models are already cached).

**Notes on embeddings**

- The embedder borrows the LLM engine's tokenizer, so it requires the LLM to be loaded — use `--startup_load full` or `partial` (`none` is rejected with `--embed_repo`).
- The embedding model always runs on **CPU** regardless of `--backend` (it is small; the LLM still uses GPU/Metal when selected).
- `embeddinggemma-300m` ships per-SoC NPU builds (`.qualcomm.*`, `.mediatek.*`, `.google.tensor_g5.*`) **and** generic CPU builds (`...seqNNNN_mixed-precision.tflite`). The resolver automatically prefers the generic CPU build matching `--embed_seq_len`. Output dimension is 768.

---

## Using the API

### Chat completion (non-streaming)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Explain photosynthesis briefly."}],
    "temperature": 0.7
  }'
```

### Chat completion (streaming, SSE)

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a haiku."}],"stream":true}'
```

### Drop-in with the OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### List models / health

```bash
curl http://localhost:8090/v1/models
curl http://localhost:8090/health
```

### Embeddings

Requires the server to be started with `--embed_repo` (see the [Embeddings](#cli-flags) flags).

```bash
curl -s -X POST 'http://localhost:8090/v1/embeddings' \
    -H 'Content-Type: application/json' \
    -d '{"input":"Embedded it","model":"litert-community/embeddinggemma-300m"}' | jq
```



`input` also accepts an array of strings for batch embedding. With the OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="not-needed")
v = client.embeddings.create(
    model="litert-community/embeddinggemma-300m",
    input="Embedded it",
).data[0].embedding
print(len(v))  # 768
```

### Multimodal (image / audio)
```bash
curl -s -N -X POST 'http://localhost:8090/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
            "max_tokens":256,
            "messages":[
                {
                    "content":[
                        {
                            "type":"text",
                            "text":"What is the weather today?"
                        }
                    ],
                    "role":"user"
                }
            ],
            "model":"litert-community/gemma-4-E4B-it-litert-lm",
            "stream":false,
            "temperature":0.7
        }' \
    --max-time 200 | jq
```


```bash
# Image via local path
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "/path/to/photo.jpg"}},
      {"type": "text", "text": "What is in this image?"}
    ]}]
  }'

# Another image
curl -s -N -X POST 'http://localhost:8090/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
            "max_tokens":256,
            "messages":[
                {
                    "content":[
                        {
                            "type":"image_url",
                            "image_url":{"url":"/Volumes/Sources/Documents/Projects/receipts/costco.jpg"}
                        },
                        {
                            "type":"text",
                            "text":"What is in this image?"
                        }
                    ],
                    "role":"user"
                }
            ],
            "model":"litert-community/gemma-4-E4B-it-litert-lm",
            "stream":false,
            "temperature":0.7
        }' \
    --max-time 200 | jq  

# Audio via base64
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": [
      {"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}},
      {"type": "text", "text": "Transcribe this."}
    ]}]
  }'
```

Supported image URL formats: local absolute path, `file://`, HTTP/HTTPS URL, `data:image/...;base64,...`.


## ios build
```bash
cd /Volumes/Sources/Developments/appsrc/lite-inference/LiteRT-LM && \
bazelisk build --config=ios_arm64 --output_filter="" //lite_inference_server:libLiteRtLmEngine.a 2>&1
```

## Android build
```bash
cd /Volumes/Sources/Developments/appsrc/lite-inference/LiteRT-LM && \
bazelisk build --config=android_arm64 --output_filter="" //lite_inference_server:libLiteRtLmEngine.so 2>&1
```

## New Build

cd LiteRT-LM && bazelisk build //lite_inference_server:libLiteRtLmEngine.so
cp bazel-bin/lite_inference_server/libLiteRtLmEngine.so prebuilt/macos_arm64/

cmake --build /Volumes/Sources/Developments/appsrc/lite-inference/build --config Release -j$(sysctl -n hw.logicalcpu) 2>&1 | tail -20

cmake -B /Volumes/Sources/Developments/appsrc/lite-inference/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENSSL_ROOT_DIR=$(brew --prefix openssl) \
  -Wno-dev \
  -S /Volumes/Sources/Developments/appsrc/lite-inference && \
cmake --build /Volumes/Sources/Developments/appsrc/lite-inference/build \
  --config Release -j$(sysctl -n hw.logicalcpu) 2>&1 | tail -30


## Benchmark
```bash
# Basic: 10 requests, 4 concurrent
python3 benchmark.py

# Custom concurrency and count
python3 benchmark.py -c 8 -n 20

# With streaming (measures TTFT)
python3 benchmark.py --stream -c 4 -n 10

# Custom prompt and URL
python3 benchmark.py --url http://localhost:8080 --prompt "Explain gravity in one sentence." -c 2 -n 5

# With API key
python3 benchmark.py --api-key mykey -c 4 -n 10

```


otool -L /Volumes/Sources/Developments/appsrc/lite-inference/build/libLiteRtTopKMetalSampler.dylib | head -5


python3 benchmark.py \ 
  --image /Volumes/Sources/Documents/Projects/receipts/costco.jpg \
  --prompt "What is in this image?" \
  -n 10 -c 10 2>&1


cmake -S . -B build 2>&1 | tail -10 && cmake --build build --target litert_server -j 2>&1 | tail -40

cmake -B build 2>&1 | tail -3 && cmake --build build -j8 2>&1 | tail -10

cmake -B build -GNinja 2>&1 | tail -5 && cmake --build build -j8 2>&1 | tail -10