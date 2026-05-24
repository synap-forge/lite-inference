#pragma once

#include <stddef.h>
#include <stdint.h>

// C ABI for Dart FFI / Android JNI / iOS.
// All strings are UTF-8, null-terminated. Caller owns nothing returned by
// value; pointers returned from Create() must be freed with the matching
// Destroy() call.

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Export macro
//
// The library is built with hidden default visibility (CXX_VISIBILITY_PRESET
// hidden) so internal C++/LiteRT-LM symbols don't leak and clash with other
// native libs in a Flutter app. Public C API functions must opt back in to
// default visibility or they won't be resolvable via dlsym/Dart FFI.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
  #define LITE_INFERENCE_EXPORT __declspec(dllexport)
#else
  #define LITE_INFERENCE_EXPORT __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
typedef struct LiteInferenceEngine LiteInferenceEngine;

// ---------------------------------------------------------------------------
// Options passed to LiteInferenceEngineCreate()
// ---------------------------------------------------------------------------
typedef struct {
  const char* model_path;       // Absolute path to .litertlm / .tflite file
  const char* backend;          // "gpu" | "cpu" | "auto"  (default: "auto")
  int         force_gpu;        // 1 = abort if GPU unavailable, 0 = allow CPU fallback
  int         multimodal;       // 1 = enable vision + audio backends
  int         mtp;              // 1 = enable multi-token prediction / speculative decoding
} LiteInferenceOptions;

// ---------------------------------------------------------------------------
// Generation parameters
// ---------------------------------------------------------------------------
typedef struct {
  float temperature;            // 0.0–2.0, default 1.0
  float top_p;                  // 0.0–1.0, default 1.0
  int   top_k;                  // 0 = engine default
  int   max_tokens;             // 0 = engine default
  int   visual_token_budget;    // default 256
} LiteInferenceGenParams;

// ---------------------------------------------------------------------------
// Message role constants
// ---------------------------------------------------------------------------
#define LITE_INFERENCE_ROLE_SYSTEM    "system"
#define LITE_INFERENCE_ROLE_USER      "user"
#define LITE_INFERENCE_ROLE_ASSISTANT "assistant"

// ---------------------------------------------------------------------------
// Streaming callback
// Return 0 to cancel generation, non-zero to continue.
// `delta`     — null-terminated token text
// `user_data` — opaque pointer passed to LiteInferenceGenerateStream()
// ---------------------------------------------------------------------------
typedef int (*LiteInferenceTokenCallback)(const char* delta, void* user_data);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create and load an inference engine. Returns NULL on failure; call
// LiteInferenceGetError() (before any other call) to retrieve the message.
LITE_INFERENCE_EXPORT
LiteInferenceEngine* LiteInferenceEngineCreate(const LiteInferenceOptions* opts);

// Destroy engine and free all associated resources.
LITE_INFERENCE_EXPORT
void LiteInferenceEngineDestroy(LiteInferenceEngine* engine);

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
const char* LiteInferenceModelId(const LiteInferenceEngine* engine);
LITE_INFERENCE_EXPORT
const char* LiteInferenceActiveBackend(const LiteInferenceEngine* engine);

// Warm up the engine (optional; runs a short generation to prime GPU pipelines).
LITE_INFERENCE_EXPORT
void LiteInferenceWarmup(LiteInferenceEngine* engine);

// ---------------------------------------------------------------------------
// Inference — single-turn, blocking
//
// `roles`    — array of `n_messages` role strings
// `contents` — array of `n_messages` content strings
// Returns heap-allocated UTF-8 response; caller must free with LiteInferenceFreeString().
// Returns NULL on error; call LiteInferenceGetError() for details.
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
char* LiteInferenceGenerate(
    LiteInferenceEngine*         engine,
    const char* const*           roles,
    const char* const*           contents,
    int                          n_messages,
    const LiteInferenceGenParams* params);

// ---------------------------------------------------------------------------
// Inference — streaming (callback)
//
// Calls `callback` for each token delta. Returns 1 on success, 0 on error.
// NOTE: the callback is invoked from LiteRT-LM's internal engine thread, not
// the caller's thread. From Dart, prefer LiteInferenceGenerateStreamPort()
// instead — Dart FFI callbacks cannot be invoked from arbitrary native threads.
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
int LiteInferenceGenerateStream(
    LiteInferenceEngine*          engine,
    const char* const*            roles,
    const char* const*            contents,
    int                           n_messages,
    const LiteInferenceGenParams* params,
    LiteInferenceTokenCallback    callback,
    void*                         user_data);

// ---------------------------------------------------------------------------
// Dart DL API initialisation
//
// Must be called once (per process) with NativeApi.initializeApiDLData before
// using LiteInferenceGenerateStreamPort(). Returns 0 on success.
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
int LiteInferenceInitDartApi(void* data);

// ---------------------------------------------------------------------------
// Inference — streaming (Dart native port)
//
// Streams token deltas to a Dart SendPort via Dart_PostCObject. This is
// thread-safe: LiteRT-LM may invoke its internal callback from any thread, and
// posting to a port is safe from any thread. Each token is posted as a Dart
// String. When generation completes, an empty-string sentinel "" is posted,
// followed by either nothing (success) or the error is retrievable via
// LiteInferenceGetError() — callers should check the return value.
//
// Blocks until generation finishes. Returns 1 on success, 0 on error.
// `send_port` is the SendPort.nativePort integer.
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
int LiteInferenceGenerateStreamPort(
    LiteInferenceEngine*          engine,
    const char* const*            roles,
    const char* const*            contents,
    int                           n_messages,
    const LiteInferenceGenParams* params,
    int64_t                       send_port);

// ---------------------------------------------------------------------------
// Error handling
//
// Returns the last error message for this engine (or the global create error
// when engine is NULL). The string is valid until the next API call on this
// engine. Never NULL — returns "" when no error.
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
const char* LiteInferenceGetError(const LiteInferenceEngine* engine);

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
LITE_INFERENCE_EXPORT
void LiteInferenceFreeString(char* str);

#ifdef __cplusplus
}  // extern "C"
#endif
