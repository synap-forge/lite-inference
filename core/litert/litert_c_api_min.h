#pragma once
// Minimal forward declarations for the LiteRt C API needed by EmbeddingEngine.
// Derived from the public LiteRt API (google-ai-edge/LiteRT on GitHub).
// We declare only what we use so we avoid a hard dependency on LiteRt headers
// that are not included in the prebuilt distribution.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// Opaque types
// --------------------------------------------------------------------------
typedef struct LiteRtModelT*                LiteRtModel;
typedef struct LiteRtCompiledModelT*        LiteRtCompiledModel;
typedef struct LiteRtEnvironmentT*          LiteRtEnvironment;
typedef struct LiteRtTensorBufferT*         LiteRtTensorBuffer;
typedef struct LiteRtTensorBufferRequirementsT* LiteRtTensorBufferRequirements;
typedef struct LiteRtOptionsT*              LiteRtOptions;

// --------------------------------------------------------------------------
// Status
// --------------------------------------------------------------------------
typedef int32_t LiteRtStatus;
static const LiteRtStatus kLiteRtStatusOk = 0;

// --------------------------------------------------------------------------
// TensorBufferType
// --------------------------------------------------------------------------
typedef enum {
  kLiteRtTensorBufferTypeUnknown    = 0,
  kLiteRtTensorBufferTypeHostMemory = 1,
  // Others exist but we only use HostMemory for CPU inference.
} LiteRtTensorBufferType;

// --------------------------------------------------------------------------
// Model
// --------------------------------------------------------------------------
#if defined(_WIN32)
#define LITERT_C_MIN_EXPORT __declspec(dllimport)
#else
#define LITERT_C_MIN_EXPORT
#endif

LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtCreateModelFromFile(const char* path, LiteRtModel* model);

LITERT_C_MIN_EXPORT
void LiteRtDestroyModel(LiteRtModel model);

// --------------------------------------------------------------------------
// Options (for CreateCompiledModel)
// --------------------------------------------------------------------------
LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options);

LITERT_C_MIN_EXPORT
void LiteRtDestroyOptions(LiteRtOptions options);

// --------------------------------------------------------------------------
// CompiledModel
// --------------------------------------------------------------------------
LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtCreateCompiledModel(LiteRtModel model,
                                       LiteRtEnvironment env,
                                       LiteRtOptions options,
                                       LiteRtCompiledModel* compiled_model);

LITERT_C_MIN_EXPORT
void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled_model);

LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel compiled_model,
    int signature_index,
    int input_index,
    const LiteRtTensorBufferRequirements** requirements);

LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model,
    int signature_index,
    int output_index,
    const LiteRtTensorBufferRequirements** requirements);

// --------------------------------------------------------------------------
// TensorBufferRequirements
// --------------------------------------------------------------------------
LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtGetTensorBufferRequirementsBufferSize(
    const LiteRtTensorBufferRequirements* requirements,
    size_t* buffer_size);

// --------------------------------------------------------------------------
// TensorBuffer
// --------------------------------------------------------------------------
LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtTensorBufferType buffer_type,
    const LiteRtTensorBufferRequirements* requirements,
    LiteRtTensorBuffer* tensor_buffer);

LITERT_C_MIN_EXPORT
void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer tensor_buffer);

LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                    void** host_mem_addr,
                                    LiteRtTensorBuffer event);

LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer tensor_buffer);

// --------------------------------------------------------------------------
// RunCompiledModel
// --------------------------------------------------------------------------
LITERT_C_MIN_EXPORT
LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel compiled_model,
                                    int signature_index,
                                    size_t num_inputs,
                                    LiteRtTensorBuffer* input_buffers,
                                    size_t num_outputs,
                                    LiteRtTensorBuffer* output_buffers);

#ifdef __cplusplus
}
#endif
