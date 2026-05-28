#include "core/litert/embedding_engine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "c/engine.h"
#include "core/litert/litert_c_api_min.h"

namespace lite_inference {

struct EmbeddingEngine::Impl {
  LiteRtModel         model    = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  LiteRtOptions       options  = nullptr;
  LiteRtLmEngine*     tokenizer = nullptr;
  int seq_len = 0;
  int dim     = 0;
};

EmbeddingEngine::~EmbeddingEngine() {
  if (!impl_) return;
  if (impl_->compiled) LiteRtDestroyCompiledModel(impl_->compiled);
  if (impl_->options)  LiteRtDestroyOptions(impl_->options);
  if (impl_->model)    LiteRtDestroyModel(impl_->model);
}

std::unique_ptr<EmbeddingEngine> EmbeddingEngine::Create(
    const std::string& model_path,
    const std::string& model_id,
    int seq_len,
    LiteRtLmEngine* tokenizer,
    std::string& error_out) {
  if (!tokenizer) {
    error_out = "EmbeddingEngine requires a loaded LLM engine for tokenization";
    return nullptr;
  }

  auto self = std::unique_ptr<EmbeddingEngine>(new EmbeddingEngine());
  self->impl_ = std::make_unique<Impl>();
  self->impl_->tokenizer = tokenizer;
  self->impl_->seq_len   = seq_len;
  self->model_id_        = model_id;
  self->seq_len_         = seq_len;

  LiteRtStatus st = LiteRtCreateModelFromFile(model_path.c_str(),
                                              &self->impl_->model);
  if (st != kLiteRtStatusOk) {
    error_out = "LiteRtCreateModelFromFile failed (status=" +
                std::to_string(st) + ") for: " + model_path;
    return nullptr;
  }

  st = LiteRtCreateOptions(&self->impl_->options);
  if (st != kLiteRtStatusOk) {
    error_out = "LiteRtCreateOptions failed (status=" + std::to_string(st) + ")";
    return nullptr;
  }

  st = LiteRtCreateCompiledModel(self->impl_->model, nullptr,
                                 self->impl_->options,
                                 &self->impl_->compiled);
  if (st != kLiteRtStatusOk) {
    error_out = "LiteRtCreateCompiledModel failed (status=" +
                std::to_string(st) + ")";
    return nullptr;
  }

  // Probe the output buffer size to determine the embedding dimension.
  // Output shape is [1, dim] float32 → size = dim * 4.
  const LiteRtTensorBufferRequirements* out_req = nullptr;
  st = LiteRtGetCompiledModelOutputBufferRequirements(
      self->impl_->compiled, 0, 0, &out_req);
  if (st != kLiteRtStatusOk || !out_req) {
    error_out = "Failed to get output buffer requirements";
    return nullptr;
  }
  size_t out_bytes = 0;
  LiteRtGetTensorBufferRequirementsBufferSize(out_req, &out_bytes);
  self->impl_->dim = static_cast<int>(out_bytes / sizeof(float));
  self->dim_       = self->impl_->dim;

  fprintf(stderr, "[EmbeddingEngine] Loaded %s  seq=%d  dim=%d\n",
          model_id.c_str(), seq_len, self->dim_);
  return self;
}

std::vector<float> EmbeddingEngine::Embed(const std::string& text,
                                          std::string& error_out) {
  // ---------- Tokenize via borrowed LLM engine tokenizer ----------
  LiteRtLmTokenizeResult* tok =
      litert_lm_engine_tokenize(impl_->tokenizer, text.c_str());
  if (!tok) {
    error_out = "Tokenization failed";
    return {};
  }
  size_t num_tokens = litert_lm_tokenize_result_get_num_tokens(tok);
  const int* token_ids = litert_lm_tokenize_result_get_tokens(tok);

  // Build int32 input padded/truncated to seq_len.
  std::vector<int32_t> input(impl_->seq_len, 0);
  size_t copy_n = std::min(num_tokens, static_cast<size_t>(impl_->seq_len));
  for (size_t i = 0; i < copy_n; ++i)
    input[i] = static_cast<int32_t>(token_ids[i]);
  litert_lm_tokenize_result_delete(tok);

  // ---------- Allocate input buffer ----------
  const LiteRtTensorBufferRequirements* in_req = nullptr;
  LiteRtStatus st = LiteRtGetCompiledModelInputBufferRequirements(
      impl_->compiled, 0, 0, &in_req);
  if (st != kLiteRtStatusOk || !in_req) {
    error_out = "Failed to get input buffer requirements";
    return {};
  }

  LiteRtTensorBuffer in_buf = nullptr;
  st = LiteRtCreateManagedTensorBuffer(kLiteRtTensorBufferTypeHostMemory,
                                       in_req, &in_buf);
  if (st != kLiteRtStatusOk || !in_buf) {
    error_out = "Failed to create input tensor buffer";
    return {};
  }

  // ---------- Allocate output buffer ----------
  const LiteRtTensorBufferRequirements* out_req = nullptr;
  st = LiteRtGetCompiledModelOutputBufferRequirements(
      impl_->compiled, 0, 0, &out_req);
  if (st != kLiteRtStatusOk || !out_req) {
    LiteRtDestroyTensorBuffer(in_buf);
    error_out = "Failed to get output buffer requirements";
    return {};
  }

  LiteRtTensorBuffer out_buf = nullptr;
  st = LiteRtCreateManagedTensorBuffer(kLiteRtTensorBufferTypeHostMemory,
                                       out_req, &out_buf);
  if (st != kLiteRtStatusOk || !out_buf) {
    LiteRtDestroyTensorBuffer(in_buf);
    error_out = "Failed to create output tensor buffer";
    return {};
  }

  // ---------- Write tokens into input buffer ----------
  void* in_ptr = nullptr;
  st = LiteRtLockTensorBuffer(in_buf, &in_ptr, nullptr);
  if (st != kLiteRtStatusOk || !in_ptr) {
    LiteRtDestroyTensorBuffer(in_buf);
    LiteRtDestroyTensorBuffer(out_buf);
    error_out = "Failed to lock input tensor buffer";
    return {};
  }
  std::memcpy(in_ptr, input.data(), input.size() * sizeof(int32_t));
  LiteRtUnlockTensorBuffer(in_buf);

  // ---------- Run ----------
  st = LiteRtRunCompiledModel(impl_->compiled, 0, 1, &in_buf, 1, &out_buf);
  if (st != kLiteRtStatusOk) {
    LiteRtDestroyTensorBuffer(in_buf);
    LiteRtDestroyTensorBuffer(out_buf);
    error_out = "LiteRtRunCompiledModel failed (status=" +
                std::to_string(st) + ")";
    return {};
  }

  // ---------- Read output ----------
  void* out_ptr = nullptr;
  st = LiteRtLockTensorBuffer(out_buf, &out_ptr, nullptr);
  if (st != kLiteRtStatusOk || !out_ptr) {
    LiteRtDestroyTensorBuffer(in_buf);
    LiteRtDestroyTensorBuffer(out_buf);
    error_out = "Failed to lock output tensor buffer";
    return {};
  }

  std::vector<float> embedding(impl_->dim);
  std::memcpy(embedding.data(), out_ptr, impl_->dim * sizeof(float));
  LiteRtUnlockTensorBuffer(out_buf);

  LiteRtDestroyTensorBuffer(in_buf);
  LiteRtDestroyTensorBuffer(out_buf);

  // ---------- L2 normalize ----------
  float norm = 0.0f;
  for (float v : embedding) norm += v * v;
  norm = std::sqrt(norm);
  if (norm > 1e-9f)
    for (float& v : embedding) v /= norm;

  return embedding;
}

}  // namespace lite_inference
