#include "core/litert/embedding_engine.h"

#include <memory>
#include <string>
#include <vector>

#include "c/engine.h"

namespace lite_inference {

struct EmbeddingEngine::Impl {
  LiteRtLmEmbedder* embedder = nullptr;
};

EmbeddingEngine::~EmbeddingEngine() {
  if (impl_ && impl_->embedder) {
    litert_lm_embedder_delete(impl_->embedder);
  }
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

  // The embedder is compiled inside the engine .so, where the litert C++
  // Environment is available and reused. This is the only place a standalone
  // .tflite can be compiled against a non-null environment.
  LiteRtLmEmbedder* embedder =
      litert_lm_engine_create_embedder(tokenizer, model_path.c_str(), seq_len);
  if (!embedder) {
    error_out = "litert_lm_engine_create_embedder failed for: " + model_path +
                " (see engine logs for details)";
    return nullptr;
  }

  auto self = std::unique_ptr<EmbeddingEngine>(new EmbeddingEngine());
  self->impl_ = std::make_unique<Impl>();
  self->impl_->embedder = embedder;
  self->model_id_ = model_id;
  self->seq_len_  = seq_len;
  self->dim_      = litert_lm_embedder_get_dim(embedder);
  return self;
}

std::vector<float> EmbeddingEngine::Embed(const std::string& text,
                                          std::string& error_out) {
  if (!impl_ || !impl_->embedder) {
    error_out = "Embedding engine not initialized";
    return {};
  }

  LiteRtLmEmbedResult* result =
      litert_lm_embedder_embed(impl_->embedder, text.c_str());
  if (!result) {
    error_out = "litert_lm_embedder_embed failed (see engine logs for details)";
    return {};
  }

  const float* values = litert_lm_embed_result_get_values(result);
  size_t n = litert_lm_embed_result_get_num_floats(result);
  std::vector<float> embedding(values, values + n);
  litert_lm_embed_result_delete(result);
  return embedding;
}

}  // namespace lite_inference
