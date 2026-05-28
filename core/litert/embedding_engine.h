#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward-declare the LiteRT-LM engine opaque type so we can borrow its
// tokenizer without pulling in the full C API header.
struct LiteRtLmEngine;

namespace lite_inference {

// Runs a TFLite embedding model (e.g. embeddinggemma-300M) using the LiteRT
// CompiledModel C API loaded at runtime via dlopen. Tokenization is delegated
// to a borrowed LiteRtLmEngine* (must outlive this object).
//
// Expected model I/O (embeddinggemma-300M):
//   Input  [1, seq_len]  int32   — token IDs, zero-padded
//   Output [1, dim]      float32 — L2-normalized embedding vector
class EmbeddingEngine {
 public:
  // model_path: path to the .tflite file
  // model_id:   display name for the API
  // seq_len:    context length the model was built for (matches filename, e.g. 1024)
  // tokenizer:  borrowed pointer to an already-created LiteRtLmEngine whose
  //             tokenizer will be used; must remain valid for the lifetime of
  //             this EmbeddingEngine.
  static std::unique_ptr<EmbeddingEngine> Create(
      const std::string& model_path,
      const std::string& model_id,
      int seq_len,
      LiteRtLmEngine* tokenizer,
      std::string& error_out);

  ~EmbeddingEngine();

  // Embed a single text string. Returns an L2-normalized float vector.
  // Returns empty vector on failure; error_out carries the message.
  std::vector<float> Embed(const std::string& text, std::string& error_out);

  const std::string& model_id() const { return model_id_; }
  int                dim()       const { return dim_; }
  int                seq_len()   const { return seq_len_; }

 private:
  EmbeddingEngine() = default;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_id_;
  int dim_     = 0;
  int seq_len_ = 0;
};

}  // namespace lite_inference
