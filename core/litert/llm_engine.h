#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/engine.h"

namespace lite_inference {

struct EngineOptions {
  std::string backend = "gpu"; // "gpu" | "cpu"
  bool multimodal     = false; // enable vision + audio sub-backends (cpu)
  bool mtp            = false; // enable speculative decoding (multi-token prediction)
  bool force_gpu      = false; // hard-fail if GPU init fails; no CPU fallback
};

// LiteRT-LM backend. Model is loaded once at Create() time.
class LlmEngine : public EngineBase {
 public:
  // model_id: display name exposed via the API (e.g. HF repo id).
  // If empty, derived from model_path filename.
  static std::unique_ptr<LlmEngine> Create(const std::string& model_path,
                                           const std::string& model_id,
                                           const EngineOptions& opts,
                                           std::string& error_out);

  ~LlmEngine() override;

  std::string Generate(const std::vector<ChatMessage>& messages,
                       const GenerationParams& params,
                       std::string& error_out) override;

  bool GenerateStream(const std::vector<ChatMessage>& messages,
                      const GenerationParams& params,
                      const TokenCallback& on_token,
                      std::string& error_out) override;

  // Runs a minimal inference to trigger lazy GPU/sampler initialization so the
  // first real request doesn't pay the cold-start cost.
  void Warmup() override;

  const std::string& model_id()       const override { return model_id_; }
  const std::string& active_backend() const override { return active_backend_; }

 private:
  LlmEngine() = default;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_id_;
  std::string active_backend_;
};

}  // namespace lite_inference
