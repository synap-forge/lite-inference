#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lite_inference {

enum class MediaType { kImage, kAudio };

struct MediaAttachment {
  MediaType            type;
  std::string          path;  // set for file:// / local / http paths (after download)
  std::vector<uint8_t> data;  // set for base64-decoded inline data
};

struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant"
  std::string content;
  std::vector<MediaAttachment> media;
};

struct GenerationParams {
  float temperature         = 1.0f;
  float top_p               = 1.0f;
  int   top_k               = 0;    // 0 = engine default
  int   max_tokens          = 0;    // 0 = engine default
  int   visual_token_budget = 256;
};

// Return false to cancel streaming.
using TokenCallback = std::function<bool(const std::string& delta)>;

// Abstract inference backend. Concrete implementations live in core/litert/
// and (future) core/mlx/.
class EngineBase {
 public:
  virtual ~EngineBase() = default;

  virtual std::string Generate(const std::vector<ChatMessage>& messages,
                               const GenerationParams& params,
                               std::string& error_out) = 0;

  virtual bool GenerateStream(const std::vector<ChatMessage>& messages,
                              const GenerationParams& params,
                              const TokenCallback& on_token,
                              std::string& error_out) = 0;

  // Runs a tiny inference to surface init errors (GPU sampler unavailable,
  // missing bundle sections, etc.) before the engine starts taking real
  // traffic. Returns true on success; on failure, `error_out` carries the
  // engine-side message and the engine should not be used.
  virtual bool Warmup(std::string& error_out) = 0;

  virtual const std::string& model_id()       const = 0;
  virtual const std::string& active_backend() const = 0;
  virtual bool                multimodal()    const = 0;
  virtual bool                mtp()           const = 0;
  virtual size_t              context_length() const = 0;  // 0 = unknown
};

}  // namespace lite_inference
