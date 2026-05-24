#pragma once
#include <string>

namespace lite_inference {

struct ResolveOptions {
  std::string model_path;   // explicit override; skips HF resolution if set
  std::string repo_id     = "litert-community/gemma-4-E2B-it-litert-lm";
  std::string filename;     // empty = autodetect *.litertlm
  std::string revision    = "main";
};

// Returns the absolute path to the model file, downloading if necessary.
// On failure returns "" and sets error_out.
std::string ResolveModel(const ResolveOptions& options, std::string& error_out);

std::string HfHubDir();

}  // namespace lite_inference
