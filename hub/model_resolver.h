#pragma once
#include <string>
#include <vector>

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

struct CachedModel {
  std::string repo_id;      // e.g. "litert-community/gemma-4-E2B-it-litert-lm"
  std::string revision;     // e.g. "main"
  std::string filename;     // e.g. "gemma-4-E2B-it.litertlm"
  std::string path;         // absolute path to the model file
};

// Lists every fully-downloaded model file under HfHubDir(). Excludes in-flight
// downloads (.part / .aria2 control files).
std::vector<CachedModel> ListCachedModels();

}  // namespace lite_inference
