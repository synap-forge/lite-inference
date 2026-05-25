#include "hub/model_resolver.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace lite_inference {
namespace fs = std::filesystem;

namespace {

std::string EnvOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::string(v) : fallback;
}

std::string HomeDir() {
  return EnvOr("HOME", EnvOr("USERPROFILE", "."));
}

// "org/repo-name" → "models--org--repo-name"
std::string RepoFolder(const std::string& repo_id) {
  std::string out = "models--";
  for (size_t i = 0; i < repo_id.size(); ++i) {
    if (repo_id[i] == '/') out += "--";
    else out += repo_id[i];
  }
  return out;
}

std::string ReadRef(const fs::path& repo_dir, const std::string& revision) {
  std::error_code ec;
  fs::path ref = repo_dir / "refs" / revision;
  if (!fs::exists(ref, ec)) return "";
  std::ifstream in(ref);
  std::string rev;
  std::getline(in, rev);
  return rev;
}

std::string FindModelFile(const fs::path& dir) {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    auto ext = e.path().extension();
    if (ext == ".litertlm" || ext == ".tflite") return e.path().string();
  }
  return "";
}

std::string FindCached(const fs::path& repo_dir, const std::string& revision,
                       const std::string& filename) {
  std::error_code ec;
  std::vector<fs::path> candidates;
  std::string rev = ReadRef(repo_dir, revision);
  if (!rev.empty()) candidates.push_back(repo_dir / "snapshots" / rev);
  fs::path snaps = repo_dir / "snapshots";
  if (fs::is_directory(snaps, ec))
    for (const auto& d : fs::directory_iterator(snaps, ec))
      if (d.is_directory()) candidates.push_back(d.path());
  for (const auto& snap : candidates) {
    if (!filename.empty()) {
      fs::path f = snap / filename;
      if (fs::exists(f, ec)) return f.string();
    } else {
      std::string found = FindModelFile(snap);
      if (!found.empty()) return found;
    }
  }
  return "";
}

// Downloads via the `curl` CLI which handles HF's redirect chain (huggingface.co
// → xethub CDN with pre-signed URLs) correctly out of the box, including
// proper TLS/SNI handling that httplib struggles with on the xethub CDN.
bool HttpsDownload(const std::string& repo_id, const std::string& revision,
                   const std::string& filename, const fs::path& dest,
                   std::string& error_out) {
  const std::string url = "https://huggingface.co/" + repo_id +
                          "/resolve/" + revision + "/" + filename;

  std::error_code ec;
  fs::create_directories(dest.parent_path(), ec);
  fs::path tmp = dest.string() + ".part";

  fprintf(stderr, "Downloading %s ...\n", url.c_str());

  std::string cmd = "curl --location --fail --progress-bar"
                    " --output " + std::string(tmp) +
                    " \"" + url + "\"";
  const char* token = std::getenv("HF_TOKEN");
  if (token && *token)
    cmd = "curl --location --fail --progress-bar"
          " --header \"Authorization: Bearer " + std::string(token) + "\""
          " --output " + std::string(tmp) +
          " \"" + url + "\"";

  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    fs::remove(tmp, ec);
    error_out = "curl failed (exit " + std::to_string(rc) + ") for " + url +
                (rc == 22 ? " (HTTP error — check HF_TOKEN for gated repos)" : "");
    return false;
  }

  fs::rename(tmp, dest, ec);
  if (ec) { error_out = "Cannot rename " + tmp.string(); return false; }
  fprintf(stderr, "Saved to %s\n", dest.string().c_str());
  return true;
}

}  // namespace

std::string HfHubDir() {
  return EnvOr("HF_HOME", HomeDir() + "/.cache/huggingface") + "/hub";
}

std::string ResolveModel(const ResolveOptions& options, std::string& error_out) {
  // 1. Explicit path.
  if (!options.model_path.empty()) {
    std::error_code ec;
    if (!fs::exists(options.model_path, ec)) {
      error_out = "--model_path not found: " + options.model_path;
      return {};
    }
    return options.model_path;
  }

  if (options.repo_id.empty()) {
    error_out = "Provide --model_path or --hf_repo";
    return {};
  }

  fs::path repo_dir = fs::path(HfHubDir()) / RepoFolder(options.repo_id);

  // 2. Check cache.
  std::string cached = FindCached(repo_dir, options.revision, options.filename);
  if (!cached.empty()) {
    fprintf(stderr, "Using cached model: %s\n", cached.c_str());
    return cached;
  }

  // 3. Download.
  std::string filename = options.filename;
  if (filename.empty()) {
    std::string name = options.repo_id;
    auto slash = name.rfind('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    const std::string suffix = "-litert-lm";
    if (name.size() > suffix.size() &&
        name.substr(name.size() - suffix.size()) == suffix) {
      name = name.substr(0, name.size() - suffix.size());
      filename = name + ".litertlm";
      fprintf(stderr, "No --hf_file given; trying %s\n", filename.c_str());
    } else {
      error_out =
          "Cannot derive filename from repo '" + options.repo_id +
          "'. Pass --hf_file with the exact filename (e.g. "
          "DeepSeek-R1-Distill-Qwen-1.5B_seq128_f32_ekv1280.tflite).";
      return {};
    }
  }

  fs::path dest = repo_dir / "snapshots" / options.revision / filename;
  if (!HttpsDownload(options.repo_id, options.revision, filename, dest, error_out))
    return {};
  return dest.string();
}

}  // namespace lite_inference
