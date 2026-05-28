#include "hub/model_resolver.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "nlohmann/json.hpp"

namespace lite_inference {
namespace fs = std::filesystem;

namespace {

int ScoreFile(const std::string& name);  // defined below

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
  std::string best;
  int best_score = -1;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    int s = ScoreFile(e.path().filename().string());
    if (s > best_score) { best_score = s; best = e.path().string(); }
  }
  return best;
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

// Lists every file in `repo_id` at `revision` via the public HF API. Returns
// an empty vector on failure (and sets `error_out`).
std::vector<std::string> HfListFiles(const std::string& repo_id,
                                     const std::string& revision,
                                     std::string& error_out) {
  std::vector<std::string> out;
  const std::string url = "https://huggingface.co/api/models/" + repo_id +
                          "/tree/" + revision + "?recursive=true";
  std::string tmp = "/tmp/litert_hf_list_" + std::to_string(getpid()) + ".json";

  std::string cmd = "curl --silent --show-error --fail --location"
                    " --connect-timeout 15 --max-time 30"
                    " --output " + tmp + " \"" + url + "\"";
  const char* token = std::getenv("HF_TOKEN");
  if (token && *token)
    cmd = "curl --silent --show-error --fail --location"
          " --connect-timeout 15 --max-time 30"
          " --header \"Authorization: Bearer " + std::string(token) + "\""
          " --output " + tmp + " \"" + url + "\"";

  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::error_code ec; fs::remove(tmp, ec);
    error_out = "HF API list failed (curl exit " + std::to_string(rc) +
                ") for " + repo_id;
    return out;
  }

  std::ifstream in(tmp);
  std::string body((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  std::error_code ec; fs::remove(tmp, ec);

  try {
    auto j = nlohmann::json::parse(body);
    if (!j.is_array()) {
      error_out = "HF API returned non-array for " + repo_id;
      return out;
    }
    for (const auto& entry : j) {
      if (entry.value("type", std::string{}) != "file") continue;
      out.push_back(entry.value("path", std::string{}));
    }
  } catch (const std::exception& e) {
    error_out = std::string("Failed to parse HF API response: ") + e.what();
    return {};
  }
  return out;
}

// Scores a filename so a higher number = better choice. The chosen file is
// the one the engine is most likely to load successfully and run well on a
// generic CPU/GPU host. Heuristics, in order of weight:
//   - extension: .litertlm (10000) > .tflite (5000) > others (0)
//   - quantization in name: q8 > q4 > f32/f16 (we prefer accuracy when small)
//   - context size (ekv####): larger is better, up to a cap
//   - penalty for hardware-specific suffixes (mt6989, sm8550, Tensor_G5, etc.)
//   - penalty for "_web" variants
int ScoreFile(const std::string& name) {
  auto ends_with = [&](const char* s) {
    size_t n = std::strlen(s);
    return name.size() >= n &&
           name.compare(name.size() - n, n, s) == 0;
  };
  auto contains = [&](const char* s) {
    return name.find(s) != std::string::npos;
  };

  int score = 0;
  if (ends_with(".litertlm")) score += 10000;
  else if (ends_with(".tflite")) score += 5000;
  else return -1;  // Anything else (.task, .gitattributes, README) is rejected.

  // Quantization preference.
  if (contains("q8")) score += 300;
  else if (contains("q4")) score += 200;
  else if (contains("int4")) score += 150;
  else if (contains("int8")) score += 250;
  else if (contains("f32")) score += 100;
  else if (contains("f16")) score += 120;

  // Context length: pull the integer after "ekv". Larger context = better.
  auto pos = name.find("ekv");
  if (pos != std::string::npos) {
    int ekv = 0;
    for (size_t i = pos + 3; i < name.size() && std::isdigit((unsigned char)name[i]); ++i)
      ekv = ekv * 10 + (name[i] - '0');
    if (ekv > 131072) ekv = 131072;  // cap at 128K (Gemma 3n E2B/E4B max ctx).
    score += ekv / 64;                // ekv1280 -> 20, ekv4096 -> 64, ekv131072 -> 2048.
  }

  // Hardware-specific builds: should only be picked if nothing else exists.
  // SoC tags we've seen in litert-community: mt6989/mt6991/mt6993 (MediaTek),
  // sm8550/sm8650/sm8750/sm8850 (Qualcomm Snapdragon), Tensor_G5 (Pixel).
  const char* hw_tags[] = {
    "mt6989", "mt6991", "mt6993", "mt6995",
    "sm8550", "sm8650", "sm8750", "sm8850",
    "Tensor_G5", "Tensor_G4",
  };
  for (const char* tag : hw_tags) {
    if (contains(tag)) { score -= 2000; break; }
  }

  if (contains("_web") || contains("-web")) score -= 500;

  return score;
}

// Picks the highest-scoring supported file from `files`. Returns "" if none
// of the candidates are supported (.litertlm / .tflite).
std::string PickBestFile(const std::vector<std::string>& files) {
  std::string best;
  int best_score = -1;
  for (const auto& f : files) {
    // Skip files in subdirectories - the engine expects a single top-level
    // model file and the cache layout doesn't preserve subdir structure.
    if (f.find('/') != std::string::npos) continue;
    int s = ScoreFile(f);
    if (s > best_score) { best_score = s; best = f; }
  }
  return best;
}

bool HasCommand(const std::string& name) {
  // `command -v` is POSIX; exit 0 iff resolvable. >/dev/null silences output.
  std::string probe = "command -v " + name + " >/dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

// aria2c: parallel ranged GETs (-x N connections per host, -s N splits per
// file). Native resume via the .aria2 control file. Much faster on HF's
// xethub CDN than single-stream curl, and more resilient to mid-stream drops.
bool Aria2Download(const std::string& url, const fs::path& tmp,
                   std::string& error_out) {
  const std::string base =
      "aria2c --continue=true --auto-file-renaming=false"
      " --max-connection-per-server=16 --split=16 --min-split-size=1M"
      " --max-tries=0 --retry-wait=5"
      " --connect-timeout=30 --timeout=60"
      " --lowest-speed-limit=1K"
      " --console-log-level=warn --summary-interval=5";

  std::string cmd = base +
                    " --dir=\"" + tmp.parent_path().string() + "\""
                    " --out=\"" + tmp.filename().string() + "\""
                    " \"" + url + "\"";
  const char* token = std::getenv("HF_TOKEN");
  if (token && *token)
    cmd = base +
          " --header=\"Authorization: Bearer " + std::string(token) + "\""
          " --dir=\"" + tmp.parent_path().string() + "\""
          " --out=\"" + tmp.filename().string() + "\""
          " \"" + url + "\"";

  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    error_out = "aria2c failed (exit " + std::to_string(rc) + ") for " + url;
    return false;
  }
  return true;
}

// Curl fallback for environments where aria2c isn't available (e.g. local dev
// builds outside the container). Single-stream but with aggressive resume.
bool CurlDownload(const std::string& url, const fs::path& tmp,
                  std::string& error_out) {
  const std::string base_flags =
      "curl --location --fail --progress-bar"
      " --continue-at -"
      " --retry 50 --retry-delay 5 --retry-all-errors"
      " --retry-max-time 0"
      " --connect-timeout 30"
      " --speed-limit 1024 --speed-time 20";

  std::string cmd = base_flags +
                    " --output " + std::string(tmp) +
                    " \"" + url + "\"";
  const char* token = std::getenv("HF_TOKEN");
  if (token && *token)
    cmd = base_flags +
          " --header \"Authorization: Bearer " + std::string(token) + "\""
          " --output " + std::string(tmp) +
          " \"" + url + "\"";

  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    error_out = "curl failed (exit " + std::to_string(rc) + ") for " + url +
                (rc == 22 ? " (HTTP error — check HF_TOKEN for gated repos)" : "");
    return false;
  }
  return true;
}

// Resolves HF's redirect chain (huggingface.co → xethub CDN with pre-signed
// URLs). Prefers aria2c for parallel ranged downloads; falls back to curl.
bool HttpsDownload(const std::string& repo_id, const std::string& revision,
                   const std::string& filename, const fs::path& dest,
                   std::string& error_out) {
  const std::string url = "https://huggingface.co/" + repo_id +
                          "/resolve/" + revision + "/" + filename;

  std::error_code ec;
  fs::create_directories(dest.parent_path(), ec);
  fs::path tmp = dest.string() + ".part";

  fprintf(stderr, "Downloading %s ...\n", url.c_str());

  bool ok = HasCommand("aria2c")
                ? Aria2Download(url, tmp, error_out)
                : CurlDownload(url, tmp, error_out);
  // Keep .part on failure so the next run can resume.
  if (!ok) return false;

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
    std::string list_err;
    auto files = HfListFiles(options.repo_id, options.revision, list_err);
    if (files.empty()) {
      error_out = "Could not list files in '" + options.repo_id + "': " +
                  list_err;
      return {};
    }
    filename = PickBestFile(files);
    if (filename.empty()) {
      error_out = "No supported model file (.litertlm or .tflite) in '" +
                  options.repo_id + "'. The repo only contains other formats "
                  "(e.g. .task) which this engine does not load.";
      return {};
    }
    fprintf(stderr, "Auto-selected file from %s: %s\n",
            options.repo_id.c_str(), filename.c_str());
  }

  fs::path dest = repo_dir / "snapshots" / options.revision / filename;
  if (!HttpsDownload(options.repo_id, options.revision, filename, dest, error_out))
    return {};
  return dest.string();
}

// Inverse of RepoFolder(): "models--org--name" → "org/name".
// Returns "" if `folder` doesn't have the expected prefix.
static std::string FolderToRepo(const std::string& folder) {
  const std::string kPrefix = "models--";
  if (folder.compare(0, kPrefix.size(), kPrefix) != 0) return "";
  std::string rest = folder.substr(kPrefix.size());
  // HF cache encodes the org/name slash as "--"; only the first "--" is the
  // separator. Repo names themselves can contain dashes but never "--".
  auto sep = rest.find("--");
  if (sep == std::string::npos) return "";
  return rest.substr(0, sep) + "/" + rest.substr(sep + 2);
}

std::vector<CachedModel> ListCachedModels() {
  std::vector<CachedModel> out;
  std::error_code ec;
  fs::path root(HfHubDir());
  if (!fs::exists(root, ec)) return out;

  for (const auto& repo_entry : fs::directory_iterator(root, ec)) {
    if (!repo_entry.is_directory()) continue;
    std::string repo_id = FolderToRepo(repo_entry.path().filename().string());
    if (repo_id.empty()) continue;

    fs::path snapshots = repo_entry.path() / "snapshots";
    if (!fs::exists(snapshots, ec)) continue;

    for (const auto& snap_entry : fs::directory_iterator(snapshots, ec)) {
      if (!snap_entry.is_directory()) continue;
      const std::string revision = snap_entry.path().filename().string();

      for (const auto& f : fs::directory_iterator(snap_entry.path(), ec)) {
        auto ext = f.path().extension();
        if (ext != ".litertlm" && ext != ".tflite") continue;
        // Skip in-flight downloads; aria2 leaves a sibling .aria2 file until
        // the transfer finishes. The .part variant is filtered by extension.
        fs::path control = f.path().string() + ".aria2";
        if (fs::exists(control, ec)) continue;
        out.push_back({repo_id, revision,
                       f.path().filename().string(),
                       f.path().string()});
      }
    }
  }
  return out;
}

}  // namespace lite_inference
