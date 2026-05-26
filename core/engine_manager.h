#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "core/engine.h"

namespace lite_inference {

// Holds the currently-loaded inference engine behind a shared_mutex so request
// handlers can keep using the engine while a swap is requested. The swap waits
// for all in-flight requests to release their guards before destroying the old
// engine and installing the new one.
//
// Lifecycle:
//   1. Construct with the initial engine.
//   2. Request handlers call Acquire() and use the returned guard's engine().
//      Multiple handlers can hold guards concurrently.
//   3. POST /v1/models/load calls Swap(builder). The builder is invoked under
//      the write lock; on success, the returned engine replaces the old one
//      and the old engine is destroyed when the write lock releases.
class EngineManager {
 public:
  // Builder closes over the resolved model path + options; returns a fully
  // initialized engine or nullptr on failure (with `error_out` set).
  using Builder = std::function<std::unique_ptr<EngineBase>(std::string& error_out)>;

  explicit EngineManager(std::unique_ptr<EngineBase> initial)
      : engine_(std::move(initial)) {}

  // Read-side guard: keeps a shared lock on the manager while the caller holds
  // it. The pointer is stable for the guard's lifetime.
  class Guard {
   public:
    EngineBase& engine() const { return *engine_; }
    EngineBase* operator->() const { return engine_; }
    explicit operator bool() const { return engine_ != nullptr; }

   private:
    friend class EngineManager;
    Guard(std::shared_lock<std::shared_mutex> lock, EngineBase* engine)
        : lock_(std::move(lock)), engine_(engine) {}

    std::shared_lock<std::shared_mutex> lock_;
    EngineBase*                         engine_;
  };

  Guard Acquire() {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return Guard(std::move(lk), engine_.get());
  }

  // Blocks until all outstanding Guards are released, then constructs the new
  // engine via `build`. If construction fails, the old engine is preserved
  // and the function returns false with `error_out` set.
  bool Swap(const Builder& build, std::string& error_out) {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    auto next = build(error_out);
    if (!next) return false;
    engine_ = std::move(next);
    return true;
  }

 private:
  std::shared_mutex            mtx_;
  std::unique_ptr<EngineBase>  engine_;
};

}  // namespace lite_inference
