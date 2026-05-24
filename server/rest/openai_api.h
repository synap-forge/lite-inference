#pragma once
#include <string>
#include "core/engine.h"

namespace lite_inference {

struct ServerOptions {
  std::string host    = "0.0.0.0";
  int         port    = 8080;
  std::string api_key;
};

int RunServer(EngineBase& engine, const ServerOptions& options);

}  // namespace lite_inference
