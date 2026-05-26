#pragma once
//
// OpenAPI 3.1 registry and a small DSL for declaring routes next to their
// handlers. Each Route(...) call records path, method, summary/tags, request
// body schema, and response schemas in a process-wide registry, then forwards
// the handler to httplib. BuildOpenApiSpec() walks the registry.

#include <functional>
#include <string>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "server/rest/reflect.h"

namespace lite_inference::openapi {

struct ResponseSpec {
  int                status      = 200;
  std::string        description = "OK";
  nlohmann::json     schema;          // may be empty for "no body"
  std::string        content_type = "application/json";
};

struct RouteSpec {
  std::string                method;     // GET, POST...
  std::string                path;       // /v1/...
  std::string                summary;
  std::vector<std::string>   tags;
  nlohmann::json             request_schema;       // empty if none
  std::string                request_content_type = "application/json";
  std::vector<ResponseSpec>  responses;
};

const std::vector<RouteSpec>& Registry();

// Builds an OpenAPI 3.1 document from the registry.
nlohmann::json BuildSpec(const std::string& title, const std::string& version);

class RouteBuilder {
 public:
  RouteBuilder(httplib::Server& svr, std::string method, std::string path);

  RouteBuilder& Summary(std::string s) { spec_.summary = std::move(s); return *this; }
  RouteBuilder& Tag(std::string t)     { spec_.tags.push_back(std::move(t)); return *this; }

  template <class T>
  RouteBuilder& Body(std::string content_type = "application/json") {
    spec_.request_schema       = reflect::Schema<T>();
    spec_.request_content_type = std::move(content_type);
    return *this;
  }

  template <class T>
  RouteBuilder& Response(int status, std::string description = "OK",
                         std::string content_type = "application/json") {
    spec_.responses.push_back({status, std::move(description),
                               reflect::Schema<T>(), std::move(content_type)});
    return *this;
  }

  // Response with no schema (e.g. SSE stream described only by content type).
  RouteBuilder& RawResponse(int status, std::string description,
                            std::string content_type) {
    spec_.responses.push_back({status, std::move(description),
                               nlohmann::json::object(), std::move(content_type)});
    return *this;
  }

  // Registers `handler` with httplib and commits the spec.
  void Handler(httplib::Server::Handler handler);

 private:
  httplib::Server* svr_;
  RouteSpec        spec_;
};

inline RouteBuilder Route(httplib::Server& svr, std::string method,
                          std::string path) {
  return RouteBuilder(svr, std::move(method), std::move(path));
}

}  // namespace lite_inference::openapi
