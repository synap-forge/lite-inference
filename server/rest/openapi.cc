#include "server/rest/openapi.h"

#include <mutex>
#include <utility>

namespace lite_inference::openapi {

namespace {
std::mutex& Mtx() { static std::mutex m; return m; }
std::vector<RouteSpec>& Reg() { static std::vector<RouteSpec> r; return r; }
}  // namespace

const std::vector<RouteSpec>& Registry() { return Reg(); }

RouteBuilder::RouteBuilder(httplib::Server& svr, std::string method,
                           std::string path)
    : svr_(&svr) {
  spec_.method = std::move(method);
  spec_.path   = std::move(path);
}

void RouteBuilder::Handler(httplib::Server::Handler handler) {
  const std::string& m = spec_.method;
  if      (m == "GET")    svr_->Get(spec_.path, handler);
  else if (m == "POST")   svr_->Post(spec_.path, handler);
  else if (m == "PUT")    svr_->Put(spec_.path, handler);
  else if (m == "DELETE") svr_->Delete(spec_.path, handler);
  else if (m == "PATCH")  svr_->Patch(spec_.path, handler);
  {
    std::lock_guard<std::mutex> lk(Mtx());
    Reg().push_back(spec_);
  }
}

nlohmann::json BuildSpec(const std::string& title, const std::string& version) {
  nlohmann::json paths = nlohmann::json::object();
  for (const auto& r : Registry()) {
    nlohmann::json op = {
        {"summary", r.summary},
    };
    if (!r.tags.empty()) op["tags"] = r.tags;
    if (!r.request_schema.is_null() && !r.request_schema.empty()) {
      op["requestBody"] = {
          {"required", true},
          {"content", {{r.request_content_type, {{"schema", r.request_schema}}}}},
      };
    }
    nlohmann::json responses = nlohmann::json::object();
    if (r.responses.empty()) {
      responses["200"] = {{"description", "OK"}};
    } else {
      for (const auto& resp : r.responses) {
        nlohmann::json entry = {{"description", resp.description}};
        if (!resp.schema.is_null() && !resp.schema.empty()) {
          entry["content"] = {{resp.content_type, {{"schema", resp.schema}}}};
        } else {
          entry["content"] = {{resp.content_type, nlohmann::json::object()}};
        }
        responses[std::to_string(resp.status)] = entry;
      }
    }
    op["responses"] = responses;

    std::string method_lc = r.method;
    for (auto& c : method_lc) c = static_cast<char>(::tolower(c));
    if (!paths.contains(r.path)) paths[r.path] = nlohmann::json::object();
    paths[r.path][method_lc] = op;
  }
  return {
      {"openapi", "3.1.0"},
      {"info", {{"title", title}, {"version", version}}},
      {"paths", paths},
  };
}

}  // namespace lite_inference::openapi
