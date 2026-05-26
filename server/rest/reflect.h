#pragma once
//
// Tiny compile-time reflection used to derive JSON Schema from C++ DTOs.
// Declare reflectable fields with LITERT_REFLECT(Type, f1, f2, ...).
//
// The schema walker (Schema<T>()) recurses into nested reflected structs,
// std::vector<U>, std::optional<U>, std::string, integral, floating, and bool.
// Unknown types fall back to an empty schema (any).

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "nlohmann/json.hpp"

namespace lite_inference::reflect {

template <class T> struct is_optional : std::false_type {};
template <class T> struct is_optional<std::optional<T>> : std::true_type {};

template <class T> struct is_vector : std::false_type {};
template <class T, class A> struct is_vector<std::vector<T, A>> : std::true_type {};

template <class T, class = void>
struct has_reflect : std::false_type {};

template <class T>
struct has_reflect<T, std::void_t<decltype(T::LiteRtReflectedFieldNames())>>
    : std::true_type {};

// Forward declaration; specialised by user code if needed.
template <class T>
nlohmann::json Schema();

template <class T>
nlohmann::json SchemaForScalar() {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<U, bool>) {
    return {{"type", "boolean"}};
  } else if constexpr (std::is_integral_v<U>) {
    return {{"type", "integer"}};
  } else if constexpr (std::is_floating_point_v<U>) {
    return {{"type", "number"}};
  } else if constexpr (std::is_same_v<U, std::string>) {
    return {{"type", "string"}};
  } else if constexpr (is_optional<U>::value) {
    return Schema<typename U::value_type>();
  } else if constexpr (is_vector<U>::value) {
    return {{"type", "array"},
            {"items", Schema<typename U::value_type>()}};
  } else if constexpr (has_reflect<U>::value) {
    return Schema<U>();
  } else {
    return nlohmann::json::object();  // unknown -> any
  }
}

template <class T>
nlohmann::json Schema() {
  if constexpr (has_reflect<T>::value) {
    nlohmann::json props = nlohmann::json::object();
    nlohmann::json required = nlohmann::json::array();
    T sample{};
    const char* const* names = T::LiteRtReflectedFieldNames();
    std::size_t i = 0;
    T::LiteRtForEachField(sample, [&](auto& field) {
      using F = std::remove_reference_t<decltype(field)>;
      props[names[i]] = SchemaForScalar<F>();
      if (!is_optional<F>::value) required.push_back(names[i]);
      ++i;
    });
    nlohmann::json s = {{"type", "object"}, {"properties", props}};
    if (!required.empty()) s["required"] = required;
    return s;
  } else {
    return SchemaForScalar<T>();
  }
}

}  // namespace lite_inference::reflect

// Variadic helpers. We need both a field-walker and a names array.
#define LITERT_REFLECT_EXPAND(x) x
#define LITERT_REFLECT_STRINGIFY(x) #x

// For up to 16 fields. Extend by adding more lines if needed.
#define LITERT_REFLECT_NARGS_(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,N,...) N
#define LITERT_REFLECT_NARGS(...) \
  LITERT_REFLECT_EXPAND(LITERT_REFLECT_NARGS_(__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1))

#define LITERT_REFLECT_FE_1(F,x)              F(x)
#define LITERT_REFLECT_FE_2(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_1(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_3(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_2(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_4(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_3(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_5(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_4(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_6(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_5(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_7(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_6(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_8(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_7(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_9(F,x,...)          F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_8(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_10(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_9(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_11(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_10(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_12(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_11(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_13(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_12(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_14(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_13(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_15(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_14(F,__VA_ARGS__))
#define LITERT_REFLECT_FE_16(F,x,...)         F(x) LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_15(F,__VA_ARGS__))

#define LITERT_REFLECT_FE_DISPATCH_(N) LITERT_REFLECT_FE_##N
#define LITERT_REFLECT_FE_DISPATCH(N)  LITERT_REFLECT_FE_DISPATCH_(N)
#define LITERT_REFLECT_FOR_EACH(F, ...) \
  LITERT_REFLECT_EXPAND(LITERT_REFLECT_FE_DISPATCH(LITERT_REFLECT_NARGS(__VA_ARGS__))(F, __VA_ARGS__))

#define LITERT_REFLECT_VISIT_(field) fn(v.field);
#define LITERT_REFLECT_NAME_(field)  LITERT_REFLECT_STRINGIFY(field),

#define LITERT_REFLECT(T, ...)                                          \
  template <class Fn>                                                   \
  static void LiteRtForEachField(T& v, Fn fn) {                         \
    LITERT_REFLECT_FOR_EACH(LITERT_REFLECT_VISIT_, __VA_ARGS__)         \
  }                                                                     \
  static const char* const* LiteRtReflectedFieldNames() {               \
    static const char* names[] = {                                      \
      LITERT_REFLECT_FOR_EACH(LITERT_REFLECT_NAME_, __VA_ARGS__)        \
      nullptr                                                           \
    };                                                                  \
    return names;                                                       \
  }
