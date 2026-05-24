cmake_minimum_required(VERSION 3.20)
project(litert_inference VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# LiteRT-LM location (inherit from parent or set independently)
# ---------------------------------------------------------------------------
if(NOT DEFINED LITERT_LM_DIR)
  # When built as part of the root project this is already set.
  # When built standalone (e.g. from the Flutter plugin), fall back to the
  # submodule inside the repo root (one level up from library/).
  set(LITERT_LM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../LiteRT-LM")
endif()
get_filename_component(LITERT_LM_DIR "${LITERT_LM_DIR}" ABSOLUTE)

if(NOT EXISTS "${LITERT_LM_DIR}/c/engine.h")
  message(FATAL_ERROR
    "LiteRT-LM submodule not initialised at ${LITERT_LM_DIR}.\n"
    "Run: git submodule update --init --recursive\n"
    "Or set -DLITERT_LM_DIR=/path/to/LiteRT-LM")
endif()

# ---------------------------------------------------------------------------
# Platform → prebuilt shared library
# ---------------------------------------------------------------------------
if(ANDROID)
  # Android ABI is set by the NDK toolchain (ANDROID_ABI).
  if(ANDROID_ABI STREQUAL "x86_64")
    set(LITERT_PREBUILT_SUBDIR "android_x86_64")
  elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(LITERT_PREBUILT_SUBDIR "android_arm32")
  else()
    set(LITERT_PREBUILT_SUBDIR "android_arm64")   # arm64-v8a (default)
  endif()
  set(LITERT_LIB_NAME "libLiteRt.so")
  if(NOT DEFINED LITERT_ENGINE_LIB)
    set(LITERT_ENGINE_LIB "${LITERT_LM_DIR}/bazel-bin/lite_inference_server/libLiteRtLmEngine.so")
  endif()
elseif(IOS)
  set(LITERT_PREBUILT_SUBDIR "ios_arm64")
  set(LITERT_LIB_NAME "libLiteRt.a")
  if(NOT DEFINED LITERT_ENGINE_LIB)
    set(LITERT_ENGINE_LIB "${LITERT_LM_DIR}/bazel-bin/lite_inference_server/libLiteRtLmEngine.a")
  endif()
elseif(APPLE)
  if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(LITERT_PREBUILT_SUBDIR "macos_x86_64")
  else()
    set(LITERT_PREBUILT_SUBDIR "macos_arm64")
  endif()
  set(LITERT_LIB_NAME "libLiteRt.dylib")
  if(NOT DEFINED LITERT_ENGINE_LIB)
    set(LITERT_ENGINE_LIB "${LITERT_LM_DIR}/bazel-bin/lite_inference_server/libLiteRtLmEngine.so")
  endif()
elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LITERT_PREBUILT_SUBDIR "linux_arm64")
  else()
    set(LITERT_PREBUILT_SUBDIR "linux_x86_64")
  endif()
  set(LITERT_LIB_NAME "libLiteRt.so")
  if(NOT DEFINED LITERT_ENGINE_LIB)
    set(LITERT_ENGINE_LIB "${LITERT_LM_DIR}/bazel-bin/lite_inference_server/libLiteRtLmEngine.so")
  endif()
endif()

# Import Bazel-built C API library (only when not already defined by a parent).
if(NOT EXISTS "${LITERT_ENGINE_LIB}")
  message(FATAL_ERROR
    "LiteRtLmEngine library not found: ${LITERT_ENGINE_LIB}\n"
    "Build it once with:\n"
    "  cd ${LITERT_LM_DIR}\n"
    "  bazelisk build //lite_inference_server:libLiteRtLmEngine.so")
endif()

if(NOT TARGET litert_lm)
  if(IOS)
    add_library(litert_lm STATIC IMPORTED)
    set_target_properties(litert_lm PROPERTIES
      IMPORTED_LOCATION "${LITERT_ENGINE_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${LITERT_LM_DIR}"
    )
  elseif(ANDROID)
    # Android's dynamic linker only accepts bare filenames in NEEDED entries.
    # CMake always uses IMPORTED_LOCATION verbatim as the linker input, which
    # embeds the path. Instead we use an INTERFACE target that passes -L + -l:
    # to the linker — the -l: form looks up the file by name in -L dirs and
    # records only the bare filename in the NEEDED entry.
    get_filename_component(_engine_name "${LITERT_ENGINE_LIB}" NAME)
    get_filename_component(_engine_dir  "${LITERT_ENGINE_LIB}" DIRECTORY)
    add_library(litert_lm INTERFACE IMPORTED)
    set_target_properties(litert_lm PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LITERT_LM_DIR}"
      INTERFACE_LINK_OPTIONS        "-L${_engine_dir};-l:${_engine_name}"
    )
  else()
    add_library(litert_lm SHARED IMPORTED)
    set_target_properties(litert_lm PROPERTIES
      IMPORTED_LOCATION "${LITERT_ENGINE_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${LITERT_LM_DIR}"
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# litert_inference shared library
# ---------------------------------------------------------------------------
# On iOS, Flutter plugins are built as static frameworks, but the FFI loader
# requires a dylib. We build SHARED on all platforms; the Flutter plugin's
# podspec/build.gradle controls how it's packaged.
add_library(litert_inference SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/litert_inference.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/../core/litert/llm_engine.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/../core/litert/webgpu_sampler_shim.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dart/dart_api_dl.c"
)

# nlohmann_json: reuse from parent if already fetched, otherwise fetch here.
if(NOT TARGET nlohmann_json::nlohmann_json)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
    CMAKE_ARGS -DJSON_BuildTests=OFF
  )
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

target_include_directories(litert_inference
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}"           # exposes litert_inference.h
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.."        # core/engine.h etc.
)

target_link_libraries(litert_inference PRIVATE litert_lm nlohmann_json::nlohmann_json)

# Visibility: hide all symbols except the public C API so the library is safe
# to dlopen alongside other native libs in a Flutter app.
set_target_properties(litert_inference PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if(ANDROID)
  target_link_libraries(litert_inference PRIVATE log android)
  set_target_properties(litert_inference PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
elseif(APPLE AND NOT IOS)
  set_target_properties(litert_inference PROPERTIES
    INSTALL_RPATH "@loader_path"
  )
elseif(UNIX)
  set_target_properties(litert_inference PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS litert_inference DESTINATION lib)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/litert_inference.h" DESTINATION include)
