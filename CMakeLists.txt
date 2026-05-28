cmake_minimum_required(VERSION 3.20)
project(lite_inference VERSION 1.0 LANGUAGES CXX C)

# Build version string. Set by CI from the git tag (e.g. "v1.2.3"); falls
# back to "dev" for local builds. Surfaced via --version and /health.
if(NOT DEFINED LITERT_VERSION OR LITERT_VERSION STREQUAL "")
  set(LITERT_VERSION "dev")
endif()

# ---------------------------------------------------------------------------
# C++ 20 baseline — enforced for all targets in this project.
# CMAKE_CXX_EXTENSIONS OFF disables -std=gnu++20 so we get strict -std=c++20.
# AppleClang 15+ (Xcode 15, macOS 14+) and Clang 16+ both fully support C++20.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Verify the compiler actually supports C++20 before proceeding.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)
if(NOT COMPILER_SUPPORTS_CXX20)
  message(FATAL_ERROR
    "Compiler does not support -std=c++20.\n"
    "Requires AppleClang 15+ (Xcode 15) or Clang 16+.\n"
    "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# LiteRT-LM location
# Lives as a git submodule at LiteRT-LM/ inside this repo.
# Override with -DLITERT_LM_DIR=... if needed.
# ---------------------------------------------------------------------------
if(NOT DEFINED LITERT_LM_DIR)
  set(LITERT_LM_DIR "${CMAKE_SOURCE_DIR}/LiteRT-LM")
endif()
get_filename_component(LITERT_LM_DIR "${LITERT_LM_DIR}" ABSOLUTE)

if(NOT EXISTS "${LITERT_LM_DIR}/c/engine.h")
  message(FATAL_ERROR
    "LiteRT-LM submodule not initialised at ${LITERT_LM_DIR}.\n"
    "Run: git submodule update --init --recursive\n"
    "Or set -DLITERT_LM_DIR=/path/to/LiteRT-LM")
endif()

# ---------------------------------------------------------------------------
# Platform → prebuilt subdir
# ---------------------------------------------------------------------------
if(APPLE)
  if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(LITERT_PREBUILT_SUBDIR "macos_x86_64")
  else()
    set(LITERT_PREBUILT_SUBDIR "macos_arm64")
  endif()
  set(LITERT_LIB_NAME "libLiteRt.dylib")
elseif(WIN32)
  set(LITERT_PREBUILT_SUBDIR "windows_x86_64")
  set(LITERT_LIB_NAME "LiteRt.dll")
elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LITERT_PREBUILT_SUBDIR "linux_arm64")
  else()
    set(LITERT_PREBUILT_SUBDIR "linux_x86_64")
  endif()
  set(LITERT_LIB_NAME "libLiteRt.so")
endif()

set(LITERT_PREBUILT_DIR "${LITERT_LM_DIR}/prebuilt/${LITERT_PREBUILT_SUBDIR}")
set(LITERT_LIB_PATH "${LITERT_PREBUILT_DIR}/${LITERT_LIB_NAME}")

# The C API engine library is built by Bazel. The shared bazel-bin/ symlink
# always points at the *last-built* platform, so building Android/iOS clobbers
# the macOS lib there. To keep platforms independent, the server reads from a
# stable per-platform copy under prebuilt/, falling back to bazel-bin if the
# copy doesn't exist yet.
#
# Refresh the macOS copy after a host build with:
#   cd LiteRT-LM && bazelisk build //lite_inference_server:libLiteRtLmEngine.so
#   cp bazel-bin/lite_inference_server/libLiteRtLmEngine.so \
#      prebuilt/macos_arm64/libLiteRtLmEngine.so
# LiteRt runtime library (for embedding / tflite inference).
# Lives alongside the LM engine lib in the same prebuilt dir.
if(WIN32)
  set(LITERT_RUNTIME_LIB_FILE "libLiteRt.dll")
  set(LITERT_RUNTIME_IMPORT_LIB "${LITERT_PREBUILT_DIR}/libLiteRt.lib")
elseif(APPLE)
  set(LITERT_RUNTIME_LIB_FILE "libLiteRt.dylib")
else()
  set(LITERT_RUNTIME_LIB_FILE "libLiteRt.so")
endif()
set(LITERT_RUNTIME_LIB "${LITERT_PREBUILT_DIR}/${LITERT_RUNTIME_LIB_FILE}")

set(LITERT_ENGINE_LIB_NAME "libLiteRtLmEngine")
if(WIN32)
  set(LITERT_ENGINE_LIB_FILE "${LITERT_ENGINE_LIB_NAME}.dll")
else()
  set(LITERT_ENGINE_LIB_FILE "${LITERT_ENGINE_LIB_NAME}.so")
endif()

set(LITERT_ENGINE_LIB_STABLE "${LITERT_PREBUILT_DIR}/${LITERT_ENGINE_LIB_FILE}")
set(LITERT_ENGINE_LIB_BAZEL "${LITERT_LM_DIR}/bazel-bin/lite_inference_server/${LITERT_ENGINE_LIB_FILE}")

if(EXISTS "${LITERT_ENGINE_LIB_STABLE}")
  set(LITERT_ENGINE_LIB "${LITERT_ENGINE_LIB_STABLE}")
else()
  set(LITERT_ENGINE_LIB "${LITERT_ENGINE_LIB_BAZEL}")
endif()

if(NOT EXISTS "${LITERT_ENGINE_LIB}")
  message(FATAL_ERROR
    "LiteRtLmEngine shared lib not found.\n"
    "Looked in:\n"
    "  ${LITERT_ENGINE_LIB_STABLE}\n"
    "  ${LITERT_ENGINE_LIB_BAZEL}\n"
    "Build it for the host platform with:\n"
    "  cd ${LITERT_LM_DIR}\n"
    "  bazelisk build //lite_inference_server:libLiteRtLmEngine.so\n"
    "Then copy it to ${LITERT_PREBUILT_DIR}/ to keep it stable.")
endif()

# Import the Bazel-built C API shared library (LM engine).
add_library(litert_lm SHARED IMPORTED)
set_target_properties(litert_lm PROPERTIES
  IMPORTED_LOCATION "${LITERT_ENGINE_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${LITERT_LM_DIR}"
)

# Import the LiteRt runtime library (for tflite / embedding inference).
if(EXISTS "${LITERT_RUNTIME_LIB}")
  add_library(litert_runtime SHARED IMPORTED)
  if(WIN32)
    set_target_properties(litert_runtime PROPERTIES
      IMPORTED_LOCATION "${LITERT_RUNTIME_LIB}"
      IMPORTED_IMPLIB   "${LITERT_RUNTIME_IMPORT_LIB}"
    )
  else()
    set_target_properties(litert_runtime PROPERTIES
      IMPORTED_LOCATION "${LITERT_RUNTIME_LIB}"
    )
  endif()
  set(HAVE_LITERT_RUNTIME TRUE)
else()
  message(WARNING "libLiteRt not found at ${LITERT_RUNTIME_LIB}; "
                  "embedding engine will not be available.")
  set(HAVE_LITERT_RUNTIME FALSE)
endif()

# ---------------------------------------------------------------------------
# Third-party: cpp-httplib (header-only) + nlohmann/json (header-only)
# Fetched via FetchContent so no manual install needed.
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(cpp_httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz
  URL_HASH SHA256=a0567bcd6c3fe5cef1b329b96245119047f876b49e06cc129a36a7a8dffe173e
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
  URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
  CMAKE_ARGS -DJSON_BuildTests=OFF
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)

set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)   # keep header-only
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cpp_httplib nlohmann_json)

# ---------------------------------------------------------------------------
# Server binary
# ---------------------------------------------------------------------------
add_executable(litert_server
  server/main.cc
  server/rest/openai_api.cc
  server/rest/openapi.cc
  core/litert/llm_engine.cc
  core/litert/embedding_engine.cc
  core/litert/webgpu_sampler_shim.c
  hub/model_resolver.cc
)

target_include_directories(litert_server PRIVATE
  "${CMAKE_SOURCE_DIR}"
  "${cpp_httplib_SOURCE_DIR}"
)

# OpenSSL for HTTPS in model_resolver (system OpenSSL, available on macOS via Homebrew).
find_package(OpenSSL REQUIRED)

target_link_libraries(litert_server PRIVATE
  litert_lm
  nlohmann_json::nlohmann_json
  OpenSSL::SSL
  OpenSSL::Crypto
)

if(HAVE_LITERT_RUNTIME)
  target_link_libraries(litert_server PRIVATE litert_runtime)
  target_compile_definitions(litert_server PRIVATE HAVE_LITERT_RUNTIME=1)
endif()

target_compile_definitions(litert_server PRIVATE
  CPPHTTPLIB_OPENSSL_SUPPORT
  LITERT_VERSION="${LITERT_VERSION}"
)

if(APPLE)
  set_target_properties(litert_server PROPERTIES
    INSTALL_RPATH "@executable_path"
  )
  # Copy libLiteRtLmEngine.so next to the binary and fix its install name so
  # the binary can find it via @executable_path at runtime.
  # Also copy all prebuilt dylibs (GPU delegates, constraint providers, etc.)
  # so --backend=gpu works without any manual setup.
  #
  # The engine lib's recorded install name depends on where Bazel built it, so
  # we derive it at build time with otool rather than hardcoding the bazel path.
  file(GLOB LITERT_PREBUILT_DYLIBS "${LITERT_PREBUILT_DIR}/*.dylib")
  # The macos_arm64 prebuilts ship libLiteRtWebGpuAccelerator.dylib + its paired
  # libLiteRtTopKWebGpuSampler.dylib, but the sampler dylib was built against
  # an older API and is missing 4 of 7 expected C entrypoints (notably
  # LiteRtTopKWebGpuSampler_UpdateConfig). When the WebGPU accelerator is
  # loaded, the sampler factory tries WebGPU first, fails, then falls back to
  # the CPU sampler, which then crashes trying to read GPU-resident logits
  # (top_p_cpu_sampler.cc -> litert_tensor_buffer ERROR). Symptom: warmup or
  # first inference fails with INTERNAL on Apple Silicon.
  #
  # Excluding both dylibs forces the sampler factory down its Metal path,
  # which works correctly with libLiteRtTopKMetalSampler.dylib. Re-enable
  # once upstream ships a complete WebGPU sampler for macOS.
  list(FILTER LITERT_PREBUILT_DYLIBS EXCLUDE REGEX "libLiteRt(TopK)?WebGpu[A-Za-z]*\\.dylib$")
  add_custom_command(TARGET litert_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LITERT_ENGINE_LIB}"
            "$<TARGET_FILE_DIR:litert_server>/libLiteRtLmEngine.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${LITERT_PREBUILT_DYLIBS}
            "$<TARGET_FILE_DIR:litert_server>/"
    COMMAND ${CMAKE_COMMAND} -E rm -f
            "$<TARGET_FILE_DIR:litert_server>/libLiteRtWebGpuAccelerator.dylib"
            "$<TARGET_FILE_DIR:litert_server>/libLiteRtTopKWebGpuSampler.dylib"
    COMMAND /bin/sh "${CMAKE_SOURCE_DIR}/cmake/make_portable.sh"
            "$<TARGET_FILE_DIR:litert_server>"
    COMMENT "Copying runtime dylibs and making build self-contained"
  )
elseif(UNIX)
  # Bake $ORIGIN into the build-tree binary so it can be copied straight
  # out of /build/ into a runtime image without needing `cmake --install`.
  # libLiteRtLmEngine.so is copied next to the binary by the POST_BUILD step
  # below, so $ORIGIN resolves correctly in both the build tree and the
  # final image.
  set_target_properties(litert_server PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
  )
  add_custom_command(TARGET litert_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LITERT_ENGINE_LIB}"
            "$<TARGET_FILE_DIR:litert_server>/libLiteRtLmEngine.so"
    COMMENT "Copying libLiteRtLmEngine.so next to binary"
  )
  if(HAVE_LITERT_RUNTIME)
    add_custom_command(TARGET litert_server POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${LITERT_RUNTIME_LIB}"
              "$<TARGET_FILE_DIR:litert_server>/libLiteRt.so"
      COMMENT "Copying libLiteRt.so next to binary"
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# Install: copy binary + engine lib together so the server is self-contained.
# ---------------------------------------------------------------------------
install(TARGETS litert_server DESTINATION bin)
install(FILES "${LITERT_ENGINE_LIB}" DESTINATION bin)

