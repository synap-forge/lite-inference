# ---------------------------------------------------------------------------
# Stage 1: Build — Ubuntu 24.04 (clang-18, full C++20 support)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

# Set by CI from the git tag (e.g. "v1.2.3"). Defaults to "dev" for
# local builds. Baked into the binary via -DLITERT_VERSION.
ARG VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs clang cmake ninja-build \
      libssl-dev \
      python3 openjdk-17-jdk-headless \
      curl ca-certificates patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install bazelisk (pick binary matching the build platform)
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
      aarch64) BAZEL_ARCH=arm64 ;; \
      *)        BAZEL_ARCH=amd64 ;; \
    esac && \
    curl -fsSL \
      "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-${BAZEL_ARCH}" \
      -o /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk

WORKDIR /src
COPY . .

RUN git config --global --add safe.directory /src && \
    git lfs install --skip-repo && \
    # If COPY brought in an uninitialised submodule (local build outside CI),
    # initialise it now. In CI the submodule is already populated by checkout.
    if [ ! -f LiteRT-LM/WORKSPACE ]; then \
      git submodule update --init --recursive; \
    fi

ENV CC=clang \
    CXX=clang++ \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Prepare Bazel package (copies sources + writes BUILD + patches WORKSPACE)
RUN SKIP_BAZEL_BUILD=1 bash setup.sh

# Build the shared engine lib via Bazel
RUN cd LiteRT-LM && bazelisk build //lite_inference_server:libLiteRtLmEngine.so

# Build the server binary via CMake
RUN cmake -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DLITERT_VERSION="${VERSION}" \
      -GNinja \
      -Wno-dev \
      -S /src && \
    cmake --build /build -j$(nproc)

# Bazel records build-time absolute paths as SONAMEs and NEEDED entries on
# the shared libs it produces, which the runtime loader then can't resolve.
# Normalise all the libs we ship so they reference each other by bare
# filename, with $ORIGIN as their RPATH.
RUN set -eux; \
    # Stage every .so we ship into /build/ so the patching loop sees them.
    cp /src/LiteRT-LM/prebuilt/linux_arm64/*.so /build/; \
    # Engine .so: fix its SONAME and clear bazel-injected RPATH.
    patchelf --set-soname libLiteRtLmEngine.so /build/libLiteRtLmEngine.so; \
    # Every .so we ship: bare-name SONAME, $ORIGIN RPATH, rewrite any
    # NEEDED entry that's still an absolute path to its basename.
    for lib in /build/*.so; do \
      patchelf --set-rpath '$ORIGIN' "$lib"; \
      for need in $(patchelf --print-needed "$lib"); do \
        case "$need" in /*) \
          patchelf --replace-needed "$need" "$(basename "$need")" "$lib" ;; \
        esac; \
      done; \
    done; \
    # Same NEEDED-path cleanup on the server binary.
    for need in $(patchelf --print-needed /build/litert_server); do \
      case "$need" in /*) \
        patchelf --replace-needed "$need" "$(basename "$need")" /build/litert_server ;; \
      esac; \
    done

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal Ubuntu 24.04
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ARG VERSION=dev
LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/synap-forge/lite-inference"

ENV DEBIAN_FRONTEND=noninteractive \
    LITERT_VERSION=${VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3 libstdc++6 ca-certificates curl aria2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/litert_server ./litert_server
# Engine .so + LiteRT prebuilt delegate/constraint-provider .so's, all
# staged and patchelf'd in /build/ during the builder stage so they
# reference each other by bare name with $ORIGIN RPATH.
COPY --from=builder /build/*.so ./

# All deps sit next to the binary; RPATH=$ORIGIN
# so no LD_LIBRARY_PATH needed.
ENV HF_HOME=/data/huggingface

EXPOSE 8080

ENTRYPOINT ["./litert_server"]
CMD ["--backend", "cpu", "--port", "8080", "--startup_load", "none", "--embed_repo", "litert-community/embeddinggemma-300m", "--embed_seq_len", "1024"]
