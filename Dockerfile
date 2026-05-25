# ---------------------------------------------------------------------------
# Stage 1: Build — Ubuntu 24.04 (clang-18, full C++20 support)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs clang cmake \
      libssl-dev \
      python3 openjdk-17-jdk-headless \
      curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install bazelisk
RUN curl -fsSL \
      https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
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
      -Wno-dev \
      -S /src && \
    cmake --build /build -j$(nproc)

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal Ubuntu 24.04
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3 libstdc++6 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/litert_server         ./litert_server
COPY --from=builder /build/libLiteRtLmEngine.so  ./libLiteRtLmEngine.so

# libLiteRtLmEngine.so is next to the binary; CMake sets RPATH=$ORIGIN
# so no LD_LIBRARY_PATH needed.
ENV HF_HOME=/data/huggingface

EXPOSE 8080

ENTRYPOINT ["./litert_server"]
CMD ["--backend", "cpu", "--port", "8080"]
