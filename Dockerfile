# ---------------------------------------------------------------------------
# Stage 1: Build — Oracle Linux 9 + Bazel + CMake
# ---------------------------------------------------------------------------
FROM oraclelinux:9 AS builder

RUN dnf install -y \
      git git-lfs clang cmake openssl-devel \
      python3 java-17-openjdk-devel \
      curl unzip && \
    dnf clean all

# Install bazelisk
RUN curl -fsSL \
      https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
      -o /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk

WORKDIR /src
COPY . .

# Initialise git so Bazel workspace resolution works inside Docker
RUN git config --global --add safe.directory /src && \
    git lfs install --skip-repo

ENV CC=clang \
    CXX=clang++ \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk

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
# Stage 2: Runtime — minimal Oracle Linux 9
# ---------------------------------------------------------------------------
FROM oraclelinux:9-minimal AS runtime

RUN microdnf install -y openssl libstdc++ && \
    microdnf clean all

WORKDIR /app

COPY --from=builder /build/litert_server          ./litert_server
COPY --from=builder /build/libLiteRtLmEngine.so   ./libLiteRtLmEngine.so

# libLiteRtLmEngine.so lives next to the binary; LD_LIBRARY_PATH is not needed
# because CMake sets INSTALL_RPATH="$ORIGIN".
ENV HF_HOME=/data/huggingface

EXPOSE 8080

ENTRYPOINT ["./litert_server"]
CMD ["--backend", "cpu", "--port", "8080"]
