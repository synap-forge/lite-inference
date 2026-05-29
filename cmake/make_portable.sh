#!/bin/sh
# Makes the build directory fully self-contained and portable.
# Copies OpenSSL dylibs next to the binary and rewrites all absolute/bazel
# install names to @executable_path so the bundle runs anywhere.
#
# Usage: make_portable.sh <build_dir>

set -e

BUILD_DIR="$1"
if [ -z "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/litert_server" ]; then
  echo "Usage: make_portable.sh <build_dir>" >&2
  exit 1
fi

BINARY="$BUILD_DIR/litert_server"

# ---------------------------------------------------------------------------
# 1. Copy OpenSSL into the build dir (it's the only Homebrew dependency)
# ---------------------------------------------------------------------------
for lib in libssl.3.dylib libcrypto.3.dylib; do
  src=$(otool -L "$BINARY" | awk "/$lib/"'{print $1; exit}')
  if [ -n "$src" ] && [ -f "$src" ] && [ ! -f "$BUILD_DIR/$lib" ]; then
    echo "Copying $lib from $src"
    cp "$src" "$BUILD_DIR/$lib"
    chmod 755 "$BUILD_DIR/$lib"
  fi
done

# ---------------------------------------------------------------------------
# 2. Fix install names of all libs in the build dir to use @rpath
# ---------------------------------------------------------------------------
for lib in "$BUILD_DIR"/*.dylib "$BUILD_DIR"/*.so; do
  [ -f "$lib" ] || continue
  libname=$(basename "$lib")
  current=$(otool -D "$lib" 2>/dev/null | tail -1)
  # Rewrite only if the install name isn't already @rpath or @loader_path
  case "$current" in
    @rpath/*|@loader_path/*|@executable_path/*) ;;
    *)
      echo "Fixing install name: $libname -> @rpath/$libname"
      chmod u+w "$lib"
      install_name_tool -id "@rpath/$libname" "$lib"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# 3. Fix the binary's references to all libs in the build dir
# ---------------------------------------------------------------------------
# OpenSSL
for lib in libssl.3.dylib libcrypto.3.dylib; do
  old=$(otool -L "$BINARY" | awk "/$lib/"'{print $1; exit}')
  if [ -n "$old" ]; then
    case "$old" in
      @executable_path/*) ;;
      *)
        echo "Rewriting $BINARY -> $lib"
        install_name_tool -change "$old" "@executable_path/$lib" "$BINARY"
        ;;
    esac
  fi
done

# Engine lib (libLiteRtLmEngine.so) - already handled by fix_install_name.sh
# but re-run here in case it wasn't set yet
old=$(otool -L "$BINARY" | awk '/libLiteRtLmEngine\.so/{print $1; exit}')
if [ -n "$old" ]; then
  case "$old" in
    @executable_path/*) ;;
    *)
      echo "Rewriting $BINARY -> libLiteRtLmEngine.so"
      install_name_tool -change "$old" "@executable_path/libLiteRtLmEngine.so" "$BINARY"
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# 4. Add @executable_path as rpath on the binary so @rpath libs resolve
# ---------------------------------------------------------------------------
# Check if already present
if ! otool -l "$BINARY" | grep -q "path @executable_path "; then
  echo "Adding rpath @executable_path to $BINARY"
  install_name_tool -add_rpath "@executable_path" "$BINARY"
fi

# ---------------------------------------------------------------------------
# 4b. Strip any absolute rpath pointing back into the prebuilt source tree.
#     CMake bakes in an absolute LC_RPATH to LiteRT-LM/prebuilt/<platform>
#     (where the GPU dylibs were linked from). dyld searches rpaths in order,
#     so that absolute path shadows @executable_path and re-introduces dylibs
#     we deliberately excluded from the bundle — notably the broken
#     libLiteRtWebGpuAccelerator.dylib, which drags the sampler factory down
#     the broken WebGPU path instead of the working Metal one. Remove every
#     absolute (non-@) rpath so only @executable_path / @loader_path remain.
# ---------------------------------------------------------------------------
for rp in $(otool -l "$BINARY" | awk '/cmd LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do
  case "$rp" in
    @*) ;;  # keep @executable_path / @loader_path / @rpath
    *)
      echo "Removing absolute rpath from $BINARY: $rp"
      install_name_tool -delete_rpath "$rp" "$BINARY"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# 5. Re-sign the binary — install_name_tool invalidates the code signature
#    on Apple Silicon (macOS 15+). Ad-hoc signing is sufficient for local use.
# ---------------------------------------------------------------------------
echo "Re-signing $BINARY (ad-hoc)"
codesign --force --sign - "$BINARY"

# Re-sign modified dylibs too (libssl was chmod'd + potentially modified)
for lib in "$BUILD_DIR"/libssl.3.dylib "$BUILD_DIR"/libcrypto.3.dylib "$BUILD_DIR"/libLiteRtLmEngine.so; do
  [ -f "$lib" ] && codesign --force --sign - "$lib"
done

echo ""
echo "Done. $BUILD_DIR is now self-contained."
echo "Transfer the following files together:"
ls "$BUILD_DIR"/litert_server "$BUILD_DIR"/*.dylib "$BUILD_DIR"/*.so 2>/dev/null | sed 's/^/  /'
