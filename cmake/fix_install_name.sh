#!/bin/sh
# Rewrite the litert_server binary's reference to libLiteRtLmEngine.so to use
# @executable_path. The recorded install name varies by where Bazel built the
# lib, so we read it back with otool instead of hardcoding it.
#
# Usage: fix_install_name.sh <path-to-litert_server>

set -e

binary="$1"
if [ -z "$binary" ]; then
  echo "fix_install_name.sh: missing binary path" >&2
  exit 1
fi

old=$(otool -L "$binary" | awk '/libLiteRtLmEngine\.so/{print $1; exit}')
if [ -n "$old" ]; then
  install_name_tool -change "$old" "@executable_path/libLiteRtLmEngine.so" "$binary"
fi
