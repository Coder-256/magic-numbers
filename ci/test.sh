#!/bin/sh

if [ -z "$RUST_VERSION" ]; then
    echo 'Please set RUST_VERSION to a toolchain name or channel' >&2
    exit 1
fi

iif() {
    if [ "$1" -eq 0 ]; then
        echo "$2"
    else
        echo "$3"
    fi
}

run() {
    rustup run "$RUST_VERSION" "$@"
}

echo
echo "Testing with Rust $RUST_VERSION..."
echo
BUILD_RESULT=0

run cargo build --verbose
BUILD_RESULT="$(iif "$?" "$BUILD_RESULT" 1)"

run cargo test --verbose
BUILD_RESULT="$(iif "$?" "$BUILD_RESULT" 1)"

run cargo build --no-default-features --verbose
BUILD_RESULT="$(iif "$?" "$BUILD_RESULT" 1)"

run cargo test --no-default-features --verbose
BUILD_RESULT="$(iif "$?" "$BUILD_RESULT" 1)"

exit "$BUILD_RESULT"
