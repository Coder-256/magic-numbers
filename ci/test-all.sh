#!/bin/sh

run() {
    RUST_VERSION="$1" "$(dirname "$0")/test.sh"
}

echo_status() {
    if [ "$1" -eq 0 ]; then
        echo '\033[0;32mSUCCESS\033[0m'
    else
        echo '\033[0;31FAIL\033[0m'
    fi
}

run stable
STABLE_RESULT="$?"
run beta
BETA_RESULT="$?"
run nightly
NIGHTLY_RESULT="$?"

echo
echo
echo "Result:"
echo "Stable : $(echo_status "$STABLE_RESULT")"
echo "Beta   : $(echo_status "$BETA_RESULT")"
echo "Nightly: $(echo_status "$NIGHTLY_RESULT")"

if [ "$STABLE_RESULT" -eq 0 ] && [ "$BETA_RESULT" -eq 0 ]; then
    exit 0
else
    exit 1
fi
