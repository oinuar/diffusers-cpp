#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "$PROJECT_ROOT:$PROJECT_ROOT" \
    -w "$PWD" \
    -e CARGO_TARGET_DIR \
    rust:1.88-bookworm \
    cargo "$@"
