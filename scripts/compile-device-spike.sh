#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo run --quiet --manifest-path tools/rocm-oxide-build/Cargo.toml -- "$@"
