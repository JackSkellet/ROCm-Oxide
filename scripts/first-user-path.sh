#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run cargo run --example hello_gpu
run cargo run --features device-spike --example hello_gpu_rust
run cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor

printf '\nfirst-user path passed\n'
