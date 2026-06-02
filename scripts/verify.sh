#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="full"
if [[ "${1:-}" == "--offline" ]]; then
  PROFILE="offline"
  shift
elif [[ "${1:-}" == "--quick" ]]; then
  PROFILE="quick"
  shift
elif [[ "${1:-}" == "--full" ]]; then
  PROFILE="full"
  shift
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: scripts/verify.sh [--offline|--quick|--full]

Runs the ROCm-Oxide production verification gate.

  --offline  CPU-only formatting, script syntax, and tool-crate tests.
  --quick  Unit/tool tests plus core GPU smoke coverage.
  --full   Full local gate, including heavier examples and visual artifacts.

Artifacts are written under target/production-readiness/.
USAGE
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  echo "unknown verify argument: $1" >&2
  echo "run scripts/verify.sh --help for usage" >&2
  exit 2
fi

ARTIFACT_DIR="$ROOT/target/production-readiness"
LOG="$ARTIFACT_DIR/verify-${PROFILE}.log"
mkdir -p "$ARTIFACT_DIR"
: > "$LOG"

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
  } >> "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run cargo test --manifest-path crates/rocm-oxide-kernel/Cargo.toml -- --test-threads=1
run cargo test --manifest-path tools/rocm-oxide-build/Cargo.toml -- --test-threads=1
run cargo test --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- --test-threads=1

if [[ "$PROFILE" == "offline" ]]; then
  run cargo fmt --check
  run bash -n scripts/verify.sh
  printf '\nverification profile `%s` passed; artifacts: %s\n' "$PROFILE" "$ARTIFACT_DIR"
  exit 0
fi

run cargo test -- --test-threads=1
run cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor
run cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide pipeline
run cargo run --example vector_add
run cargo run --example rust_device_generated_bindings
run cargo run --example feature_showcase
run cargo run --example performance_probe -- --json "$ARTIFACT_DIR/performance_probe.json"
run cargo run --example spectral_lattice -- --frames 1 --mode chain --output "$ARTIFACT_DIR/spectral_lattice_chain.png"

if [[ "$PROFILE" == "full" ]]; then
  run cargo run --example rust_device_add_one
  run cargo run --example rust_device_vector_add
  run cargo run --example compiler_feature_lab -- --frames 1
  run cargo run --example pinned_stream_vector_add
  run cargo run --example device_operation_chain
  run cargo run --example module_global
  run cargo run --example depth_aware_upscale
  run cargo run --example temporal_upscale
  run cargo run --example bvh_raytrace_benchmark
  run cargo run --example spectral_lattice -- --frames 3 --output "$ARTIFACT_DIR/spectral_lattice.png"
  for mode in core lds atomic chain; do
    run cargo run --example spectral_lattice -- --frames 3 --mode "$mode" --output "$ARTIFACT_DIR/spectral_lattice_${mode}.png"
  done
  run cargo run --example spectral_lattice -- --frames 1 --mode chain --resolution 4k --fps-limit 120 --gpu-work 256 --output "$ARTIFACT_DIR/spectral_lattice_4k.png"
  run cargo run
fi

printf '\nverification profile `%s` passed; artifacts: %s\n' "$PROFILE" "$ARTIFACT_DIR"
