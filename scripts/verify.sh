#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo test
cargo test --manifest-path crates/rocm-oxide-kernel/Cargo.toml
cargo test --manifest-path tools/rocm-oxide-build/Cargo.toml
cargo test --manifest-path tools/cargo-rocm-oxide/Cargo.toml
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide pipeline
cargo run --example vector_add
cargo run --example rust_device_add_one
cargo run --example rust_device_vector_add
cargo run --example rust_device_generated_bindings
cargo run --example feature_showcase
cargo run --example performance_probe -- --json target/performance_probe.json
cargo run --example pinned_stream_vector_add
cargo run --example device_operation_chain
cargo run --example module_global
cargo run --example depth_aware_upscale
cargo run --example temporal_upscale
cargo run --example bvh_raytrace_benchmark
cargo run --example spectral_lattice -- --frames 3 --output target/spectral_lattice.png
for mode in core lds atomic chain; do
  cargo run --example spectral_lattice -- --frames 3 --mode "$mode" --output "target/spectral_lattice_${mode}.png"
done
cargo run
