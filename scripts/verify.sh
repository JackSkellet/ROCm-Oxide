#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo test
cargo test --manifest-path crates/rocm-oxide-kernel/Cargo.toml
cargo test --manifest-path tools/rocm-oxide-build/Cargo.toml
cargo test --manifest-path tools/cargo-rocm-oxide/Cargo.toml
cargo test --manifest-path tools/oxide-boost/Cargo.toml
cargo run --manifest-path tools/oxide-boost/Cargo.toml -- doctor
cargo run --manifest-path tools/oxide-boost/Cargo.toml -- analyze /home/jack/.local/share/Steam/steamapps/common/Megabonk/
cargo run --manifest-path tools/oxide-boost/Cargo.toml -- analyze --edge /home/jack/.local/share/Steam/steamapps/common/Megabonk/
cargo run --manifest-path tools/oxide-boost/Cargo.toml -- deep-analyze /home/jack/.local/share/Steam/steamapps/common/Megabonk/
rm -rf target/oxide-boost-smoke
mkdir -p target/oxide-boost-smoke/game/data target/oxide-boost-smoke/home
printf original > target/oxide-boost-smoke/game/data/shader.cache
printf modified > target/oxide-boost-smoke/new-shader.cache
OXIDE_BOOST_HOME="$ROOT/target/oxide-boost-smoke/home" cargo run --manifest-path tools/oxide-boost/Cargo.toml -- patch apply --profile smoke --game-dir target/oxide-boost-smoke/game --target data/shader.cache --modified target/oxide-boost-smoke/new-shader.cache
test "$(cat target/oxide-boost-smoke/game/data/shader.cache)" = modified
OXIDE_BOOST_HOME="$ROOT/target/oxide-boost-smoke/home" cargo run --manifest-path tools/oxide-boost/Cargo.toml -- patch restore --profile smoke --game-dir target/oxide-boost-smoke/game --target data/shader.cache
test "$(cat target/oxide-boost-smoke/game/data/shader.cache)" = original
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
cargo run
