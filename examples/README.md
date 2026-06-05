# ROCm-Oxide Root Example Catalogue

The root `examples/` directory is now reserved for the SDK learning path,
diagnostics, and release gates. Larger visual, capture, artifact, benchmark, and
experimental apps live in `demo-projects/`, where each project owns its manifest,
README, dependencies, and run commands.

Generated consumer projects created by `cargo rocm-oxide new` compile their own
local device crate and do not use the source-workspace `device-spike` feature.

## Canonical SDK Examples

| Example | Path | Run command | Purpose |
| --- | --- | --- | --- |
| Hello HIPRTC | `examples/hello_gpu.rs` | `cargo run --example hello_gpu` | Smallest runtime smoke test: open a device, compile a HIPRTC kernel, launch, sync, and verify. |
| Hello Rust GPU | `examples/hello_gpu_rust.rs` | `cargo run --features device-spike --example hello_gpu_rust` | Main Rust-authored kernel walkthrough using generated typed bindings. |
| HIPRTC vector add | `examples/vector_add.rs` | `cargo run --example vector_add` | Compact raw `launch!` vector-add example. |
| Module global | `examples/module_global.rs` | `cargo run --example module_global` | HIPRTC module/global lookup and host interaction. |
| Pinned stream vector add | `examples/pinned_stream_vector_add.rs` | `cargo run --example pinned_stream_vector_add` | Pinned host memory plus stream-oriented launch flow. |
| Device operation chain | `examples/device_operation_chain.rs` | `cargo run --example device_operation_chain` | Higher-level device operation and async composition path. |
| GPU algorithms | `examples/gpu_algorithms.rs` | `cargo run --example gpu_algorithms` | High-level `rocm_oxide::gpu` reduce, scan, map, select, and sort helpers backed by rocPRIM/rocThrust. |
| Feature showcase | `examples/feature_showcase.rs` | `cargo run --features device-spike --example feature_showcase` | Broad SDK capability sample covering runtime, generated kernels, library probes, and profiling hooks. |

`scripts/first-user-path.sh` runs the two hello examples plus the local
`cargo rocm-oxide doctor` implementation. Keep it passing whenever the README
first-run commands change.

## Rust-Device Examples

These examples depend on `device-spike/`, `crates/rocm-oxide-device`, and
`crates/rocm-oxide-kernel` through the source-workspace build pipeline.

| Example | Path | Run command | Purpose |
| --- | --- | --- | --- |
| Raw Rust add-one | `examples/rust_device_add_one.rs` | `cargo run --features device-spike --example rust_device_add_one` | Loads the generated HSACO directly and launches by raw kernel name. |
| Raw Rust vector add | `examples/rust_device_vector_add.rs` | `cargo run --features device-spike --example rust_device_vector_add` | Rust-authored vector add using raw code-object loading. |
| Generated bindings | `examples/rust_device_generated_bindings.rs` | `cargo run --features device-spike --example rust_device_generated_bindings` | Typed `DeviceKernels` loading, launch validation, and generated binding ergonomics. |

## Validation And Profiling

These examples are useful as release gates, diagnostics, or hardware-profile
records.

| Example | Path | Run command | Purpose |
| --- | --- | --- | --- |
| Validation profile | `examples/validation_profile.rs` | `cargo run --example validation_profile` | Prints and optionally writes a JSON profile for the current ROCm/GPU environment. |
| Performance probe | `examples/performance_probe.rs` | `cargo run --features device-spike --example performance_probe -- --json target/performance_probe.json` | Measures selected generated-kernel/runtime paths and can write JSON output. |

## Demo Projects

Browse separated demos in `demo-projects/`:

- `demo-projects/vulkan-plasma/`
- `demo-projects/spectral-lattice/`
- `demo-projects/matrix-lens/`
- `demo-projects/window-effects-lab/`
- `demo-projects/path-reconstruction/`
- `demo-projects/orbit-field/`
- `demo-projects/gravity-storm/`
- `demo-projects/stress-gui/`
- `demo-projects/raytrace-world/`
- `demo-projects/raytrace-world-gpuarray/`
- `demo-projects/rainbow-geometry/`
- `demo-projects/possibilities-window/`
- `demo-projects/compiler-feature-lab/`
- `demo-projects/upscale-artifacts/`
- `demo-projects/bvh-raytrace-benchmark/`

For visual launch commands, see `../docs/visual-demos.md`. For the full demo
catalogue, see `../demo-projects/README.md`.
