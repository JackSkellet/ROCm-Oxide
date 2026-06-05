# ROCm-Oxide Demo Project Catalogue

This directory contains separated demo projects. Each project owns its own
manifest, README, run command, source tree, and demo-only dependencies. The root
SDK crate keeps the smaller examples needed to understand ROCm-Oxide as a
Rust-first ROCm SDK.

## Projects

| Project | Source | Run command | Notes |
| --- | --- | --- | --- |
| `demo-projects/vulkan-plasma/` | `src/main.rs` | `cd demo-projects/vulkan-plasma && ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run -- --frames 300` | CPU-generated frame used to smoke-test the Vulkan presenter path. |
| `demo-projects/spectral-lattice/` | `src/main.rs` | `cd demo-projects/spectral-lattice && cargo run -- --present vulkan --frames 300` | Large generated-kernel workbench with multiple presentation backends, screenshots, library probes, and UI state. |
| `demo-projects/matrix-lens/` | `src/main.rs` | `cd demo-projects/matrix-lens && cargo run -- --capture auto --resolution 720p --mode matrix` | Capture-heavy desktop lens with PipeWire, Wayland, xcap, dma-buf, Vulkan, and fallback capture paths. |
| `demo-projects/window-effects-lab/` | `src/main.rs` | `cd demo-projects/window-effects-lab && cargo run -- --present vulkan --frames 300 0` | Captured-window GPU effects pipeline with desktop/window selection and overlay UI. |
| `demo-projects/path-reconstruction/` | `src/bin/*.rs` | `cd demo-projects/path-reconstruction && ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_path_reconstruction -- --frames 300` | Vulkan-presented path tracing, reconstruction, denoise, and interactive variants. |
| `demo-projects/orbit-field/` | `src/main.rs` | `cd demo-projects/orbit-field && ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run -- --frames 300` | Self-contained HIPRTC procedural field rendered through the shared presenter. |
| `demo-projects/gravity-storm/` | `src/main.rs` | `cd demo-projects/gravity-storm && cargo run` | Vulkan particle app with custom interaction and external-memory presentation details. |
| `demo-projects/stress-gui/` | `src/bin/*.rs` | `cd demo-projects/stress-gui && cargo run --bin stress_test_gui -- --present vulkan --frames 300` | Bounded 2D and 3D visual stress tools with shared controls and presenter dependencies. |
| `demo-projects/raytrace-world/` | `src/main.rs` | `cd demo-projects/raytrace-world && cargo run -- --present vulkan --frames 300` | Interactive visual app with camera controls and generated Rust-device kernels. |
| `demo-projects/rainbow-geometry/` | `src/main.rs` | `cd demo-projects/rainbow-geometry && cargo run -- --present vulkan --frames 300` | Minimal generated-kernel visual frame. |
| `demo-projects/possibilities-window/` | `src/main.rs` | `cd demo-projects/possibilities-window && cargo run -- --present vulkan --frames 300` | Broad visual showcase with tabs, overlays, generated kernels, and contract checks. |
| `demo-projects/compiler-feature-lab/` | `src/main.rs` | `cd demo-projects/compiler-feature-lab && cargo run -- --present vulkan --frames 1` | Dashboard-style probe runner; useful as a demo or diagnostic app. |
| `demo-projects/upscale-artifacts/` | `src/bin/*.rs` | `cd demo-projects/upscale-artifacts && cargo run --bin depth_aware_upscale` | Image-output demos that document generated files and visual expectations. |
| `demo-projects/bvh-raytrace-benchmark/` | `src/main.rs` | `cd demo-projects/bvh-raytrace-benchmark && cargo run` | Benchmark/artifact workload with image outputs and performance expectations. |

## Shared Build Helper

Generated-kernel demo projects use `build.rs` files that include
`demo-projects/shared/device_build.rs`. That helper invokes
`tools/rocm-oxide-build` against the repository `device-spike/` crate and copies
the generated HSACO, metadata, manifest, and bindings into each demo crate's
`OUT_DIR`.

HIPRTC-only visual demos do not need this helper; they compile kernels at
runtime through the `rocm-oxide` path dependency.

## Root Examples

The root `examples/` directory intentionally stays small:

- `hello_gpu`
- `hello_gpu_rust`
- `vector_add`
- `module_global`
- `pinned_stream_vector_add`
- `device_operation_chain`
- `feature_showcase`
- `rust_device_add_one`
- `rust_device_vector_add`
- `rust_device_generated_bindings`
- `validation_profile`
- `performance_probe`
