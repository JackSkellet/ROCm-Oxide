# ROCm-Oxide Example Catalogue

This catalogue describes the current flat `examples/*.rs` surface. It is a
navigation aid for the SDK preview restructure; it does not imply that any
source has already moved.

Most examples fall into one or more explicit build paths:

- Core HIPRTC/runtime examples compile HIP C++ strings at runtime and do not
  need a feature flag.
- Rust-device examples use the repository `device-spike/` crate and must run
  with `--features device-spike` in this source workspace.
- Visual examples use demo-only UI, image, Vulkan, SDL, or presenter
  dependencies and must run with `--features visual-demos`.
- Capture examples use desktop capture dependencies and must run with
  `--features capture-demos`; that feature includes `visual-demos`.

Generated consumer projects created by `cargo rocm-oxide new` compile their own
local device crate and do not use the source-workspace `device-spike` feature.

## Canonical SDK Examples

These are the examples that should stay close to the root SDK surface during the
first cleanup pass.

| Example | Path | Run command | Purpose |
| --- | --- | --- | --- |
| Hello HIPRTC | `examples/hello_gpu.rs` | `cargo run --example hello_gpu` | Smallest runtime smoke test: open a device, compile a HIPRTC kernel, launch, sync, and verify. |
| Hello Rust GPU | `examples/hello_gpu_rust.rs` | `cargo run --features device-spike --example hello_gpu_rust` | Main Rust-authored kernel walkthrough using generated typed bindings. |
| HIPRTC vector add | `examples/vector_add.rs` | `cargo run --example vector_add` | Compact raw `launch!` vector-add example. |
| Module global | `examples/module_global.rs` | `cargo run --example module_global` | HIPRTC module/global lookup and host interaction. |
| Pinned stream vector add | `examples/pinned_stream_vector_add.rs` | `cargo run --example pinned_stream_vector_add` | Pinned host memory plus stream-oriented launch flow. |
| Device operation chain | `examples/device_operation_chain.rs` | `cargo run --example device_operation_chain` | Higher-level device operation and async composition path. |
| Feature showcase | `examples/feature_showcase.rs` | `cargo run --features device-spike --example feature_showcase` | Broad SDK capability sample covering runtime, generated kernels, library probes, and profiling hooks. |

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
records rather than first-user tutorials.

| Example | Path | Run command | Purpose |
| --- | --- | --- | --- |
| Validation profile | `examples/validation_profile.rs` | `cargo run --example validation_profile` | Prints and optionally writes a JSON profile for the current ROCm/GPU environment. |
| Performance probe | `examples/performance_probe.rs` | `cargo run --features device-spike --example performance_probe -- --json target/performance_probe.json` | Measures selected generated-kernel/runtime paths and can write JSON output. |
| BVH raytrace benchmark | `examples/bvh_raytrace_benchmark.rs` | `cargo run --features 'device-spike visual-demos' --example bvh_raytrace_benchmark` | Benchmark/artifact workload for generated Rust-device raytracing kernels. |

## Visual Demos

These are currently root examples, but most are better candidates for separated
demo projects because they bring UI, image, Vulkan, SDL, minifb, or presenter
dependencies into the SDK crate.

| Demo | Path | Typical command | Notes |
| --- | --- | --- | --- |
| Spectral lattice | `examples/spectral_lattice.rs` | `cargo run --features 'device-spike visual-demos' --example spectral_lattice -- --present vulkan` | Interactive workbench with generated kernels, overlays, screenshots, and multiple presentation paths. |
| Rainbow geometry | `examples/rainbow_geometry_window.rs` | `cargo run --features 'device-spike visual-demos' --example rainbow_geometry_window -- --present vulkan --frames 300` | Minimal generated-kernel visual frame. |
| Raytrace world | `examples/raytrace_world_gui.rs` | `cargo run --features 'device-spike visual-demos' --example raytrace_world_gui -- --present vulkan --frames 300` | Interactive raytraced world with camera controls. |
| Possibilities window | `examples/possibilities_window.rs` | `cargo run --features 'device-spike visual-demos' --example possibilities_window -- --present vulkan --frames 300` | Combined showcase for generated kernels, contract checks, module globals, post effects, and clickable tabs. |
| Stress 2D GUI | `examples/stress_test_gui.rs` | `cargo run --features 'device-spike visual-demos' --example stress_test_gui -- --present vulkan --frames 300` | Bounded 2D ALU/bitwise stress patterns. |
| Stress 3D GUI | `examples/stress_3d_gui.rs` | `cargo run --features 'device-spike visual-demos' --example stress_3d_gui -- --present vulkan --frames 300` | Bounded 3D ray/volume-style stress patterns. |
| Compiler feature lab | `examples/compiler_feature_lab.rs` | `cargo run --features 'device-spike visual-demos' --example compiler_feature_lab -- --present vulkan --frames 1` | Visual dashboard for compiler/runtime/library probes. |
| Gravity storm | `examples/gravity_storm.rs` | `cargo run --features visual-demos --example gravity_storm` | HIPRTC plus Vulkan particle simulation using external-memory style presentation. |
| Vulkan orbit field | `examples/vulkan_orbit_field.rs` | `ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --features visual-demos --example vulkan_orbit_field -- --frames 300` | HIPRTC procedural field rendered through the shared presenter. |
| Vulkan path reconstruction | `examples/vulkan_path_reconstruction.rs` | `ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --features visual-demos --example vulkan_path_reconstruction -- --frames 300` | HIPRTC path tracing, reconstruction, denoise, and tonemap path. |
| Interactive path reconstruction | `examples/vulkan_interactive_path_reconstruction_fixed4.rs` | `ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --features visual-demos --example vulkan_interactive_path_reconstruction_fixed4 -- --frames 300` | Interactive variant of the path reconstruction demo. |
| Motion-denoise path reconstruction | `examples/vulkan_interactive_path_reconstruction_motion_denoise_v2.rs` | `ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --features visual-demos --example vulkan_interactive_path_reconstruction_motion_denoise_v2 -- --frames 300` | Interactive reconstruction variant with extra samples during motion. |

For launch options on the shared visual presenter, see
`docs/visual-demos.md`.

Separated visual demo projects now live under `demo-projects/`. The first moved
demo is `demo-projects/vulkan-plasma/`, a small Vulkan-presenter smoke test
with its own manifest and run command.

## Capture Demos

Capture demos should move out of the root SDK early because they depend on
desktop compositor, PipeWire, xcap, Wayland, or live-window behavior.

| Demo | Path | Typical command | Notes |
| --- | --- | --- | --- |
| Matrix lens | `examples/matrix_lens.rs` | `cargo run --features 'device-spike capture-demos' --example matrix_lens -- --capture auto --resolution 720p --mode matrix` | Desktop lens with dma-buf, video, and pattern capture paths. |
| Window effects lab | `examples/window_effects_lab.rs` | `cargo run --features 'device-spike capture-demos' --example window_effects_lab -- --present vulkan --frames 300 0` | Captured-window effects pipeline with GPU upscaling/post effects and a control panel. |

## Artifact Demos

These examples generate image artifacts or reconstruction outputs and are better
documented as demos than as SDK tutorials.

| Demo | Path | Run command | Notes |
| --- | --- | --- | --- |
| Depth-aware upscale | `examples/depth_aware_upscale.rs` | `cargo run --features 'device-spike visual-demos' --example depth_aware_upscale` | Writes color, depth, and edge-mask artifacts. |
| Temporal upscale | `examples/temporal_upscale.rs` | `cargo run --features 'device-spike visual-demos' --example temporal_upscale` | Writes temporal upscaling artifacts. |

## Shared Support

| Path | Current role | Restructure note |
| --- | --- | --- |
| `examples/shared/visual_presenter.rs` | Shared minifb/Vulkan presenter used by multiple visual demos. | Should move with visual demo projects or become a small demo-support crate, not part of the core SDK path. |
