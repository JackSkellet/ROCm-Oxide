# ROCm-Oxide Demo Project Catalogue

This directory is reserved for separated demo projects. No source has been moved
here yet; the table below records the first-pass split candidates from the
current flat `examples/` directory.

Each future demo project should own its own:

- `Cargo.toml`
- `README.md`
- run command
- expected output, screenshot, or artifact list
- hardware, display-server, and ROCm notes
- source tree and demo-only dependencies

The root SDK crate should keep only canonical examples needed to understand
ROCm-Oxide as a Rust-first ROCm SDK.

## Candidate Demo Projects

| Future project | Current source | Why it should split |
| --- | --- | --- |
| `demo-projects/spectral-lattice/` | `examples/spectral_lattice.rs` | Large interactive workbench with multiple presentation backends, screenshots, library probes, and UI state. |
| `demo-projects/matrix-lens/` | `examples/matrix_lens.rs` | Capture-heavy desktop lens with PipeWire, Wayland, xcap, dma-buf, Vulkan, and fallback capture paths. |
| `demo-projects/window-effects-lab/` | `examples/window_effects_lab.rs` | Captured-window GPU effects pipeline with desktop/window selection and overlay UI. |
| `demo-projects/path-reconstruction/` | `examples/vulkan_path_reconstruction.rs`, `examples/vulkan_interactive_path_reconstruction_fixed4.rs`, `examples/vulkan_interactive_path_reconstruction_motion_denoise_v2.rs` | Related Vulkan-presented path tracing and reconstruction variants that need their own README and controls table. |
| `demo-projects/orbit-field/` | `examples/vulkan_orbit_field.rs` | Self-contained HIPRTC visual demo that belongs with visual apps, not SDK tutorials. |
| `demo-projects/gravity-storm/` | `examples/gravity_storm.rs` | Vulkan particle app with custom interaction and external-memory presentation details. |
| `demo-projects/stress-gui/` | `examples/stress_test_gui.rs`, `examples/stress_3d_gui.rs` | Bounded visual stress tools with shared controls and presenter dependencies. |
| `demo-projects/raytrace-world/` | `examples/raytrace_world_gui.rs` | Interactive visual app with camera controls and generated Rust-device kernels. |
| `demo-projects/possibilities-window/` | `examples/possibilities_window.rs` | Broad visual showcase with tabs, overlays, generated kernels, and contract checks. |
| `demo-projects/compiler-feature-lab/` | `examples/compiler_feature_lab.rs` | Dashboard-style probe runner; useful as a demo or diagnostic app, not a first-user SDK path. |
| `demo-projects/upscale-artifacts/` | `examples/depth_aware_upscale.rs`, `examples/temporal_upscale.rs` | Image-output demos that should document generated files and visual expectations. |
| `demo-projects/bvh-raytrace-benchmark/` | `examples/bvh_raytrace_benchmark.rs` | Benchmark/artifact workload with image outputs and performance expectations. |

## Root Examples To Keep Canonical

The likely root example set after the split is:

- `hello_gpu`
- `hello_gpu_rust`
- `vector_add`
- `module_global`
- `pinned_stream_vector_add`
- `device_operation_chain`
- `rust_device_add_one`
- `rust_device_vector_add`
- `rust_device_generated_bindings`
- `validation_profile`
- `performance_probe`

`feature_showcase` is useful during the SDK preview, but it may become either a
root integration example or a separated diagnostic project depending on how much
library/profiling surface remains in the core crate.

## Dependency Boundary

The first split should move demo-only dependencies out of the root crate where
possible:

| Dependency area | Current reason it appears | Preferred future owner |
| --- | --- | --- |
| `visual-demos` feature | visual demos and shared presenter | visual demo projects or a demo-support crate |
| `capture-demos` feature | matrix lens and window effects | capture demo projects |
| Image output crates | artifact demos and screenshots | artifact/visual demo projects |
| Vulkan/SDL/minifb paths | visual presentation | visual demo projects |
| Rust-device build path | canonical Rust GPU examples and generated-kernel demos | keep for canonical source-workspace examples; duplicate per separated demo when needed |
