# Visual Demo Reference

This page is the quick launch guide for ROCm-Oxide's visual examples.

The smaller windowed demos share `examples/shared/visual_presenter.rs`. Their
default backend is `minifb`; select the Vulkan backend with `--present vulkan`
or `ROCM_OXIDE_VISUAL_PRESENT=vulkan`. In Vulkan mode, GPU-rendered frames are
copied device-to-device into Vulkan-owned `OPAQUE_FD` memory imported by HIP.
Demos with CPU UI panels copy only their overlay rectangles from host memory
after the GPU frame is copied; they no longer read the full frame back to host
for presentation.

| Demo | What it shows | Typical launch | Launch options | Presentation/performance path |
| --- | --- | --- | --- | --- |
| `spectral_lattice` | Interactive GPU workbench with core, LDS, atomics, post effects, overlay UI, screenshots, and CPU/GL/Vulkan presentation modes. | `cargo run --features 'device-spike visual-demos' --example spectral_lattice -- --present vulkan --resolution 1440p --fps-limit 120` | `--frames N`, `--output PATH`, `--mode Core|LDS|Atomic|Chain|1-4`, `--resolution 540p|720p|1080p|1440p|4k|WIDTHxHEIGHT`, `--fps-limit FPS|uncapped`, `--gpu-work N`, `--present cpu|gl|vulkan`, `--present-scale 1|2|4` | `--present vulkan` uses Vulkan exportable device memory imported by HIP, then a device-to-device copy and Vulkan blit. `--present gl` uses HIP/OpenGL PBO interop. Default `cpu` keeps compatibility readback. |
| `matrix_lens` | Vulkan-only desktop lens with matrix, glass, thermal, and xray effects over captured desktop content. | `cargo run --features 'device-spike capture-demos' --example matrix_lens -- --capture auto --resolution 720p --mode matrix` | `--frames N`, `--output PATH`, `--mode matrix|glass|thermal|xray|0-3`, `--capture auto|dmabuf|video|pattern`, `--resolution 540p|720p|1080p|WIDTHxHEIGHT`, `--fps-limit FPS|uncapped`; `ROCM_OXIDE_MATRIX_CAPTURE_WARMUP_MS` tunes bounded live-capture warmup. | Vulkan-only. The fast path imports compositor dma-buf input into Vulkan and renders into HIP-imported Vulkan output memory. `video` and `pattern` are fallbacks for compositor or test constraints. |
| `rainbow_geometry_window` | Minimal generated Rust kernel drawing a 2D rainbow/geometry frame. | `cargo run --features 'device-spike visual-demos' --example rainbow_geometry_window -- --present vulkan --frames 300` | Shared: `--present vulkan`, `--present=vk`, `--frames N`, `ROCM_OXIDE_RAINBOW_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_PRESENT=vulkan`. | Vulkan mode presents the `DeviceBuffer<u32>` through HIP-imported Vulkan memory; no full-frame host readback. |
| `stress_test_gui` | 2D ALU/bitwise stress patterns with bounded interactive work controls. | `cargo run --features 'device-spike visual-demos' --example stress_test_gui -- --present vulkan --frames 300` | Shared: `--present vulkan`, `--frames N`, `ROCM_OXIDE_STRESS_TEST_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`; runtime keys: Left/Right mode, Up/Down/PgUp/PgDn work, `0-3` presets, Space pause. | Vulkan mode presents the GPU output buffer device-to-device; work iterations are clamped to 4096. |
| `stress_3d_gui` | 3D ray/volume-style stress patterns with bounded step controls. | `cargo run --features 'device-spike visual-demos' --example stress_3d_gui -- --present vulkan --frames 300` | Shared: `--present vulkan`, `--frames N`, `ROCM_OXIDE_STRESS_3D_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`; runtime keys: Left/Right mode, Up/Down/PgUp/PgDn steps, `1-4` presets, Space pause. | Vulkan mode presents the GPU output buffer device-to-device; step count is clamped to 4096. |
| `raytrace_world_gui` | Interactive raytraced world with camera movement, shadows, and reflections. | `cargo run --features 'device-spike visual-demos' --example raytrace_world_gui -- --present vulkan --frames 300` | Shared: `--present vulkan`, `--frames N`, `ROCM_OXIDE_RAYTRACE_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`; `ROCM_OXIDE_RAYTRACE_TOGGLE_REFLECTIONS_AT=N` for tests; runtime keys: WASD move, arrows look, Shift speed, `1` shadows, `2` reflections, Space pause, `R` reset. | Vulkan mode keeps the rendered frame on the GPU and presents it through HIP-imported Vulkan memory. Camera parameters are small host-to-device uploads. |
| `possibilities_window` | Combined showcase for generated kernels, runtime contract checks, HIPRTC/module globals, post effects, ray flags, and clickable tabs. | `cargo run --features 'device-spike visual-demos' --example possibilities_window -- --present vulkan --frames 300` | Shared: `--present vulkan`, `--frames N`, `ROCM_OXIDE_WINDOW_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`; runtime keys: `1-4` modes, arrows mode/work, `P` post effect, `C` contract check, `S/R` ray flags, Space pause. | Vulkan mode copies the GPU frame device-to-device, then copies only the fixed CPU overlay panels/buttons into the swapchain image. Full-frame readback was removed from the presentation loop. |
| `compiler_feature_lab` | CPU-rasterized dashboard of compiler/runtime/library probe results. | `cargo run --features 'device-spike visual-demos' --example compiler_feature_lab -- --present vulkan --frames 1` | Shared: `--present vulkan`, `--frames N`, `ROCM_OXIDE_FEATURE_LAB_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`; runtime keys: `1-0` select probes, Left/Right cycle, Up/Down scale, Space pause, `R` rerun probes. | Probe kernels still read back small result vectors by design. The visual frame is CPU-rendered directly into presenter-owned memory; Vulkan mode uploads that frame through the shared presenter. |
| `window_effects_lab` | Captured-window effects pipeline with GPU upscaling/post effects and a left-side control panel. | `cargo run --features 'device-spike capture-demos' --example window_effects_lab -- --present vulkan --frames 300 0` | Shared: `--present vulkan`, `--frames N`, `ROCM_OXIDE_WINDOW_FX_MAX_FRAMES`, `ROCM_OXIDE_VISUAL_MAX_FRAMES`; `ROCM_OXIDE_WINDOW_FX_TARGET` chooses a captured window; optional positional selector is an index or title substring; runtime keys: Left/Right mode, Up/Down sharpness, Space freeze. | Vulkan mode copies the GPU output device-to-device, then copies only the left control panel from host memory. Capture input can still require host upload depending on compositor path. |

## Shared Presenter Notes

| Need | Use |
| --- | --- |
| Force Vulkan for one run | `cargo run --features '<required demo features>' --example <demo> -- --present vulkan` |
| Force Vulkan for all shared demos in a shell | `export ROCM_OXIDE_VISUAL_PRESENT=vulkan` |
| Run bounded smoke frames | `cargo run --features '<required demo features>' --example <demo> -- --present vulkan --frames 1` |
| Cap all shared demos without editing commands | `ROCM_OXIDE_VISUAL_MAX_FRAMES=1` |
| Use compatibility presenter | omit `--present` or set `ROCM_OXIDE_VISUAL_PRESENT=minifb` |

`spectral_lattice` and `matrix_lens` have their own specialized presenters
because they support resolution changes, external input capture, GL interop, or
more involved overlay composition. The smaller demos use the shared presenter.
