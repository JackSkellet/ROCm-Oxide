# Visual Demo Reference

Visual, capture, artifact, and benchmark demos now live in `demo-projects/`.
Each demo has its own `Cargo.toml` and README so root SDK builds do not need
windowing, Vulkan, image, capture, or presenter dependencies.

## Launch Table

| Demo project | What it shows | Typical launch |
| --- | --- | --- |
| `demo-projects/vulkan-plasma/` | CPU-generated Vulkan-presenter smoke test. | `cd demo-projects/vulkan-plasma && ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run -- --frames 300` |
| `demo-projects/spectral-lattice/` | Interactive generated-kernel workbench with overlays, screenshots, CPU/GL/Vulkan presentation modes, and library probes. | `cd demo-projects/spectral-lattice && cargo run -- --present vulkan --resolution 1440p --fps-limit 120` |
| `demo-projects/matrix-lens/` | Vulkan-only desktop lens with matrix, glass, thermal, and xray effects over captured desktop content. | `cd demo-projects/matrix-lens && cargo run -- --capture auto --resolution 720p --mode matrix` |
| `demo-projects/rainbow-geometry/` | Minimal generated Rust kernel drawing a 2D rainbow/geometry frame. | `cd demo-projects/rainbow-geometry && cargo run -- --present vulkan --frames 300` |
| `demo-projects/stress-gui/` | 2D and 3D ALU/bitwise stress patterns with bounded interactive work controls. | `cd demo-projects/stress-gui && cargo run --bin stress_test_gui -- --present vulkan --frames 300` |
| `demo-projects/raytrace-world/` | Interactive raytraced world with camera movement, shadows, and reflections. | `cd demo-projects/raytrace-world && cargo run -- --present vulkan --frames 300` |
| `demo-projects/possibilities-window/` | Combined showcase for generated kernels, runtime contract checks, HIPRTC/module globals, post effects, ray flags, and clickable tabs. | `cd demo-projects/possibilities-window && cargo run -- --present vulkan --frames 300` |
| `demo-projects/compiler-feature-lab/` | CPU-rasterized dashboard of compiler/runtime/library probe results. | `cd demo-projects/compiler-feature-lab && cargo run -- --present vulkan --frames 1` |
| `demo-projects/window-effects-lab/` | Captured-window effects pipeline with GPU upscaling/post effects and a left-side control panel. | `cd demo-projects/window-effects-lab && cargo run -- --present vulkan --frames 300 0` |
| `demo-projects/path-reconstruction/` | HIPRTC path tracing, reconstruction, denoise, and interactive motion variants. | `cd demo-projects/path-reconstruction && ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_path_reconstruction -- --frames 300` |
| `demo-projects/orbit-field/` | HIPRTC procedural field rendered through the shared presenter. | `cd demo-projects/orbit-field && ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run -- --frames 300` |
| `demo-projects/gravity-storm/` | HIPRTC plus Vulkan particle simulation with external-memory presentation details. | `cd demo-projects/gravity-storm && cargo run` |
| `demo-projects/upscale-artifacts/` | Generated-kernel PNG artifact demos. | `cd demo-projects/upscale-artifacts && cargo run --bin depth_aware_upscale` |
| `demo-projects/bvh-raytrace-benchmark/` | Generated-kernel raytrace benchmark and comparison artifacts. | `cd demo-projects/bvh-raytrace-benchmark && cargo run` |

## Shared Presenter Notes

Smaller windowed demo projects carry a local `src/visual_presenter.rs` copied
from the original shared presenter. Their default backend is `minifb`; select
the Vulkan backend with `--present vulkan` or
`ROCM_OXIDE_VISUAL_PRESENT=vulkan`.

In Vulkan mode, GPU-rendered frames are copied device-to-device into
Vulkan-owned `OPAQUE_FD` memory imported by HIP. Demos with CPU UI panels copy
only their overlay rectangles from host memory after the GPU frame is copied.
