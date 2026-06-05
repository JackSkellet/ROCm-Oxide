# Path Reconstruction

HIPRTC path tracing, reconstruction, denoise, and interactive motion variants
presented through the shared Vulkan presenter.

```sh
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_path_reconstruction -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_fixed4 -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_motion_denoise_v2 -- --frames 300
```
