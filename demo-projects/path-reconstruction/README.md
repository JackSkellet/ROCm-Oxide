# Path Reconstruction

HIPRTC path tracing, reconstruction, denoise, and interactive motion variants
presented through the shared Vulkan presenter.

```sh
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_path_reconstruction -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_fixed4 -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_motion_denoise_v2 -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_motion_denoise_v2_gpuarray -- --frames 300
```

## GpuArray Comparison

`vulkan_interactive_path_reconstruction_motion_denoise_v2_gpuarray` is a copy
of the motion-denoise v2 demo that uses `GpuArray<T>` for the host-owned frame,
accumulation, and parameter buffers. The HIPRTC kernel source is intentionally
unchanged so the two binaries are directly comparable.

```text
Original motion-denoise v2 binary: 862 lines
GpuArray copy binary:             862 lines
Delta:                              0 lines
```

Reproduce the measurement with:

```sh
wc -l \
  src/bin/vulkan_interactive_path_reconstruction_motion_denoise_v2.rs \
  src/bin/vulkan_interactive_path_reconstruction_motion_denoise_v2_gpuarray.rs
```
