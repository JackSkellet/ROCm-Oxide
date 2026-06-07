# Path Reconstruction

Experimental application-style HIPRTC path tracing, reconstruction, denoise,
and interactive motion variants presented through the shared Vulkan presenter.
This demo lives under `demo-projects/` because it is a larger downstream-style
app, not a minimal SDK example.

## What It Demonstrates

The binaries in this crate exercise:

- HIPRTC kernel compilation from host Rust.
- Progressive GPU accumulation buffers.
- GPU edge-aware reconstruction, denoise, and tonemap passes.
- Vulkan presentation of GPU-rendered frames.
- Interactive camera, object, light, material, and exposure controls.
- A direct `DeviceBuffer<T>` versus `GpuArray<T>` host-buffer comparison.

In this repo, "reconstruction" means a compact GPU post process that combines
progressive path-traced samples with an edge-aware spatial filter and filmic
tonemapping. It is not NVIDIA/DLSS neural ray reconstruction, does not use a
neural model, and does not implement optical-flow or frame-generation features.

## Binaries

```sh
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_path_reconstruction -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_fixed4 -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_motion_denoise_v2 -- --frames 300
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --bin vulkan_interactive_path_reconstruction_motion_denoise_v2_gpuarray -- --frames 300
```

| Binary | Resolution | What it shows |
| --- | --- | --- |
| `vulkan_path_reconstruction` | 960x540 | Fixed scene path tracing with exposure, aperture, focus, reconstruction toggle, and accumulation reset controls. |
| `vulkan_interactive_path_reconstruction_fixed4` | 960x540 | Interactive camera/object/light scene using `DeviceBuffer<T>` for frame, accumulation, and parameter buffers. |
| `vulkan_interactive_path_reconstruction_motion_denoise_v2` | 1920x1080 | Higher-resolution interactive variant that traces extra samples while motion is happening and then returns to normal progressive accumulation once stable. |
| `vulkan_interactive_path_reconstruction_motion_denoise_v2_gpuarray` | 1920x1080 | Same HIPRTC kernels and controls as motion-denoise v2, but host-owned frame, accumulation, and parameter buffers use `GpuArray<T>`. |

All binaries accept `--frames N` through the shared presenter helper. When no
CLI frame limit is supplied, `vulkan_path_reconstruction` also reads
`ROCM_OXIDE_PATH_RECONSTRUCTION_FRAMES`, while the interactive binaries read
`ROCM_OXIDE_INTERACTIVE_PATH_RECON_FRAMES`. All four binaries also fall back to
`ROCM_OXIDE_VISUAL_MAX_FRAMES`.

The default presenter can fall back to `minifb`; set
`ROCM_OXIDE_VISUAL_PRESENT=vulkan` to force the Vulkan path documented here.

## Controls

`vulkan_path_reconstruction`:

- `Esc`: quit.
- `Space`: toggle reconstruction.
- `R`: reset accumulation.
- `Up` / `Down`: adjust exposure.
- `A` / `D`: adjust aperture and reset accumulation.
- `Left` / `Right`: adjust focus distance and reset accumulation.

Interactive variants:

- `W` / `S`: move camera forward/back.
- `A` / `D`: strafe camera left/right.
- `Left` / `Right`: yaw camera.
- `Up` / `Down`: pitch camera.
- `1`, `2`, `3`, `4`: select object or light.
- `5` / `6`: move selected object or light left/right.
- `7` / `8`: move selected object or light forward/back.
- `PageUp` / `PageDown`: move selected object or light up/down.
- `Space`: toggle reconstruction.
- `R`: reset accumulation.
- `P`: pause/resume animation.
- `C`: cycle material preset.
- `9` / `0`: exposure down/up.
- `LeftShift`: hold for faster movement or adjustment.
- Motion-denoise v2 variants only: `LeftShift` + `Up` / `Down` adjusts focus
  distance instead of pitch.
- `Esc`: quit.

The interactive variants use time-scaled movement so controls remain usable at
both low and high frame rates.

## Motion Preview And Noise

Path tracing is noisy when the accumulation buffer has only a few samples.
Camera moves, object moves, material changes, aperture/focus changes, and manual
resets clear accumulation, so the first frames after a change are expected to be
noisier.

The motion-denoise v2 binaries compensate by doing extra tracing work while the
camera or scene is moving and for the first stable frames after movement. The
reconstruction pass is also forced on during motion/early-stable frames even if
the toggle is off, then returns to the user-selected reconstruction state once
the view has stabilized.

## Performance Expectations

These are visual integration demos, not production renderers or benchmark
contracts. Expected behavior:

- `vulkan_path_reconstruction` and `vulkan_interactive_path_reconstruction_fixed4`
  render at 960x540 and are the lighter variants.
- The motion-denoise v2 binaries render at 1920x1080 and intentionally do more
  work during motion, so they are heavier.
- The Vulkan presenter path can run well above 60 FPS on fast GPUs, which is why
  interactive controls are time-scaled.
- The displayed image should become cleaner as stable frames accumulate.
- Noisy frames during motion or immediately after reset are normal.
- Exact FPS depends on GPU, ROCm version, desktop compositor, presenter backend,
  and whether the scene is moving.

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
