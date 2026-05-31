# ROCm-Oxide Implementation Tasks

This list tracks the next core implementation work after removing the old
side-tool path.
The order is intentional: tighten the kernel ABI first, then build higher-level
features on top of stronger contracts.

## Active Sequence

- [ ] Typed device slices:
  - [x] add `DeviceSlice<T>` and `DeviceSliceMut<T>` to device code
  - [x] mirror the ABI shape on the host side
  - [x] teach generated bindings to pass pointer/length pairs automatically
  - [x] reject obvious mutable-buffer aliasing before launch
  - [x] convert simple kernels before large demo kernels
  - [x] convert image, upscaling, stress, and raytrace kernels
- [ ] Constant/global memory source markers:
  - [ ] add a marker such as `#[device_global]` or `#[constant]`
  - [ ] lower marked globals with ROCm address-space awareness
  - [ ] connect marked globals to typed host views
- [ ] Math intrinsic lowering:
  - [ ] map common `f32`/`f64` math calls to AMDGPU/ROCm-supported lowering
  - [ ] add tests for `sqrt`, `rsqrt`, `sin`, `cos`, `atan`, min/max, and NaN behavior
- [ ] Explicit memory-scope atomics:
  - [ ] expose workgroup/device/system scope where ROCm supports it
  - [ ] keep relaxed/basic atomics as the compatibility path
- [ ] Generated lazy operations:
  - [ ] allow generated kernel bindings to return `DeviceOperation` values
  - [ ] support stream-pool scheduling without eager launch
  - [ ] keep the immediate launch API as a convenience wrapper

## Later

- [ ] Direct exported generic-kernel monomorphization without wrapper functions.
- [ ] ROCm-specific replacements for CUDA cluster launch, TMA, and WGMMA concepts.
- [ ] ROCm code-object artifact linking layer once the basic artifact model is stable.
