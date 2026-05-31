# CUDA Oxide Parity Checklist

This checklist tracks the AMD/ROCm equivalents of the official
`NVlabs/cuda-oxide` feature surface. Demos are treated as regression coverage,
not the product.

## Implemented In This Prototype

- [x] Rust-authored device crate compiled to AMDGPU LLVM IR.
- [x] `#[kernel]` allowlist for launchable device entry points.
- [x] LLVM IR post-pass from Rust functions to AMDGPU/HSA kernels.
- [x] `.hsaco` generation through ROCm `llc` and `clang`.
- [x] Kernel descriptor validation with `llvm-readelf`.
- [x] Generated metadata JSON for device artifacts.
- [x] Generated typed host bindings for marked kernels.
- [x] Host-side launch validation for grid/block shape.
- [x] `LaunchConfig::for_num_elems` default block-size helper.
- [x] Host-side launch-shape validation.
- [x] Source-level per-buffer length contracts.
- [x] Generated validation for mixed-resolution buffer lengths.
- [x] Device-side `DeviceSlice<T>` and `DeviceSliceMut<T>` ABI for simple kernels.
- [x] Generated host bindings scalarize device slices to pointer/length launch args.
- [x] Generated mutable-buffer alias rejection.
- [x] `rocm-oxide-build --doctor` prerequisite check.
- [x] `rocm-oxide-build --inspect-metadata` artifact summary.
- [x] HIP stream wrapper.
- [x] HIP event wrapper and elapsed GPU timing.
- [x] Pinned host buffer wrapper.
- [x] Stream-aware async H2D/D2H copies.
- [x] Stream-aware raw kernel launch.
- [x] Lazy host-side `DeviceOperation` model.
- [x] Stream-pool operation scheduling.
- [x] HIP module global lookup.
- [x] Typed host setters/getters for module globals.
- [x] `rocm-oxide-device` no-std helper crate for AMDGPU intrinsics.
- [x] Dispatch-packet-derived `block_dim_*`, `grid_dim_*`, and `global_id_*`
      helpers so kernels do not need a host-passed `block_x` scalar.
- [x] Dynamic launch-sized LDS pointer helper over Rust's GPU workgroup-memory
      intrinsic.
- [x] Nightly `rust-toolchain.toml` with `rust-src` for building `core` on the
      AMDGPU target.
- [x] Doctor probe that verifies `core` actually builds for `amdgcn-amd-amdhsa`.
- [x] Compiler diagnostics include source spans for marked kernels.
- [x] Compiler IR pointer propagation beyond `getelementptr`.
- [x] Cross-crate kernel discovery and bundling for local path dependencies.
- [x] Captured-environment ABI path through mirrored `repr(C)` structs.
- [x] Generated-kernel performance probe for vector, ABI, 3D stress, and
      raytrace kernels.
- [x] Per-kernel resource inspection for VGPR, SGPR, LDS, private segment,
      kernarg, spills, wavefront, and dynamic stack.
- [x] JSON benchmark snapshots with timings and resource counters.

## High-Priority Remaining Work

- [ ] Implementation task sequence is tracked in
      [implementation-tasks.md](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/docs/implementation-tasks.md).
- [x] Cargo subcommand equivalent to `cargo oxide`:
  - [x] `cargo rocm-oxide build`
  - [x] `cargo rocm-oxide run`
  - [x] `cargo rocm-oxide doctor`
  - [x] `cargo rocm-oxide inspect`
  - [x] `cargo rocm-oxide pipeline`
  - [x] `cargo rocm-oxide new`
- [x] Embedded artifact module:
  - [x] copy `.hsaco`, metadata, and bindings into `OUT_DIR`
  - [x] load embedded bytes instead of filesystem paths
  - [x] preserve debug mode that writes artifacts to disk
- [x] Stronger artifact metadata:
  - [x] scalar ABI width per argument
  - [x] pointer address space per argument
  - [x] max workgroup size
  - [x] static shared/LDS usage
  - [x] VGPR/SGPR counts from code object metadata
- [x] Runtime safety:
  - [x] fallible allocation-size overflow errors instead of panics
  - [x] stream-ordered allocation/free where supported by HIP
  - [x] negative launch tests for generated buffer/block contracts
  - [x] pinned-buffer synchronous copy helpers
- [x] Async execution layer:
  - [x] lazy `DeviceOperation` trait
  - [x] sync and async execution entry points
  - [x] operation chaining
  - [x] stream pool scheduling
  - [x] keep in-flight results alive if futures are dropped
- [x] Constant/global memory:
  - [x] source marker for device globals
  - [x] address-space-aware lowering for ROCm globals
  - [x] module global lookup through HIP
  - [x] typed host setters/getters
  - [x] load-time size validation for typed host views
- [ ] Device API surface:
  - [x] thread/block/grid helpers instead of raw `core::arch::amdgpu`
  - [x] warp/wavefront helpers
  - [x] barriers
  - [x] dynamic shared memory/LDS wrappers
  - [x] typed device slices for pointer/length kernel ABI
  - [x] basic `u32` atomics
  - [x] explicit memory-scope atomics
  - [x] math intrinsic lowering
- [ ] Compiler completeness:
  - [x] support more pointer-producing IR ops beyond `getelementptr`
  - [x] preserve source signature and contract spans in diagnostics
  - [x] catch device-codegen panics and emit actionable diagnostics
  - [x] generic helper monomorphization tests with diagnostics for exported generic kernels
  - [x] closure/captured environment ABI support through `repr(C)` environment structs
  - [x] cross-crate kernel discovery and bundling for local path dependencies
  - [ ] direct exported generic-kernel monomorphization without wrapper functions

## Lower-Priority Or Vendor-Specific

- [ ] CUDA-specific TMA/WGMMA equivalents need ROCm-specific replacements, not
      direct ports.
- [ ] cuBLASDx/cuFFTDx interop maps to rocBLAS/rocFFT or HIP library FFI and
      should be a separate integration layer.
- [ ] NVVM/LTOIR and nvJitLink concepts map only partially to ROCm code-object
      linking; implement after the basic artifact model is stable.
- [ ] Fine-grained LLVM `syncscope` selection for ROCm atomics should follow the
      typed scoped API once the backend has a source marker path for that IR.
