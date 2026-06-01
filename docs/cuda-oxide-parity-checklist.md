# CUDA Oxide Parity Checklist

This checklist tracks the AMD/ROCm equivalents of the official
`NVlabs/cuda-oxide` feature surface. Demos are treated as regression coverage,
not the product.

Primary upstream reference:
[NVIDIA cuda-oxide Supported Features](https://nvlabs.github.io/cuda-oxide/appendix/supported-features.html).

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
- [x] Explicit HIP graph builder for empty nodes, dependencies, memcpy, memset,
      kernel nodes, node retargeting, instantiate/replay, and exec update.
- [x] HIP graph memory allocation/free nodes with a graph-managed allocation
      plan object.
- [x] Owned HIP memory pools with access-policy controls.
- [x] HIP VMM-backed device virtual memory reserve/map/access wrapper.
- [x] rocPRIM/hipCUB-backed `u32` sum reduction plus inclusive/exclusive scan
      wrappers over `DeviceBuffer`.
- [x] Matrix integration candidate reporting for hipBLASLt, Composable Kernel,
      and rocWMMA, plus hipBLASLt handle/version loading.
- [x] HIPRTC specialization cache keyed by backend, architecture, source,
      options, and launch metadata, with COMGR availability probing for a future
      code-object compiler backend.
- [x] Lazy host-side `DeviceOperation` model.
- [x] Stream-pool operation scheduling.
- [x] Generated kernel bindings can return lazy `DeviceOperation` launch jobs.
- [x] Generated kernel bindings can add validated kernel nodes to explicit HIP
      graphs.
- [x] HIP module global lookup.
- [x] Typed host setters/getters for module globals.
- [x] `rocm-oxide-device` no-std helper crate for AMDGPU intrinsics.
- [x] Dispatch-packet-derived `block_dim_*`, `grid_dim_*`, and `global_id_*`
      helpers so kernels do not need a host-passed `block_x` scalar.
- [x] CUDA-like cooperative group handles for thread blocks, wavefronts, and
      static tiles over AMD workgroup/wavefront intrinsics.
- [x] Dynamic launch-sized LDS pointer helper over Rust's GPU workgroup-memory
      intrinsic.
- [x] Nightly `rust-toolchain.toml` with `rust-src` for building `core` on the
      AMDGPU target.
- [x] Doctor probe that verifies `core` actually builds for `amdgcn-amd-amdhsa`.
- [x] Compiler diagnostics include source spans for marked kernels.
- [x] Compiler IR pointer propagation beyond `getelementptr`.
- [x] Cross-crate kernel discovery and bundling for local path dependencies.
- [x] Captured-environment ABI path through mirrored `repr(C)` structs.
- [x] Generated bindings scalarize known `repr(C)` by-value struct arguments to
      match AMDGPU kernel ABI lowering.
- [x] GPU-smoked compiler parity kernel for enums, custom discriminants,
      `Option`, `Result`, match lowering, arrays, loops, and scalar casts.
- [x] Device-side `DisjointSliceMut`, thread-index witness, and workgroup
      barrier token helpers for safer per-thread writes and block sync.
- [x] Scoped `i32`, `u64`, and `i64` atomics plus wavefront shuffle, match,
      vote, and additional reduction helpers.
- [x] Generated-kernel performance probe for vector, ABI, 3D stress, and
      raytrace kernels.
- [x] Per-kernel resource inspection for VGPR, SGPR, LDS, private segment,
      kernarg, spills, wavefront, and dynamic stack.
- [x] JSON benchmark snapshots with timings and resource counters.

## High-Priority Remaining Work

- [ ] ASAP feature parity sequence is tracked in
      [implementation-tasks.md](/home/kjwtil/Documents/ROCm-Oxide/docs/implementation-tasks.md).
- [ ] Compiler memory model parity:
  - [ ] HMM-style host-visible reference captures where ROCm memory properties
        and host-native atomics make the access pattern valid
  - [ ] default `repr(Rust)` struct layout matching from rustc layout facts
  - [ ] dynamic field offset and padding metadata for generated bindings
- [ ] Compiler type-system parity:
  - [x] enum, `Option`, `Result`, and custom discriminant GPU smoke coverage
  - [x] struct literals, field access, and pass-by-value `repr(C)` binding tests
  - [ ] return-by-value tests
  - [x] array construction, constant indexing, runtime indexing, and mutable
        array lowering
  - [ ] SIMD/vector register helper type once a real AMDGPU use case is chosen
  - [x] slice scalarization at kernel boundaries for `DeviceSlice<T>` and
        `DeviceSliceMut<T>`
- [ ] Closure parity:
  - [ ] move closures captured by value
  - [ ] reference closures gated by safe ROCm host-visible memory semantics
  - [ ] host-to-device closure arguments
  - [ ] device-internal closures passed to device functions
- [ ] Control-flow, arithmetic, and casting parity:
  - [x] integer and enum `match` lowering smoke coverage
  - [x] `while`, `loop`, `break`, and `continue` smoke coverage
  - [ ] nested loops, range loops, and iterator-like slice loops
  - [ ] 64-bit integer arithmetic audit across generated kernels
  - [ ] integer, float, pointer, and bitcast conversion test matrix
- [ ] Interop and compilation pipeline parity:
  - [x] cross-crate kernel discovery and bundling
  - [x] cargo subcommand and pipeline inspection equivalents
  - [x] AMDGPU LLVM IR to `.hsaco` code-object path
  - [ ] COMGR compile/link backend for persistent code-object caching
  - [ ] ROCm replacement for CUDA LTOIR/nvJitLink interop using AMD LLVM IR,
        HIP modules, and ROCm libraries
  - [ ] debug-info and debugger workflow equivalent for ROCgdb or ROCm-native
        tooling
- [x] Runtime safety parity:
  - [x] `DisjointSlice`-style output wrapper
  - [x] thread-index witness type for safe per-thread writes
  - [x] managed barrier typestate API for block/LDS synchronization
- [x] Cargo subcommand equivalent to `cargo oxide`:
  - [x] `cargo rocm-oxide build`
  - [x] `cargo rocm-oxide run`
  - [x] `cargo rocm-oxide doctor`
  - [x] `cargo rocm-oxide inspect`
  - [x] `cargo rocm-oxide pipeline`
  - [x] `cargo rocm-oxide profile`
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
  - [x] GPU-side memset and device-to-device buffer copies
  - [x] negative launch tests for generated buffer/block contracts
  - [x] pinned-buffer synchronous copy helpers
- [x] Generated launch hot paths:
  - [x] cache HIP module function handles as generated `Kernel` fields
  - [x] checked direct `*_on_stream` generated launch methods
  - [x] unsafe unchecked generated launches for prevalidated tight loops
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
  - [x] cooperative group handles
  - [x] warp/wavefront helpers
  - [x] barriers
  - [x] dynamic shared memory/LDS wrappers
  - [x] typed device slices for pointer/length kernel ABI
  - [x] basic `u32` atomics
  - [x] explicit memory-scope atomics
  - [x] math intrinsic lowering
  - [x] broader typed atomics for signed integer and 64-bit integer operations by
        memory scope
  - [ ] supported float atomics by memory scope
  - [x] wavefront shuffle up/down/xor and typed `i32`/`f32` variants
  - [x] match helpers and broader vote operations
  - [x] wavefront reductions over sum/min/max and bitwise ops
  - [ ] block reductions/scans and broader scalar-type coverage
  - [ ] debug helpers for printf/assert, clock, trap, breakpoint, and profiler
        triggers where ROCm exposes a stable path
- [ ] Compiler completeness:
  - [x] support more pointer-producing IR ops beyond `getelementptr`
  - [x] preserve source signature and contract spans in diagnostics
  - [x] catch device-codegen panics and emit actionable diagnostics
  - [x] generic helper monomorphization tests with diagnostics for exported generic kernels
  - [x] closure/captured environment ABI support through `repr(C)` environment structs
  - [x] cross-crate kernel discovery and bundling for local path dependencies
  - [x] direct exported generic-kernel monomorphization without wrapper functions

## Lower-Priority Or Vendor-Specific

- [x] CUDA feature research and implementation order is tracked in
      [cuda-future-work.md](/home/kjwtil/Documents/ROCm-Oxide/docs/cuda-future-work.md).
- [x] CUDA-specific TMA/WGMMA equivalents need ROCm-specific replacements, not
      direct ports.
- [x] cuBLASDx/cuFFTDx interop maps to rocBLAS/rocFFT or HIP library FFI and
      should be a separate integration layer.
- [ ] TMA-style async tensor copies should map to stream-ordered HIP copies,
      explicit LDS staging, and pipeline tokens only after synchronization
      semantics are validated on AMD hardware.
- [ ] WGMMA-style matrix operations should map to rocWMMA, hipBLASLt,
      Composable Kernel, rocBLAS, or tiled Rust kernels.
- [ ] DSMEM and CUDA cluster launch should map to HIP cooperative launch where
      available, otherwise graph/stream-scheduled tiling plus global-memory
      rendezvous.
- [ ] NVVM/LTOIR and nvJitLink concepts map only partially to ROCm code-object
      linking; implement after the basic artifact model is stable.
- [ ] Fine-grained LLVM `syncscope` selection for ROCm atomics should follow the
      typed scoped API once the backend has a source marker path for that IR.
