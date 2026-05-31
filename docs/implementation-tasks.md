# ROCm-Oxide Implementation Tasks

This list tracks the next core implementation work after removing the old
side-tool path.
The order is intentional: tighten the kernel ABI first, then build higher-level
features on top of stronger contracts.

## Active Sequence

- [x] Typed device slices:
  - [x] add `DeviceSlice<T>` and `DeviceSliceMut<T>` to device code
  - [x] mirror the ABI shape on the host side
  - [x] teach generated bindings to pass pointer/length pairs automatically
  - [x] reject obvious mutable-buffer aliasing before launch
  - [x] convert simple kernels before large demo kernels
  - [x] convert image, upscaling, stress, and raytrace kernels
- [x] Constant/global/shared memory source markers:
  - [x] add markers such as `#[device_global]`, `#[constant]`, and `#[shared]`
  - [x] lower marked globals with ROCm address-space awareness
  - [x] connect host-visible marked globals to typed host views
- [x] Math intrinsic lowering:
  - [x] map common `f32`/`f64` math calls to AMDGPU/ROCm-supported lowering
  - [x] add tests for `sqrt`, `rsqrt`, `sin`, `cos`, `atan`, min/max, and NaN behavior
- [x] Explicit memory-scope atomics:
  - [x] expose workgroup/device/system scope where ROCm supports it
  - [x] keep relaxed/basic atomics as the compatibility path
- [x] Generated lazy operations:
  - [x] allow generated kernel bindings to return `DeviceOperation` values
  - [x] support stream-pool scheduling without eager launch
  - [x] keep the immediate launch API as a convenience wrapper

## Later

## Roadmap Inputs

Local probes on 2026-05-31:

- GPU target: `gfx1201`, AMD Radeon RX 9070 XT.
- HIP/runtime: `7.2.53211-364a905`; AMD LLVM/clang: `22.0.0git`.
- Device limits from `rocminfo`: wavefront size 32, max workgroup size 1024,
  max waves per CU 32, 64 KB group/LDS segment.
- HIP host-memory properties: one device; managed memory, concurrent managed
  access, host-native atomics, host mapped memory, host registration, and memory
  pools are available; direct host access to device-resident managed memory,
  pageable-memory access, and registered host-pointer reuse are not reported on
  this dGPU.
- Current generated artifact: 17 kernels, 23 buffer contracts, one linked
  object input, max VGPR 33, max SGPR 26, max kernarg 368 bytes, max static LDS
  1024 bytes, max private segment 260 bytes, one dynamic-LDS kernel, and no
  dynamic stack users.
- Current scoped atomic IR emits global-memory `atomicrmw` with explicit
  `syncscope("workgroup")` or `syncscope("agent")` where requested. System scope
  intentionally uses the AMDGPU backend default because the local LLVM backend
  rejects explicit non-inclusive `syncscope("system")`.

## Next Roadmap

### P0: Backend Correctness

- [x] Scope-specific atomic lowering:
  - [x] preserve source-level workgroup/device/system scope markers through IR
  - [x] lower workgroup/device scope to AMDGPU LLVM `syncscope` and keep system
        scope on the backend default
  - [x] verify transformed IR and disassembled ISA for the scoped atomic kernel
  - [x] smoke-test scoped atomics on device-memory counters at runtime
  - [x] extend runtime coverage across default/coarse device, fine-grained device, and mapped host-visible pools
  - [x] add negative docs/tests for system-scope atomics that downgrade on coarse memory
- [x] LDS/shared-memory dynamic path:
  - [x] add a real tiled/reduction kernel that uses dynamic LDS
  - [x] validate requested `shared_mem_bytes` against device and kernel limits
  - [x] report static and dynamic LDS in generated metadata and host bindings
  - [x] expose ergonomic typed workgroup scratch helpers in device/host code
- [x] LDS/shared-memory static and ISA verification:
  - [x] add a static LDS kernel once address-space-safe Rust syntax is settled
  - [x] verify LDS IR and ISA for dynamic plus static cases
  - [x] feed LDS pressure into occupancy planning
- [x] Occupancy and resource model:
  - [x] expose per-kernel VGPR, SGPR, LDS, private segment, kernarg, and wavefront metadata at runtime
  - [x] switch `performance_probe` to the generated runtime resource table
  - [x] wrap HIP occupancy APIs for launch planning
  - [x] add benchmark output that flags occupancy limiters and spills
  - [x] turn resource/occupancy facts into generated launch-shape recommendations

### P1: Runtime Orchestration

- [x] HIP graph capture for `DeviceOperation` pipelines:
  - [x] keep operations stream-only and graph-capturable
  - [x] add graph instantiate/launch wrappers
  - [x] verify graph replay for generated kernel bindings
- [x] Stream-ordered allocation maturity:
  - [x] add memory-pool controls around `hipMallocAsync`/`hipFreeAsync`
  - [x] preserve allocation lifetimes across queued generated operations
  - [x] document stream-ordering requirements for async buffers
- [x] Multi-device and host-memory coherence:
  - [x] model coarse/fine-grained memory pools and host visibility
  - [x] add pinned, managed, and peer-memory contract tests
  - [x] expose device properties needed by generated launch validation

### P1: Compiler Completeness

- [x] Direct exported generic-kernel monomorphization without wrapper functions:
  - [x] accept `#[kernel(monomorphize(...))]` on generic device kernels
  - [x] emit concrete exported entry points from the proc macro
  - [x] discover monomorphized entry points and generate typed host bindings
  - [x] verify with a real `generic_copy_u32` HSACO kernel and showcase launch
- [x] ROCm code-object artifact linking layer:
  - [x] support linking multiple generated objects
  - [x] preserve link inputs in generated metadata
  - [x] require merged linked-HSACO metadata for every generated kernel
  - [x] investigate HIP module/library enumeration and loading APIs for artifact inspection
- [x] Toolchain discovery hardening:
  - [x] prefer explicit tool overrides, `ROCM_PATH`/`HIP_PATH`, `/opt/rocm`, then `PATH`
  - [x] validate `llc`, `clang`, `llvm-readelf`, `llvm-objdump`, `rocminfo`, and `rocm_agent_enumerator`
  - [x] emit one doctor report with versions, target arch, `rust-src`, and build-std status

### P2: ROCm-Specific Feature Parity

- [x] ROCm-specific replacements for CUDA cluster launch, TMA, and WGMMA concepts:
  - [x] expose HIP cooperative module launches and cooperative-launch device properties
  - [x] add an AMD-specific feature-parity planner for cluster/TMA/WGMMA ports
  - [x] document the explicit replacement model instead of pretending CUDA-only concepts are ABI-compatible
- [x] rocBLAS/rocFFT/library interop layer after the code-object model is stable:
  - [x] dynamically load rocBLAS/rocFFT so missing optional libraries do not break the core runtime
  - [x] expose a checked rocBLAS SGEMM wrapper for `DeviceBuffer<f32>`
  - [x] expose first rocFFT setup/plan/execute wrappers for in-place complex `f32` buffers
- [ ] ROCm Compute Profiler integration for achieved occupancy and memory behavior.
