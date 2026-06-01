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

Local probes:

- 2026-05-31 home workstation: `gfx1201`, AMD Radeon RX 9070 XT.
  HIP/runtime `7.2.53211-364a905`; AMD LLVM/clang `22.0.0git`. HIP reported
  managed memory, concurrent managed access, host-native atomics, host mapped
  memory, host registration, and memory pools. Current generated artifact on
  that probe: 21 kernels, 33 buffer contracts, one linked object input, max VGPR
  33, max SGPR 28, max kernarg 368 bytes, max static LDS 1024 bytes, max
  private segment 260 bytes, two dynamic-LDS kernels, and no dynamic stack
  users.
- 2026-06-01 local workstation: `gfx1100`, AMD Radeon RX 7900 XT. HIP/runtime
  `7.2.53211-364a905`; AMD LLVM/clang `22.0.0git`. HIP reported managed memory,
  concurrent managed access, host mapped memory, host registration, and memory
  pools; direct host access to device-resident managed memory, pageable-memory
  access, registered host-pointer reuse, and host-native PCIe atomics are not
  reported on this topology. The RX 7900 XT path negotiates an upstream
  `8GT/s x4` PCIe link, which makes full-frame CPU readback/present paths
  bandwidth-sensitive at 1440p and 4K. Current generated artifact on this probe:
  21 kernels, 33 buffer contracts, one linked object input, max VGPR 34, max
  SGPR 34, max kernarg 368 bytes, max static LDS 1024 bytes, max private
  segment 260 bytes, two dynamic-LDS kernels, and no dynamic stack users.
- Both probes report wavefront size 32, max workgroup size 1024, max waves per
  CU 32, and 64 KB group/LDS segment.
- Current scoped atomic IR emits global-memory `atomicrmw` with explicit
  `syncscope("workgroup")` or `syncscope("agent")` where requested. System scope
  intentionally uses the AMDGPU backend default because the local LLVM backend
  rejects explicit non-inclusive `syncscope("system")`. The `gfx1201`
  disassembler output printed expected `scope:SCOPE_*` labels, while this
  `gfx1100` disassembler output can omit them, so the build verifies IR scope
  mapping plus atomic ISA and treats printed scope labels as optional extra
  evidence.

## Next Roadmap

### P0: Backend Correctness

- [x] Scope-specific atomic lowering:
  - [x] preserve source-level workgroup/device/system scope markers through IR
  - [x] lower workgroup/device scope to AMDGPU LLVM `syncscope` and keep system
        scope on the backend default
  - [x] verify transformed IR and disassembled ISA for the scoped atomic kernel
  - [x] smoke-test scoped atomics on device-memory counters at runtime
  - [x] extend runtime coverage across default/coarse device and fine-grained device pools
  - [x] gate mapped/managed host-visible atomic smoke tests on host-native PCIe atomics
        so the `gfx1201` path can run them and the `gfx1100` PCIe-switch path can skip them
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
- [ ] GPU-native presentation path for `spectral_lattice`:
  - [x] add a scaled minifb presentation mode so `720p` can present as a 1440p-sized window and `1080p` can present as a 4K-sized window without native-resolution readback
  - [ ] replace the live `minifb` CPU framebuffer path with Vulkan/OpenGL texture presentation or ROCm graphics interop
  - [ ] avoid full-frame VRAM-to-host readback every frame for 1440p and 4K interactive runs
  - [ ] keep the existing CPU readback path for headless PNG export and simple compatibility smoke tests
