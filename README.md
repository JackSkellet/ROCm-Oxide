# ROCm-Oxide

ROCm-Oxide is a runnable AMD/ROCm path toward a CUDA Oxide-like Rust GPU stack.

It is not yet a full Rust-to-AMDGPU compiler, but it now has two working pieces:
a Rust ROCm runtime layer and a direct Rust device-code spike.

- compile a HIP kernel at runtime with HIPRTC
- load the generated AMD GPU code object with `hipModuleLoadData`
- launch the kernel from Rust through HIP runtime FFI
- compile a tiny `#![no_std]` Rust device crate to AMDGPU LLVM IR with nightly
- post-process that IR into a launchable `.hsaco`
- verify results on the CPU

The design documents describe the larger compiler path needed to make this feel
like CUDA Oxide: Rust kernel syntax, MIR/lowering, AMDGPU LLVM IR/code objects,
and a safe Rust runtime facade.

## Validated Machine Profiles

This workspace has been checked against these ROCm machines:

- Local workstation, 2026-06-01: `gfx1100`, AMD Radeon RX 7900 XT, HIP
  `7.2.53211-364a905`, AMD clang `22.0.0git`.
- Home workstation, 2026-05-31: `gfx1201`, AMD Radeon RX 9070 XT, HIP
  `7.2.53211-364a905`, AMD clang `22.0.0git`.

## Run

```bash
cargo run
```

Override the GPU architecture if needed:

```bash
ROCM_OXIDE_ARCH=gfx1100 cargo run
ROCM_OXIDE_ARCH=gfx1201 cargo run
```

The binary will otherwise try to detect the first `gfx*` target from
`/opt/rocm/bin/rocminfo`.

You can also run the reusable example:

```bash
cargo run --example vector_add
```

Run the Rust-authored device-kernel spike:

```bash
cargo run --example rust_device_add_one
cargo run --example rust_device_vector_add
cargo run --example rust_device_generated_bindings
cargo run --example feature_showcase
cargo run --example performance_probe
cargo run --example possibilities_window
cargo run --example device_operation_chain
cargo run --example module_global
cargo run --example rainbow_geometry_window
cargo run --example stress_test_gui
cargo run --example stress_3d_gui
cargo run --example spectral_lattice
cargo run --example spectral_lattice -- --frames 3
cargo run --example spectral_lattice -- --frames 3 --mode atomic
cargo run --example spectral_lattice -- --frames 1 --resolution 4k --fps-limit 120 --gpu-work 256
cargo run --example spectral_lattice -- --resolution 720p --present-scale 2 --fps-limit uncapped
cargo run --example spectral_lattice -- --present gl --resolution 1440p --fps-limit uncapped
```

`spectral_lattice` is an interactive visual workbench with clickable mode tabs
for distinct GPU paths: the core Rust-authored kernel, a dynamic-LDS tile
reduction pass, a device-scope atomic histogram overlay, and a chained
post-process pass. It also includes warp/gain/speed sliders, rocBLAS palette
reseeding, generated-binding contract checks, live kernel resource facts,
library availability, runtime FPS-limit and resolution controls up to 4K, and a
headless `--frames` path for CI/preview PNGs. The `--gpu-work` CLI flag and
matching GUI slider increase per-pixel ALU work inside the Rust-authored kernel,
while the overlay reports GPU event time separately from the host readback path.
The default live GUI presents through a CPU framebuffer, so high-resolution
interactive FPS can be limited by full-frame VRAM-to-host readback and the
windowing copy rather than by kernel throughput. The local `gfx1100` workstation
is especially sensitive to this because the RX 7900 XT path negotiates an
upstream `8GT/s x4` PCIe link. Use `--present gl` to route the final device
buffer through a HIP-registered OpenGL pixel buffer and texture instead of
reading every live frame back through the CPU. The minifb path remains the
compatibility default and the headless PNG export path. Use `--present-scale 2`
or press `M` in the live demo to keep the render buffer smaller while presenting
a larger window; for example, `--resolution 720p --present-scale 2` opens a
1440p-sized window with one quarter of the native 1440p readback traffic.

The root [build.rs](/home/kjwtil/Documents/ROCm-Oxide/build.rs)
generates device artifacts before the host crate compiles. It exposes these
compile-time environment variables to host code:

- `ROCM_OXIDE_DEVICE_HSACO`
- `ROCM_OXIDE_DEVICE_BINDINGS`
- `ROCM_OXIDE_DEVICE_METADATA`

The manual compatibility script still exists:

```bash
./scripts/compile-device-spike.sh
```

It delegates to the Rust build tool in
[tools/rocm-oxide-build](/home/kjwtil/Documents/ROCm-Oxide/tools/rocm-oxide-build/src/main.rs).
That tool handles architecture detection, nightly Rust device compilation, LLVM
IR rewriting, ROCm object generation, `.hsaco` linking, and kernel descriptor
validation.

## Current Rust API

The project now has a small library layer:

```rust
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

let device = Device::first()?;
let module = device.compile_hip_source(KERNEL)?;
let kernel = module.kernel(c"vector_add")?;

let d_a = DeviceBuffer::from_slice(&a)?;
let d_b = DeviceBuffer::from_slice(&b)?;
let d_out = DeviceBuffer::<f32>::new(a.len())?;

unsafe {
    rocm_oxide::launch!(
        kernel,
        LaunchConfig::for_num_elems(a.len()),
        d_out.as_mut_ptr(),
        d_a.as_ptr(),
        d_b.as_ptr(),
        a.len() as u64,
    )?;
}
rocm_oxide::hip::synchronize()?;
let out = d_out.copy_to_vec()?;
```

The `unsafe` launch is intentional: the host cannot verify that the raw kernel
argument list matches the compiled GPU kernel ABI.

## What This Proves

This demonstrates two important pieces:

- Rust can own the ROCm runtime surface cleanly.
- Nightly Rust can emit AMDGPU LLVM IR for a `#![no_std]` device crate when
  `core` is built for `amdgcn-amd-amdhsa`.

The `device-spike` path still uses a narrow IR post-pass because rustc emits an
ordinary device function, while ROCm/HSA needs launchable kernels to use the
`amdgpu_kernel` calling convention, global address-space pointer arguments, and a
kernel descriptor.

## Compiler Feasibility Check

This Rust toolchain reports an AMDGPU target:

```bash
rustc --print target-list | rg amdgcn
# amdgcn-amd-amdhsa
```

But direct `no_std` compilation currently fails on stable because `core` is not
available for that target:

```text
can't find crate for `core`
```

The working spike uses nightly:

```bash
rustup toolchain install nightly --component rust-src
./scripts/compile-device-spike.sh
```

The script:

1. builds `core` for `amdgcn-amd-amdhsa` with `-Z build-std=core`
2. emits LLVM IR for [device-spike/src/lib.rs](/home/kjwtil/Documents/ROCm-Oxide/device-spike/src/lib.rs)
3. discovers functions marked with `#[kernel]`
4. rewrites those Rust functions into launchable AMDGPU kernels
5. lowers it with ROCm `llc`
6. links a `.hsaco` with ROCm `clang`

The IR rewrite is no longer tied to rustc's temporary SSA names. It propagates
global address-space pointer types from kernel pointer arguments through
`getelementptr` results and validates that linked kernel descriptors exist.

The build also emits host-consumable artifacts next to the `.hsaco`:

- `rocm_oxide_device_spike.metadata.json`
- `rocm_oxide_device_spike.bindings.rs`

The compiler path now preserves source spans for kernel diagnostics, rewrites
more global-pointer-producing IR than just `getelementptr`, catches internal
rewrite panics as actionable diagnostics, discovers kernel-bearing local path
dependencies for bundling, records rustc-reported AMDGPU layout facts for
device structs, and mirrors layout-proven `repr(C)` and default `repr(Rust)`
payloads into host bindings.
The repo pins nightly Rust in `rust-toolchain.toml` so `cargo` commands use a
toolchain with `rust-src`; `rocm-oxide-build --doctor` also probes that `core`
can actually be built for `amdgcn-amd-amdhsa`.

The generated bindings expose typed host calls such as:

```rust
unsafe {
    kernels.vector_add(config, &d_out, &d_a, &d_b)?;
}
```

Bindings validate launch shape before entering HIP. They check grid/block
sanity, typed device-slice lengths, obvious mutable-buffer aliasing, legacy
`n`-sized buffer kernels, and explicit source-level buffer contracts such as:

```rust
// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(motion_reactive)=pixel_count/4*3
#[kernel]
pub unsafe extern "C" fn temporal_reconstruct_upscale(/* ... */) {}
```

Those contracts are also written into the generated metadata JSON. More detail
is in [docs/kernel-contracts.md](/home/kjwtil/Documents/ROCm-Oxide/docs/kernel-contracts.md).

The runtime also has the first host-side pieces needed for cuda-oxide-style
execution ergonomics on ROCm:

- `PinnedHostBuffer<T>` for HIP pinned host memory
- `Stream` for explicit HIP stream ownership
- `Event` for GPU-side elapsed-time measurement
- stream-aware async host/device copies
- device-to-device `DeviceBuffer` copies plus sync/async GPU-side memset and
  `set_zero` helpers for avoiding host staging on hot reset/copy paths
- stream-aware raw kernel launch
- synchronous pinned-buffer copies
- explicit fine-grained device allocation through `DeviceBuffer::new_fine_grained`
- mapped coherent pinned host buffers for host-visible GPU access
- `ManagedBuffer<T>` for HIP managed memory with fine/coarse-grain host
  visibility modeling
- HIP stream-ordered `DeviceBuffer::new_async` and explicit `free_async`
- `MemPool` controls for HIP default/current memory pools, release thresholds,
  reuse toggles, stats, trimming, and `DeviceBuffer::new_from_pool_async`
- owned HIP memory pools with access-policy controls, plus
  `DeviceVirtualMemory` for device-local HIP VMM reserve/map/access lifetimes
- rocPRIM/hipCUB-backed `RocPrim` wrappers for `u32`/`i32`/`f32` sum reduction
  and prefix scans, plus `u32` radix sort, flagged select, and transform-add
  over `DeviceBuffer`
- matrix integration candidate reporting for hipBLASLt, Composable Kernel, and
  rocWMMA, plus hipBLASLt handle/version loading
- HIPRTC runtime compilation through a process-wide specialization cache, plus
  an explicit COMGR HIP source compile/link backend with a persistent
  code-object cache keyed by backend, architecture, source, options, and launch
  metadata
- `Device::properties`, `Device::all`, and peer-access probes for
  multi-device/host-memory launch validation
- fallible allocation-size and copy-length validation instead of panics
- lazy `DeviceOperation` values with `.sync`, `.sync_on`, `.async_on`,
  `.async_in`, `.capture_graph`, `.capture_graph_on`, `.map`, `.and_then`, and
  `.zip`
- HIP stream-capture graph wrappers with `CapturedGraph::launch_on` and
  `launch_and_sync_on` replay
- explicit HIP graph builder wrappers for empty/dependency/memcpy/memset/kernel
  nodes, graph memory allocation/free nodes, node retargeting,
  instantiate/replay, and exec update
- `StreamPool` round-robin scheduling for operation pipelines
- `DeviceFuture::wait` plus `Future` support; dropping the future does not
  cancel already submitted/started work
- `Module::global::<T>` and typed `Global<T>` setters/getters over
  `hipModuleGetGlobal`
- `Kernel::occupancy_max_potential_block_size`,
  `Kernel::occupancy_max_active_blocks_per_multiprocessor`, and
  `Kernel::occupancy_for_config` wrappers over HIP occupancy planning APIs
- `Kernel::recommend_1d_launch` and generated
  `DeviceKernels::recommend_1d_launch` helpers that turn occupancy plus
  generated resource metadata into a concrete 1D launch shape
- generated `DeviceKernels` cache `Kernel` handles at load time, expose checked
  direct `*_on_stream` launches, and expose unsafe `*_unchecked` /
  `*_on_stream_unchecked` hot-loop launches for callers that prevalidate config,
  buffer lengths, aliasing, and stream/device association
- `#[device_global]`, `#[constant]`, and `#[shared]` markers for Rust-authored
  device globals, with generated typed host accessors where host-visible and
  ROCm address-space lowering
- generated-kernel performance probes without GUI/readback timing noise

`cargo rocm-oxide inspect` prints per-kernel code-object resources such as VGPR,
SGPR, static LDS, dynamic LDS usage, private segment bytes, kernarg size, spills,
wavefront size, and dynamic-stack usage.
Generated bindings expose the same facts through `DEVICE_KERNEL_RESOURCES`,
`DeviceKernels::resources()`, `DeviceKernels::resource(name)`, and a
`DeviceKernels::module()` accessor for lower-level runtime queries such as HIP
occupancy planning. Generated bindings also expose cached `Kernel` handles via
`DeviceKernels::kernel(name)`, `recommend_1d_launch` for occupancy-guided 1D
launch suggestions, checked direct stream launch methods, and unsafe unchecked
launch methods for already-validated hot paths. Generated bindings also expose
unsafe `*_graph_node` helpers for inserting validated kernel launches into
explicit HIP graphs.

`cargo run --example performance_probe -- --json target/performance_probe.json`
reports HIP-event GPU time for generated Rust kernels and can write benchmark
snapshots with the same per-kernel resource counters. Rows now include HIP
occupancy-derived active blocks/waves per HIP multiprocessor plus flags for
spills, private memory, LDS use, low occupancy, and high register pressure. The
`stress_pattern` rows are exact-loop synthetic load, while `stress_3d` and
raytrace rows are scene-dependent and may saturate when rays hit early.

`rocm-oxide-build` now has two inspection commands:

```bash
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- --doctor
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- \
  --inspect-metadata device-spike/target/amdgcn-amd-amdhsa/release/rocm_oxide_device_spike.metadata.json
```

The parity checklist against official `NVlabs/cuda-oxide` is tracked in
[docs/cuda-oxide-parity-checklist.md](/home/kjwtil/Documents/ROCm-Oxide/docs/cuda-oxide-parity-checklist.md).
Book-derived AMD adaptations are tracked in
[docs/cuda-oxide-book-rocm-adaptation.md](/home/kjwtil/Documents/ROCm-Oxide/docs/cuda-oxide-book-rocm-adaptation.md).
CUDA feature research and future ROCm-Oxide implementation order are tracked in
[docs/cuda-future-work.md](/home/kjwtil/Documents/ROCm-Oxide/docs/cuda-future-work.md).
The ASAP feature-parity sprint against NVIDIA's published cuda-oxide supported
feature matrix is tracked in
[docs/cuda-oxide-parity-checklist.md](/home/kjwtil/Documents/ROCm-Oxide/docs/cuda-oxide-parity-checklist.md)
and
[docs/implementation-tasks.md](/home/kjwtil/Documents/ROCm-Oxide/docs/implementation-tasks.md).

There is also a cargo subcommand wrapper in
[tools/cargo-rocm-oxide](/home/kjwtil/Documents/ROCm-Oxide/tools/cargo-rocm-oxide/src/main.rs):

```bash
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide build
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide inspect
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide run --example rust_device_generated_bindings
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide pipeline
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide profile
```

When installed as `cargo-rocm-oxide`, those become `cargo rocm-oxide ...`.
`profile` prefers ROCm Compute Profiler (`rocprof-compute profile`) and falls
back to `rocprofv3 --pmc Wavefronts` when only ROCprofiler-SDK is available.
Use `--pmc COUNTER[,COUNTER...]` to override the fallback counters, and use
`--trace` for `rocprofv3 --sys-trace --stats`; set `ROCM_OXIDE_PROFILER` when
the profiler binary is outside `PATH`, `/opt/rocm/bin`, or the locally extracted
`target/rocm-packages/root/opt/rocm/bin`.

## Device Kernel Shape

Device kernels now use an explicit marker from
[crates/rocm-oxide-kernel](/home/kjwtil/Documents/ROCm-Oxide/crates/rocm-oxide-kernel/src/lib.rs):

```rust
use rocm_oxide_device as gpu;
use rocm_oxide_kernel::kernel;

#[kernel]
pub unsafe extern "C" fn vector_add(
    out: gpu::DeviceSliceMut<f32>,
    a: gpu::DeviceSlice<f32>,
    b: gpu::DeviceSlice<f32>,
) {
    let i = gpu::global_id_x();
    if i < out.len() {
        let lhs = unsafe { a.read_unchecked(i) };
        let rhs = unsafe { b.read_unchecked(i) };
        unsafe { out.write_unchecked(i, lhs + rhs) };
    }
}
```

The macro exports a stable symbol name. The build tool uses the marker as the
kernel allowlist, so helper functions can remain ordinary device functions.
Generic kernels can be exported without a handwritten monomorphic wrapper by
listing concrete instantiations on the marker:

```rust
#[kernel(monomorphize(u32))]
pub unsafe extern "C" fn generic_copy<T: Copy>(
    out: gpu::DeviceSliceMut<T>,
    input: gpu::DeviceSlice<T>,
    n: usize,
) {
    let i = gpu::global_id_x();
    if i < n {
        let value = unsafe { input.read_unchecked(i) };
        unsafe { out.write_unchecked(i, value) };
    }
}
```

The macro emits the concrete exported entry point, and `rocm-oxide-build`
generates the typed host binding for that monomorphized kernel.

[crates/rocm-oxide-device](/home/kjwtil/Documents/ROCm-Oxide/crates/rocm-oxide-device/src/lib.rs)
now wraps the raw AMDGPU intrinsics used by kernels. It provides thread/block
IDs, dispatch-packet-derived block/grid dimensions, global IDs, wavefront
metadata, barriers, dynamic launch-sized LDS pointers, workgroup
synchronization for static `#[shared]` LDS kernels, ballot/reduction helpers,
typed device slices, math helpers for `sqrt`, `rsqrt`, `sin`, `cos`, `atan`,
min/max, scoped `u32`/`i32`/`u64`/`i64` atomics for
workgroup/device/system intent, wavefront shuffle/match/vote helpers,
`DisjointSliceMut`, a thread-index witness, and a managed workgroup barrier
token so device code does not need to call `core::arch::amdgpu` directly.
Atomic memory visibility rules are documented in
[docs/atomic-scopes.md](/home/kjwtil/Documents/ROCm-Oxide/docs/atomic-scopes.md).
Host-memory coherence rules are documented in
[docs/host-memory-coherence.md](/home/kjwtil/Documents/ROCm-Oxide/docs/host-memory-coherence.md).
Code-object linking rules are documented in
[docs/code-object-linking.md](/home/kjwtil/Documents/ROCm-Oxide/docs/code-object-linking.md).
Toolchain discovery and doctor-report rules are documented in
[docs/toolchain-discovery.md](/home/kjwtil/Documents/ROCm-Oxide/docs/toolchain-discovery.md).
Stream-ordered allocation rules are documented in
[docs/stream-ordered-allocation.md](/home/kjwtil/Documents/ROCm-Oxide/docs/stream-ordered-allocation.md).

## Roadmap

The next priority is practical feature parity with NVIDIA cuda-oxide's
published supported-feature matrix:
[Supported Features](https://nvlabs.github.io/cuda-oxide/appendix/supported-features.html).
ROCm-Oxide should close the user-visible Rust GPU programming gaps quickly, but
it must do that as a ROCm-native stack. That means no promise of CUDA binary
compatibility, PTX compatibility, NVVM ABI compatibility, TMA, WGMMA, or DSMEM
as NVIDIA hardware concepts. The target is source-level ergonomics and equivalent
capability where AMD hardware and ROCm libraries provide a real path.

This roadmap is grounded in the validated probe targets:

- `gfx1201`, AMD Radeon RX 9070 XT: one device, managed memory, concurrent
  managed access, host-native atomics, host mapped memory, host registration,
  and memory pools were reported available. Current generated artifact after
  the control-flow/cast sprint slice on that probe: 25 kernels, 43 buffer
  contracts, one linked object input, max VGPR 33, max SGPR 54, max kernarg 368
  bytes, max static LDS 32768 bytes, max private segment 260 bytes, two
  dynamic-LDS kernels, and no dynamic stack users.
- `gfx1100`, AMD Radeon RX 7900 XT: one device, managed memory, concurrent
  managed access, host mapped memory, host registration, and memory pools are
  reported available; direct host access to device-resident managed memory,
  pageable-memory access, registered host-pointer reuse, and host-native PCIe
  atomics are not reported on this topology. The upstream path for the RX 7900
  XT negotiates `8GT/s x4`, so full-frame CPU readback/present workloads such as
  the current `spectral_lattice` GUI can become PCIe/display-copy bound at high
  resolutions. Current generated artifact on this probe: 22 kernels, 34 buffer
  contracts, one linked object input, max VGPR 34, max SGPR 34, max kernarg 368
  bytes, max static LDS 1024 bytes, max private segment 260 bytes, two
  dynamic-LDS kernels, and no dynamic stack users.
- Both probes used HIP/runtime `7.2.53211-364a905`, AMD LLVM/clang
  `22.0.0git`, wavefront size 32, max workgroup size 1024, max waves per CU
  32, and 64 KB group/LDS segment.
- Current scoped atomic IR reaches global-memory `atomicrmw` with explicit
  `syncscope("workgroup")` or `syncscope("agent")` where requested. System scope
  uses the AMDGPU backend default because this LLVM build rejects explicit
  non-inclusive `syncscope("system")`. The `gfx1201` probe printed expected
  `scope:SCOPE_*` labels; this `gfx1100` probe prints the expected global
  atomic instructions but omits scope labels. The build validates transformed
  IR plus atomic ISA and checks printed scope labels when the disassembler
  provides them.

### P0: Backend Correctness

- ASAP cuda-oxide parity sprint: make the current prototype comparable category
  by category against NVIDIA's matrix. The first compiler/type, runtime-safety,
  and device-API breadth slice is live: `compiler_parity_matrix` and
  `compiler_flow_cast_probe` now smoke enums, `Option`, `Result`, custom
  discriminants, match, arrays, nested/range loops, iterator-like slice walks,
  pointer casts, by-value `repr(C)` and default `repr(Rust)` struct ABI
  scalarization,
  `DisjointSliceMut`, thread-index witnesses, barrier tokens, typed integer
  atomics, wavefront shuffle/match/vote helpers, and wavefront reductions
  through generated bindings. Unsupported pointer/integer casts now fail with a
  source-linked build diagnostic instead of leaking into backend lowering.
- Scope-specific atomic verification: implemented workgroup/device
  `syncscope` lowering, keep the system-scope backend default documented, verify
  the transformed IR plus disassembled ISA, and keep runtime coverage across
  default/coarse device and fine-grained device memory. Host-visible mapped and
  managed atomic smoke tests require host-native PCIe atomics; they run on the
  `gfx1201` profile that reports that capability and skip on this `gfx1100`
  machine.
- LDS/shared-memory follow-up: static `#[shared]` lowering now emits
  addrspace(3) LDS storage, verifies dynamic and static LDS IR plus DS
  load/store ISA, and feeds static/dynamic LDS pressure into launch validation
  and HIP occupancy checks.
- Occupancy and resource model: generated resources now feed HIP occupancy
  wrappers, benchmark limiter flags, and generated 1D launch-shape
  recommendations.

### P1: Runtime Orchestration

- HIP graph orchestration: `DeviceOperation` pipelines still support stream
  capture, the runtime exposes explicit graph node builders and update wrappers,
  and generated bindings can insert validated kernel nodes into explicit graphs.
- Stream-ordered allocation maturity: memory-pool controls wrap
  `hipMallocAsync`/`hipFreeAsync`, owned memory pools and VMM-backed device
  virtual memory are available, generated operations retain queued buffer
  lifetimes, and async buffer ordering rules are documented.
- Multi-device and host-memory coherence: device properties, peer probes,
  mapped pinned memory, and managed fine/coarse visibility are modeled and
  verified through runtime checks.

### P1: Compiler Completeness

- Feature-parity compiler matrix: add targeted kernels and compiler tests for
  enums and pattern matching, array construction/indexing, integer/float/pointer
  casts, loops and iterator desugaring, struct construction/return/pass-by-value,
  and default `repr(Rust)` host/device layout matching.
- Closure coverage: `compiler_move_closure_probe_RustLayoutParams` now validates
  a generic device kernel that builds a `move` closure from a by-value captured
  environment and passes it through a device helper.
  `compiler_host_closure_arg_probe_HostAffineClosure` validates a host-provided
  closure environment implementing `FnOnce` through ROCm metadata-driven
  global-buffer kernargs. `compiler_host_reference_closure_probe_HostReferenceClosure`
  adds a pointer-bearing closure environment for mapped pinned or managed memory,
  so reference captures are only exercised when ROCm reports a host-visible
  memory path.
- ABI and layout parity: generated metadata now records rustc AMDGPU struct
  layout facts, including field offsets, padding, ABI size, and alignment;
  generated host bindings assert matching host layout and reject unsupported
  by-value payloads before launch.
- Direct exported generic-kernel monomorphization without wrapper functions:
  `#[kernel(monomorphize(...))]` now emits concrete entry points and generated
  typed host bindings.
- ROCm code-object artifact linking: multiple generated objects link into one
  HSACO, metadata records each link input, every linked kernel resource row is
  required before host bindings emit, and HIP module/library loading APIs are
  documented for the runtime inspection path.
- Toolchain discovery hardening: doctor now uses explicit tool overrides,
  `ROCM_PATH`/`HIP_PATH`, `/opt/rocm`, and `PATH`; validates `llc`, `clang`,
  `llvm-readelf`, `llvm-objdump`, `rocminfo`, `rocm_agent_enumerator`, target
  architecture, `rust-src`, and Rust `build-std` readiness in one report.

### P2: ROCm-Specific Feature Parity

- Runtime safety layer: `DisjointSliceMut`, thread-index witness, and managed
  barrier token helpers now exist in the device crate and are exercised by the
  generated-binding smoke path. Next is tightening them into compile-time or
  generated-binding checks where possible, and runtime validation where the
  information only exists at launch.
- Device API breadth: the first broader typed atomic matrix, scoped float
  atomic add/load/store wrappers, wavefront shuffle/vote/match helpers,
  wavefront reductions, scratch-backed block add/min/max/bitwise reductions,
  block add/min/max/bitwise scans, 64-bit block collective coverage, and
  smoke-safe debug helpers for dispatch id, program counter, sleep, assert/trap,
  and breakpoint entry points are live. rocTX host profiler markers/ranges and
  HIP clock-rate metadata are live; GPU printf and selectable device clock
  counters remain documented ROCm/Rust-backend gaps until a stable path exists.
- COMGR/code-object backend: `Device::compile_hip_source_comgr` now uses COMGR
  to compile HIP source to a relocatable, link it into an executable code
  object, and cache it persistently. Next is extending the same backend shape to
  ROCm library/device-object interop where HIPRTC is too narrow.
- Library parity: rocPRIM/hipCUB now covers `u32`/`i32`/`f32` sum
  reduce/scans, `u32` radix sort, flagged select, and transform-add over
  `DeviceBuffer`. Promote hipBLASLt or Composable Kernel from availability
  checks into first checked matmul descriptors and heuristics.
- CUDA-only advanced hardware mapping: keep TMA, WGMMA, DSMEM clusters, and
  nvJitLink/LTOIR as source-level rewrite targets. Use stream-ordered copies
  plus LDS staging for TMA-like flows, rocWMMA/hipBLASLt/Composable Kernel for
  matrix/tensor paths, HIP cooperative launch or graph-scheduled tiling for
  cluster-style work, and AMDGPU IR plus COMGR/clang/HSACO/HIP modules for
  CUDA artifact-link flows.
- ROCm-specific replacements for CUDA cluster launch, TMA, and WGMMA concepts:
  cooperative module-launch wrappers, device capability flags, and a parity
  planner now map these CUDA-only concepts to HIP cooperative grids,
  stream/LDS staged transfers, and rocWMMA/rocBLAS/tiled-kernel matrix paths.
- rocBLAS/rocFFT/library interop: optional dynamic loading keeps the core
  runtime buildable without every ROCm library installed, while rocBLAS SGEMM
  and first rocFFT in-place complex-plan wrappers operate on `DeviceBuffer`
  values.
- ROCm profiler integration: `cargo rocm-oxide profile` builds the default
  performance probe, runs it under `rocprof-compute profile` when installed,
  and falls back to `rocprofv3 --pmc Wavefronts` from `PATH`, `/opt/rocm/bin`,
  or `target/rocm-packages/root/opt/rocm/bin`. `--trace` uses
  `rocprofv3 --sys-trace --stats` for HIP/HSA dispatch, memory, and runtime
  traces. The runtime also exposes `RocTx` host markers, nested scoped ranges,
  and process ranges for profiler timelines.

Roadmap source docs:
[HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api_reference.html),
[HIP launch API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/launch_api.html),
[HIP module management](https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___module.html),
[HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/how-to/hip_runtime_api/hipgraph.html),
[stream ordered memory allocator](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html),
[HIP coherence control](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html),
[HIP unified memory](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/how-to/unified_memory.html),
[HIP peer-to-peer memory access](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/doxygen/html/group___peer_to_peer.html),
[HIP atomics](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_cpp_language_extensions.html#atomic-functions),
[ROCm hardware atomics](https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html),
[AMDGPU LLVM backend](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html),
[rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/),
[rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/latest/),
[ROCm rocTX](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-roctx.html),
[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/use.html),
and
[rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html).

## Verification

Current verification commands:

```bash
./scripts/verify.sh
```
