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

## Local Machine

This workspace was checked against:

- HIP: `7.2.53211-364a905`
- AMD clang: `22.0.0git`
- GPU target: `gfx1201`
- GPU: `AMD Radeon RX 9070 XT`

## Run

```bash
cargo run
```

Override the GPU architecture if needed:

```bash
ROCM_OXIDE_ARCH=gfx1100 cargo run
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
```

The root [build.rs](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/build.rs)
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
[tools/rocm-oxide-build](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/tools/rocm-oxide-build/src/main.rs).
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
2. emits LLVM IR for [device-spike/src/lib.rs](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/device-spike/src/lib.rs)
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
dependencies for bundling, and mirrors `#[repr(C)]` device structs into host
bindings for captured-environment style ABI payloads.
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
is in [docs/kernel-contracts.md](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/docs/kernel-contracts.md).

The runtime also has the first host-side pieces needed for cuda-oxide-style
execution ergonomics on ROCm:

- `PinnedHostBuffer<T>` for HIP pinned host memory
- `Stream` for explicit HIP stream ownership
- `Event` for GPU-side elapsed-time measurement
- stream-aware async host/device copies
- stream-aware raw kernel launch
- synchronous pinned-buffer copies
- HIP stream-ordered `DeviceBuffer::new_async` and explicit `free_async`
- fallible allocation-size and copy-length validation instead of panics
- lazy `DeviceOperation` values with `.sync`, `.sync_on`, `.async_on`,
  `.async_in`, `.map`, `.and_then`, and `.zip`
- `StreamPool` round-robin scheduling for operation pipelines
- `DeviceFuture::wait` plus `Future` support; dropping the future does not
  cancel already submitted/started work
- `Module::global::<T>` and typed `Global<T>` setters/getters over
  `hipModuleGetGlobal`
- `#[device_global]` and `#[constant]` markers for Rust-authored device globals,
  with generated typed host accessors and ROCm address-space lowering
- generated-kernel performance probes without GUI/readback timing noise

`cargo rocm-oxide inspect` prints per-kernel code-object resources such as VGPR,
SGPR, static LDS, dynamic LDS usage, private segment bytes, kernarg size, spills,
wavefront size, and dynamic-stack usage.

`cargo run --example performance_probe -- --json target/performance_probe.json`
reports HIP-event GPU time for generated Rust kernels and can write benchmark
snapshots with the same per-kernel resource counters. The `stress_pattern` rows
are exact-loop synthetic load, while `stress_3d` and raytrace rows are
scene-dependent and may saturate when rays hit early.

`rocm-oxide-build` now has two inspection commands:

```bash
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- --doctor
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- \
  --inspect-metadata device-spike/target/amdgcn-amd-amdhsa/release/rocm_oxide_device_spike.metadata.json
```

The parity checklist against official `NVlabs/cuda-oxide` is tracked in
[docs/cuda-oxide-parity-checklist.md](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/docs/cuda-oxide-parity-checklist.md).
Book-derived AMD adaptations are tracked in
[docs/cuda-oxide-book-rocm-adaptation.md](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/docs/cuda-oxide-book-rocm-adaptation.md).

There is also a cargo subcommand wrapper in
[tools/cargo-rocm-oxide](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/tools/cargo-rocm-oxide/src/main.rs):

```bash
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide build
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide inspect
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide run --example rust_device_generated_bindings
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide pipeline
```

When installed as `cargo-rocm-oxide`, those become `cargo rocm-oxide ...`.

## Device Kernel Shape

Device kernels now use an explicit marker from
[crates/rocm-oxide-kernel](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/crates/rocm-oxide-kernel/src/lib.rs):

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

[crates/rocm-oxide-device](/home/jack/Documents/GitKraken_Projects/ROCm-Oxide/crates/rocm-oxide-device/src/lib.rs)
now wraps the raw AMDGPU intrinsics used by kernels. It provides thread/block
IDs, dispatch-packet-derived block/grid dimensions, global IDs, wavefront
metadata, barriers, dynamic launch-sized LDS pointers, ballot/reduction helpers,
typed device slices, math helpers for `sqrt`, `rsqrt`, `sin`, `cos`, `atan`,
min/max, scoped `u32` atomics for workgroup/device/system intent, and the basic
relaxed `u32` atomic compatibility helpers so device code does not need to call
`core::arch::amdgpu` directly.

## Roadmap

This roadmap is grounded in the current local probe target:

- GPU: `gfx1201`, AMD Radeon RX 9070 XT.
- HIP/runtime: `7.2.53211-364a905`.
- AMD LLVM/clang: `22.0.0git`.
- Device limits seen through `rocminfo`: wavefront size 32, max workgroup size
  1024, max waves per CU 32, and 64 KB group/LDS segment.
- Current generated artifact: 14 kernels, 18 buffer contracts, max VGPR 33, max
  SGPR 26, max kernarg 368 bytes, static LDS 0, one dynamic-LDS kernel, and no
  dynamic stack users.
- Current scoped atomic IR reaches global-memory `atomicrmw` with explicit
  `syncscope("workgroup")` or `syncscope("agent")` where requested. System scope
  uses the AMDGPU backend default because this LLVM build rejects explicit
  non-inclusive `syncscope("system")`.

### P0: Backend Correctness

- Scope-specific atomic verification: preserve the new workgroup/device
  `syncscope` lowering, document the system-scope default, and verify resulting
  IR/ISA against coarse-grained, fine-grained, and host-visible memory.
- LDS/shared-memory follow-up: extend the new dynamic-LDS reduction and
  per-kernel validation path to static LDS cases, ISA checks, and occupancy
  planning.
- Occupancy and resource model: expose per-kernel resource metadata at runtime,
  wrap HIP occupancy APIs, and flag VGPR/SGPR/LDS/private-memory limiters in
  benchmark output.

### P1: Runtime Orchestration

- HIP graph capture for `DeviceOperation` pipelines: keep generated operations
  stream-only, add graph instantiate/launch wrappers, and verify replay for
  generated kernel bindings.
- Stream-ordered allocation maturity: add memory-pool controls around
  `hipMallocAsync`/`hipFreeAsync`, preserve queued-operation lifetimes, and
  document async buffer ordering rules.
- Multi-device and host-memory coherence: model coarse/fine-grained memory pools,
  pinned/managed/peer memory, and device properties needed by launch validation.

### P1: Compiler Completeness

- Direct exported generic-kernel monomorphization without wrapper functions.
- ROCm code-object artifact linking: link multiple generated objects, preserve
  metadata, and investigate HIP library enumeration/loading APIs.
- Toolchain discovery hardening: validate `ROCM_PATH`/`HIP_PATH`, `/opt/rocm`,
  `llc`, `clang`, `llvm-readelf`, `rocminfo`, `rocm_agent_enumerator`, target
  architecture, and Rust `build-std` readiness in one doctor report.

### P2: ROCm-Specific Feature Parity

- ROCm-specific replacements for CUDA cluster launch, TMA, and WGMMA concepts.
- rocBLAS/rocFFT/library interop after the code-object model is stable.
- ROCm Compute Profiler integration for achieved occupancy and memory behavior.

Roadmap source docs:
[HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api_reference.html),
[HIP launch API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/launch_api.html),
[HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/how-to/hip_runtime_api/hipgraph.html),
[stream ordered memory allocator](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html),
[HIP atomics](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_cpp_language_extensions.html#atomic-functions),
[ROCm hardware atomics](https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html),
[AMDGPU LLVM backend](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html),
and
[ROCm Compute Profiler occupancy examples](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.0/tutorial/profiling-by-example.html).

## Verification

Current verification commands:

```bash
./scripts/verify.sh
```
