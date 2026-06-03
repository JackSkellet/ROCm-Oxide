# API Overview

This document describes every public layer of the ROCm Oxide SDK, from the
proc-macro attributes you use in device code to the host-side runtime types
you use to manage memory, compile modules, and launch kernels.

For a hands-on introduction see [getting-started.md](getting-started.md).

---

## Architecture layers

```
┌────────────────────────────────────────────────────┐
│  Your host code (src/main.rs)                      │
│  Device, Module, Kernel, DeviceBuffer, launch!     │
│  (rocm-oxide / src/)                               │
├────────────────────────────────────────────────────┤
│  Generated typed bindings (bindings.rs)            │
│  DeviceKernels struct — one method per #[kernel]   │
├────────────────────────────────────────────────────┤
│  Build tool (rocm-oxide-build)                     │
│  Rust → LLVM IR → llc → clang → .hsaco            │
├────────────────────────────────────────────────────┤
│  Device crate (rocm-oxide-device)                  │
│  Thread IDs, barriers, atomics, math, vectors, LDS │
├────────────────────────────────────────────────────┤
│  Proc-macro crate (rocm-oxide-kernel)              │
│  #[kernel], #[device_global], #[constant], #[shared]│
└────────────────────────────────────────────────────┘
```

---

## Proc-macro layer — `rocm-oxide-kernel`

All four attributes live in this crate. Add it as a dependency to your **device**
crate (not the host crate).

### `#[kernel]`

Marks a `pub unsafe extern "C"` function as a GPU kernel entry point.
`rocm-oxide-build` discovers all `#[kernel]` functions in the device crate and
compiles them to the `.hsaco` code object.

```rust
#[kernel]
pub unsafe extern "C" fn vector_add(
    a: gpu::DeviceSlice<f32>,
    b: gpu::DeviceSlice<f32>,
    out: gpu::DeviceSliceMut<f32>,
    n: usize,
) { /* ... */ }
```

**Monomorphized kernels** — pass a `#[monomorphize(T = [f32, f64])]` attribute
to generate a separate HSACO entry for each type. The build tool emits
`vector_add_f32` and `vector_add_f64` and the generated `DeviceKernels` struct
exposes both.

### `#[device_global]`

Marks a `static mut` as a mutable device-global variable allocated in AMDGPU
global address space. Access it from the host via `Module::global(c"name")`.

### `#[constant]`

Marks a `static` as a read-only constant in AMDGPU constant address space.
Use for tables or parameters that never change during a kernel invocation.

### `#[shared]`

Marks a `static mut` as Local Data Share (LDS) memory — on-chip scratchpad
shared by all threads in the same workgroup. Fastest possible intra-block
communication. Must be declared at `#![no_std]` module scope in the device crate.

---

## Device library — `rocm-oxide-device`

This `#![no_std]` crate is compiled for `amdgcn-amd-amdhsa` and linked into
every kernel. Use it inside `device-spike/src/lib.rs`.

### Thread/block/grid indices

| Function | Equivalent | Returns |
|----------|-----------|---------|
| `thread_idx_x/y/z()` | `threadIdx.x/y/z` | `u32` |
| `block_idx_x/y/z()` | `blockIdx.x/y/z` | `u32` |
| `block_dim_x/y/z()` | `blockDim.x/y/z` | `u32` |
| `grid_dim_x/y/z()` | `gridDim.x/y/z` | `u32` |
| `global_id_x/y/z()` | `blockIdx.x * blockDim.x + threadIdx.x` | `usize` |

The most common pattern for a 1-D kernel:

```rust
let i = gpu::global_id_x();
if i < n { /* process element i */ }
```

### Wavefront / lane utilities

| Function | What it returns |
|----------|----------------|
| `lane_id()` | Lane index within the wavefront (`0..wavefront_size()`) |
| `wavefront_size()` | 64 on CDNA/GFX9, 32 on RDNA3+ gfx12 |
| `wave_id_in_workgroup()` | Which wavefront within the block |
| `is_first_lane()` | True only for lane 0 |
| `ballot(bool)` | 64-bit mask of which lanes passed `true` |
| `dispatch_id()` | Monotonic dispatch counter |

### Synchronization

| Function | Scope |
|----------|-------|
| `workgroup_barrier()` | All threads in the workgroup (block barrier) |
| `wave_barrier()` | All lanes in the current wavefront |

### Module `math`

Scalar GPU-native math for `f32` and `f64`: `sqrt`, `rsqrt`, `sin`, `cos`,
`atan`, `atan2`, `min`, `max`, `fmin`, `fmax`, `nan`. These map directly to
AMDGPU hardware intrinsics and are significantly faster than soft-emulated
versions.

### Module `vector`

`repr(C)` vector types safe to pass through kernel ABI boundaries:

| Type | Contents | Operations |
|------|---------|-----------|
| `Vec2f` | `f32 x, y` | `Add`, `Sub`, `Mul`, `Neg`, `Div`, `dot`, `length`, `normalize`, `lerp` |
| `Vec3f` | `f32 x, y, z` | above + `cross`, `reflect` |
| `F16x2` | two `f16` in one `u32` | hardware 16-bit arithmetic |

### Module `atomic`

Scoped GPU atomics. Three `AtomicScope` variants: `Workgroup`, `Device`, `System`.

Typed wrappers follow the naming pattern `{Scope}Atomic{Type}`:
`WorkgroupAtomicU32`, `DeviceAtomicF32`, `SystemAtomicU64`, etc.

```rust
use rocm_oxide_device::atomic::{DeviceAtomicU32, AtomicOrdering};
unsafe { DeviceAtomicU32::from_ptr(counter).fetch_add(1, AtomicOrdering::Relaxed); }
```

Free-function forms: `atomic_add_u32_scoped`, `atomic_store_f32_scoped`, etc.

See [docs/atomic-scopes.md](atomic-scopes.md) for coherence rules.

### Module `cooperative`

Group-level reductions, scans, and shuffles.

| Type | Granularity |
|------|------------|
| `ThreadBlock` | Full workgroup |
| `Wavefront` | 64-lane (or 32-lane) wavefront |
| `StaticTile<N>` | Sub-group of N consecutive lanes |

Obtain handles: `this_thread_block()`, `this_wavefront()`, `tiled_partition::<N>()`.

Operations: `reduce_add_{u32,i32,f32,u64,i64,f64}`, `reduce_min_*`, `reduce_max_*`,
`reduce_and_*`, `reduce_or_*`, `reduce_xor_*`; inclusive/exclusive scans; wave
shuffle (`shfl_up`, `shfl_down`, `shfl_xor`).

### Module `debug`

Not for production: `trap()`, `breakpoint()`, `sleep(N)`, `dispatch_id()`,
`program_counter()`, `assert_or_trap(condition)`. The `gpu_assert!` macro wraps
`assert_or_trap` with file/line context.

### Slice types

| Type | Use in kernel signature |
|------|------------------------|
| `DeviceSlice<T>` | Read-only view of a `DeviceBuffer<T>` |
| `DeviceSliceMut<T>` | Read-write view of a `DeviceBuffer<T>` |

Both are `repr(C)` fat pointers (address + length). The build tool converts
host `DeviceBuffer<T>` arguments into the correct `(ptr, len)` pairs.

---

## Build tool — `rocm-oxide-build`

`rocm-oxide-build` is invoked by `build.rs`. You do not call it directly in
normal workflows. It implements the full Rust → GPU pipeline:

| Step | Command | Output |
|------|---------|--------|
| Discover `#[kernel]` functions | AST scan | List of entry point names |
| Compile device crate | `cargo rustc -Z build-std=core --target amdgcn-amd-amdhsa` | `.bc` LLVM bitcode |
| Rewrite IR | Internal `transform_ir()` | `amdgpu_kernel` calling convention, address space fixes |
| Lower to object | ROCm `llc` | `.o` |
| Link code object | ROCm `clang` | `.hsaco` |
| Validate | `llvm-readelf` | Confirms kernel symbols are present |
| Emit metadata | — | `.metadata.json` (argument type records) |
| Emit bindings | — | `.bindings.rs` (typed `DeviceKernels` struct) |
| Emit manifest | — | `.manifest.json` (artifact paths) |

`build.rs` copies all artifacts to `OUT_DIR` and sets environment variables:

| Env var | Points to |
|---------|----------|
| `ROCM_OXIDE_DEVICE_HSACO` | `.hsaco` code object |
| `ROCM_OXIDE_DEVICE_BINDINGS` | `bindings.rs` with `DeviceKernels` |
| `ROCM_OXIDE_DEVICE_METADATA` | `metadata.json` |
| `ROCM_OXIDE_DEVICE_MANIFEST` | `manifest.json` |

---

## Generated typed bindings

`rocm-oxide-build` emits a `DeviceKernels` struct with one method per
`#[kernel]` function. The method signature matches the kernel exactly but
accepts host-side types (`&DeviceBuffer<T>` instead of `DeviceSlice<T>`,
`&mut DeviceBuffer<T>` instead of `DeviceSliceMut<T>`).

Include the bindings in `src/main.rs` (or `src/lib.rs`):

```rust
include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
```

Then load and use:

```rust
let kernels = DeviceKernels::load_embedded(&device)?;
kernels.fill_indices(LaunchConfig::for_num_elems(n), &mut out, n)?;
```

`load_embedded` reads the HSACO from the bytes embedded by `build.rs` at
compile time — no file path needed at runtime.

---

## Host runtime — `rocm-oxide` (src/)

| Module | Key types |
|--------|----------|
| `runtime` | `Device`, `Module`, `Kernel`, `LaunchConfig`, `DeviceLimits`, `DeviceProperties` |
| `hip` | `DeviceBuffer<T>`, `ManagedBuffer<T>`, `PinnedHostBuffer`, `Stream`, `Event`, `Graph` |
| `operation` | `ExecutionContext`, `StreamPool`, `DeviceOperation` (trait), `CapturedGraph` |
| `hiprtc` | `SpecializationCache`, HIPRTC/COMGR compilation (internal) |
| `profiling` | ROC-Tracer integration (internal) |

### `Device`

Entry point. Call `Device::first()` for single-GPU or `Device::all()` for
multi-GPU. From a device you: load modules, compile HIP C++ at runtime, create
streams, and query device properties.

### `Module`

A loaded `.hsaco` code object. Produced by `Device::load_code_object_file`,
`Device::compile_hip_source`, or `Device::compile_hip_source_comgr`. Yields
`Kernel` handles via `Module::kernel(c"name")`.

### `Kernel`

A function pointer from a loaded `Module`. `Send + Sync`. Launch with
`launch!(kernel, config, arg0, ..., argN)?`.

### `LaunchConfig`

Grid/block/shared-memory specification. Use the convenience constructors:

| Constructor | Use case |
|------------|---------|
| `for_num_elems(n)` | 1-D: n elements, 256 threads/block |
| `for_num_elems_with_block_size(n, block_x)` | 1-D: custom block size |
| `for_2d(w, h, bx, by)` | 2-D: images, matrices |
| `new(grid, block)` | Full manual control |

### `DeviceBuffer<T>`

GPU-side allocation. Allocate with `DeviceBuffer::new(n)` (uninitialized) or
`DeviceBuffer::from_slice(&[...])` (copies from host). Copy results back to the
host with `copy_to_vec()`. Requires `T: Copy` for `from_slice` and
`T: Copy + Default` for `copy_to_vec`.

### `ManagedBuffer<T>`

HIP managed (unified) memory accessible from both host and device. Useful for
incremental migration or probing, but carries coherence overhead. Prefer
`DeviceBuffer` for performance-critical paths.

### `Stream`

An ordered queue of GPU operations. Kernels and memory copies enqueued on the
same stream execute in submission order. Operations on different streams may
overlap. Synchronize with `Stream::synchronize()`.

### `ExecutionContext` and `StreamPool`

Higher-level wrappers that manage a pool of streams and a default ordering
context. Useful for pipelines that chain multiple kernel launches with
dependency tracking. See `src/operation.rs` for the `DeviceOperation` trait.

### `launch!` macro

Builds the HIP argument pointer array and calls `Kernel::launch_raw`. Supports
0–16 arguments. Returns `Result<()>`. Always append `?`.

---

## Examples

The `examples/` directory contains end-to-end programs:

| Example | Demonstrates |
|---------|-------------|
| `hello_gpu.rs` | **Start here**: minimal HIPRTC vector add, end-to-end lifecycle |
| `vector_add.rs` | Simple 1-D HIP kernel (C++ via HIPRTC) |
| `rust_device_vector_add.rs` | Same kernel written in Rust |
| `rust_device_add_one.rs` | Minimal Rust kernel, no slice types |
| `module_global.rs` | `#[device_global]` and `Module::global` |
| `rust_device_generated_bindings.rs` | Using `DeviceKernels` generated struct |
| `pinned_stream_vector_add.rs` | Pinned host memory + streams |
| `performance_probe.rs` | Occupancy, bandwidth, latency profiling |
| `validation_profile.rs` | Verifying kernel correctness |
| `gravity_storm.rs` | N-body simulation, compute-heavy |
| `bvh_raytrace_benchmark.rs` | BVH ray tracing on GPU |
| `matrix_lens.rs` | 2-D matrix kernel with shared memory |
| `spectral_lattice.rs` | FFT-style wavefront pattern |

Run any example with:

```sh
cargo run --example rust_device_vector_add
```

---

## `cargo rocm-oxide` subcommands

| Subcommand | Purpose |
|-----------|---------|
| `doctor` | Check all tool and driver prerequisites |
| `new <path>` | Scaffold a new host + device project |
| `build` | Build device crate only |
| `run` | Build and run the project |
| `verify` | Validate generated artifacts against the attached device |
| `inspect` | Print kernel metadata and argument types |
| `pipeline` | Show each build tool invocation and timing |
| `profile` | Run with ROC-Tracer enabled |
| `debug` | Launch under ROCgdb |
