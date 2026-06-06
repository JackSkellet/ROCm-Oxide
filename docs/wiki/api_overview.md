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
│  Device, Module, Kernel, DeviceBuffer, launch_1d!  │
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
    a: DeviceSlice<f32>,
    b: DeviceSlice<f32>,
    out: DeviceSliceMut<f32>,
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
| `element_index()` | 1-D global element index wrapper | `ThreadIndex` |
| `for_each_element(n, \|i\| ...)` | Run only when the 1-D element index is in bounds | `bool` |
| `DeviceSlice<T>::for_each(\|i, value\| ...)` | Method-style bounded read for this thread | `bool` |
| `DeviceSliceMut<T>::for_each_mut(\|i, out\| ...)` | Method-style bounded write for this thread | `bool` |

The most common pattern for a 1-D kernel:

```rust
for_each_element(n, |i| {
    /* process element i */
});
```

`ThreadIndex` exposes `i.as_usize()` for raw offset math and
`i.is_in_bounds(len)` for explicit bounds checks. `DeviceSlice<T>::read(i)` and
`DeviceSliceMut<T>::write(i, value)` accept `ThreadIndex` directly so
rust-analyzer completion leads from the current element to safe bounded buffer
access. For method-oriented code, `input.for_each(|i, value| { ... })` and
`out.for_each_mut(|i, out| out.write(i, value))` keep the slice itself in the
autocomplete path.

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

Prefer the bounded helpers in ordinary kernels:

```rust
for_each_element(n, |i| {
    if let Some(value) = input.read(i) {
        out.write(i, value + 1.0);
    }
});
```

For lower-level kernels, `read_unchecked(index)` and `write_unchecked(index,
value)` remain available after a caller-provided bounds check.

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

`rocm-oxide-build` emits a `DeviceKernels` struct with one validated method and
one fluent launcher per `#[kernel]` function. The validated method signature
matches the kernel exactly but accepts host-side types: both `DeviceSlice<T>`
and `DeviceSliceMut<T>` map to `&impl AsRef<DeviceBuffer<T>>` (shared
reference), so `DeviceBuffer<T>` and wrappers such as `GpuArray<T>` can use the
same generated launcher. Mutability is handled internally by calling
`.as_mut_ptr()` or `.as_ptr()` depending on the kernel parameter kind. Runtime
overlap and length checks are applied before launch.

Include the bindings in `src/main.rs` (or `src/lib.rs`):

```rust
include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
```

Then load and use:

```rust
let kernels = DeviceKernels::load_embedded(&device)?;
unsafe {
    kernels.fill_indices_launcher().grid_for(n).launch(&out, n)?;
}
```

`load_embedded` reads the HSACO from the bytes embedded by `build.rs` at
compile time — no file path needed at runtime.

---

## Host runtime — `rocm-oxide` (src/)

| Module | Key types |
|--------|----------|
| `runtime` | `Device`, `Module`, `Kernel`, `LaunchConfig`, `DeviceLimits`, `DeviceProperties` |
| `hip` | `DeviceBuffer<T>`, `ManagedBuffer<T>`, `PinnedHostBuffer`, `Stream`, `Event`, `Graph` |
| `gpu` | High-level `reduce_sum`, prefix scans, u32 sort/select/map helpers |
| `libraries` | `RocPrim`, `RocThrust`, `RocBlas`, `RocFft`, `Comgr`, `HipBlasLt` |
| `operation` | `ExecutionContext`, `StreamPool`, `DeviceOperation` (trait), `CapturedGraph` |
| `hiprtc` | `SpecializationCache`, HIPRTC/COMGR compilation (internal) |
| `profiling` | ROC-Tracer integration (internal) |
| `testing` | `GpuTestContext`, `gpu_test!` |

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
`launch!(kernel, config, arg0, ..., argN)?` or
`launch_1d!(kernel, num_elems, arg0, ..., argN)?` for 1-D kernels.

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

### `gpu` Algorithms

The host-side `rocm_oxide::gpu` module provides small rocPRIM/rocThrust-backed
helpers for useful GPU work without writing a custom kernel. The most
discoverable surface is `GpuArray<T>`:

```rust,ignore
use rocm_oxide::{GpuArray, gpu};

let input = gpu::array([1u32, 2, 3, 4])?;
assert_eq!(input.shape(), [4]);
let sum = input.sum()?;
let same_sum = gpu::reduce_sum(&input)?;

let scan = input.exclusive_scan(0)?;
let mapped = input.add_scalar(8)?;

let mapped_into = gpu::empty::<u32>(input.len())?;
input.add_scalar_into(&mapped_into, 3)?;

let params = GpuArray::from_value(7u32)?;
params.write(11)?;
let value = params.item()?;

let filled = gpu::full(3, 42u32)?;
let host = filled.to_list()?;

let mut sortable = GpuArray::from_slice(&[4u32, 1, 3, 2])?;
sortable.sort()?;
let sorted = sortable.download()?;

let flags = GpuArray::from_slice(&[1u8, 0, 1, 0])?;
let (selected, selected_count) = input.compact_by_flags(&flags)?;

let mut keys = GpuArray::from_slice(&[3u32, 1, 2])?;
let mut values = GpuArray::from_slice(&[30u32, 10, 20])?;
keys.sort_by_key(&mut values)?;
```

The lower-level free functions work directly with `DeviceBuffer<T>`:

```rust,ignore
use rocm_oxide::{DeviceBuffer, gpu};

let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4])?;
let sum = gpu::reduce_sum(&input)?;

let scan = DeviceBuffer::<u32>::new(input.len())?;
gpu::exclusive_scan(&input, &scan, 0)?;

let mut sortable = DeviceBuffer::from_slice(&[4u32, 1, 3, 2])?;
gpu::sort(&mut sortable)?;
```

`GpuArray<T>` also has `new`/`empty`, `zeros`/`zeroed`, `repeat`/`full`,
`size`, `shape`, `byte_len`, `upload`/`assign`, `copy_to_slice`, `copy_to`,
`copy_from`, `cloned`, `to_vec`/`to_list`, and `download` helpers for
script-like host code. The `gpu::array`, `gpu::empty`, `gpu::zeros`, and
`gpu::full` constructors are short aliases around the same type.
`reduce_sum`, `inclusive_scan`, and `exclusive_scan` currently support `u32`,
`i32`, and `f32`. Sorting, key/value sort, compact-by-flag, unique,
sort-unique, count, contains, and add-scalar/map-add helpers currently target
`u32`. Free functions in `gpu::` accept either `DeviceBuffer<T>` or wrappers
such as `GpuArray<T>` where practical. Use `RocPrim` and `RocThrust` directly
when you need explicit stream or temporary-storage control.

### GPU Tests

Use `gpu_test!` when a regular Rust test should run on machines with a visible
HIP device and skip cleanly on host-only machines:

```rust,ignore
rocm_oxide::gpu_test!(device_buffer_round_trip, |gpu| {
    eprintln!("running on {}", gpu.arch());
    let buffer = rocm_oxide::DeviceBuffer::from_slice(&[1u32, 2, 3])?;
    assert_eq!(buffer.copy_to_vec()?, [1, 2, 3]);
    Ok(())
});
```

The macro passes a `GpuTestContext` with the opened `Device`, ordinal,
architecture, and launch limits. It treats `Error::NoDevice` as a skip and
returns all other setup failures as test errors.

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

### `launch_1d!` and `launch_1d_with_block!`

Convenience wrappers for raw 1-D kernels. They build a `LaunchConfig` with
`LaunchConfig::for_num_elems(...)` or
`LaunchConfig::for_num_elems_with_block_size(...)`, then delegate to `launch!`.

---

## Examples

The root `examples/` directory contains SDK and diagnostic programs:

| Example | Demonstrates |
|---------|-------------|
| `hello_gpu.rs` | **Start here**: minimal HIPRTC vector add, end-to-end lifecycle |
| `vector_add.rs` | Simple 1-D HIP kernel (C++ via HIPRTC) |
| `gpu_algorithms.rs` | Small rocPRIM/rocThrust-backed algorithms through `rocm_oxide::gpu` |
| `rust_device_vector_add.rs` | Same kernel written in Rust |
| `rust_device_add_one.rs` | Minimal Rust kernel, no slice types |
| `module_global.rs` | `#[device_global]` and `Module::global` |
| `rust_device_generated_bindings.rs` | Using `DeviceKernels` generated struct |
| `pinned_stream_vector_add.rs` | Pinned host memory + streams |
| `performance_probe.rs` | Occupancy, bandwidth, latency profiling |
| `validation_profile.rs` | Verifying kernel correctness |

Run HIPRTC-only examples directly, or enable the source-workspace `device-spike`
feature for examples that use the repository's generated Rust-device bindings:

```sh
cargo run --example hello_gpu
cargo run --features device-spike --example rust_device_vector_add
```

Larger visual, capture, artifact, and benchmark demos live under
`demo-projects/` as separated crates. Run them with their own manifests, for
example:

```sh
cargo run --manifest-path demo-projects/spectral-lattice/Cargo.toml -- --frames 3
cargo run --manifest-path demo-projects/bvh-raytrace-benchmark/Cargo.toml
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
