# cuda-oxide Book Notes For ROCm-Oxide

This file tracks the pieces of the cuda-oxide Book that map cleanly to AMD
ROCm, and the places where ROCm-Oxide needs an AMD-specific design instead of a
literal CUDA port.

References:

- https://nvlabs.github.io/cuda-oxide/gpu-programming/execution-model.html
- https://nvlabs.github.io/cuda-oxide/gpu-programming/launching-kernels.html
- https://nvlabs.github.io/cuda-oxide/gpu-programming/kernels-and-device-functions.html
- https://nvlabs.github.io/cuda-oxide/gpu-programming/memory-and-data-movement.html
- https://nvlabs.github.io/cuda-oxide/async-programming/the-device-operation-model.html

## Execution Model Mapping

cuda-oxide presents the CUDA hierarchy as grid, block, and warp. The ROCm
mapping is dispatch, workgroup, and wavefront:

| cuda-oxide concept | ROCm/HSA concept | ROCm-Oxide API |
| --- | --- | --- |
| `threadIdx.{x,y,z}` | workitem id | `gpu::thread_idx_*()` |
| `blockIdx.{x,y,z}` | workgroup id | `gpu::block_idx_*()` |
| `blockDim.{x,y,z}` | workgroup size from AQL dispatch packet | `gpu::block_dim_*()` |
| `gridDim.{x,y,z}` | grid workitems / workgroup size | `gpu::grid_dim_*()` |
| `thread::index_1d()` | global workitem id | `gpu::global_id_x()` |
| warp | wavefront | `gpu::lane_id()`, `gpu::wavefront_size()` |
| shared memory | LDS / group segment | `gpu::DynamicSharedMem<T>` |

The important design choice is that block size is not a kernel scalar anymore.
ROCm-Oxide now reads workgroup dimensions from the HSA dispatch packet, matching
cuda-oxide's `thread::index_1d()` ergonomics while staying AMD-native.

## Launch Surface

cuda-oxide's `LaunchConfig::for_num_elems(N)` uses a 256-thread default and
ceil-divides the grid. ROCm-Oxide mirrors that:

```rust
let config = rocm_oxide::LaunchConfig::for_num_elems(n);
```

For tuning and GUI demos that need explicit block sizes, ROCm-Oxide keeps:

```rust
let config = rocm_oxide::LaunchConfig::for_num_elems_with_block_size(n, 128);
let image = rocm_oxide::LaunchConfig::for_2d(width, height, 16, 16);
```

The generated bindings continue to validate nonzero grid/block dimensions,
1024-thread block limits, and buffer length contracts before calling HIP.

## Kernel ABI

cuda-oxide scalarizes slices and exposes safe `DisjointSlice` bounds checks.
ROCm-Oxide now has the first AMD device-side slice ABI. Kernels can accept
`rocm_oxide_device::DeviceSlice<T>` and `DeviceSliceMut<T>`; generated host
bindings accept `DeviceBuffer<T>` values, pass pointer/length pairs to HIP, and
validate lengths plus obvious mutable-buffer aliasing before launch. The simple
`add_one`, `vector_add`, and `affine_transform` kernels use this path first. The
larger image, upscaling, stress, and raytrace kernels still need conversion.

## Memory And Synchronization

cuda-oxide's shared memory maps to AMD LDS. Static LDS needs address-space-aware
lowering, but dynamic launch-sized LDS is available through Rust's GPU
workgroup-memory intrinsic and is now exposed as:

```rust
let ptr = unsafe { rocm_oxide_device::DynamicSharedMem::<f32>::get() };
rocm_oxide_device::workgroup_barrier();
```

CUDA warp APIs map to AMD wavefront APIs. ROCm-Oxide uses "wavefront" names for
hardware-specific concepts because AMD wavefront size can differ by architecture
and compiler mode.

## Async Model

cuda-oxide separates "what GPU work should happen" from "which stream schedules
it" with `DeviceOperation`. ROCm-Oxide already has the same shape:

- `DeviceOperation::sync`, `sync_on`, `async_on`, and `async_in`
- `StreamPool` round-robin scheduling
- `DeviceFuture::wait` and `Future` support

The AMD next step is typed generated-kernel operations, so a binding can return a
lazy operation instead of launching immediately.

## CUDA-Only Features

Do not directly port CUDA cluster launch, TMA, WGMMA, NVVM, or nvJitLink APIs.
They need AMD equivalents or separate abstractions:

- Thread block clusters and DSMEM now map to the ROCm feature-parity planner:
  HIP cooperative module launches when supported, otherwise explicit
  stream/graph-scheduled workgroup tiling and global-memory rendezvous.
- TMA/WGMMA map through ROCm-specific abstractions: stream-ordered transfers
  plus LDS tile staging for memory movement, and rocWMMA/rocBLAS/tiled Rust
  kernels for matrix math.
- `rocm_advanced_hardware_rewrite_plan()` records these as source-level rewrite
  targets with `abi_compatible=false` so bindings never promise NVIDIA hardware
  ABI compatibility.
- NVVM/LTOIR/nvJitLink should become a generic artifact/link layer over HSACO
  and ROCm code objects.
