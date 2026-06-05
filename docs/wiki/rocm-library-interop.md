# ROCm Library Interop

ROCm-Oxide now has an optional library interop layer in `src/libraries.rs`.
It uses `dlopen`/`dlsym` for dynamic libraries so a missing rocFFT install does
not prevent the core compiler/runtime from building. Header-only algorithm
libraries such as rocPRIM and hipCUB are reached through a small C++ shim built
by `build.rs`.

## rocBLAS

`RocBlas::open()` loads `librocblas.so` or `librocblas.so.5`, resolves
`rocblas_create_handle`, `rocblas_destroy_handle`, `rocblas_set_stream`, and
`rocblas_sgemm`, and exposes a safe column-major `sgemm_nn` wrapper for
`DeviceBuffer<f32>`.

The wrapper validates:

- nonzero `m`, `n`, and `k`;
- `lda`, `ldb`, and `ldc` against rocBLAS column-major requirements;
- device-buffer lengths for A, B, and C before calling rocBLAS.

`RocBlasHandle::set_stream()` lets callers put BLAS work on the same HIP stream
used by generated kernel bindings and graph-capturable operations.

## rocFFT

`RocFft::open()` loads `librocfft.so` when installed and resolves setup,
cleanup, plan create/destroy, execution-info stream binding, and execute
symbols. `RocFftSession::create_1d_complex_f32_plan()` provides the first safe
plan wrapper for in-place interleaved complex `f32` buffers.

## rocPRIM / hipCUB

`RocPrim::open()` checks whether the build-time shim was compiled with
`rocprim/rocprim.hpp` and `hipcub/hipcub.hpp` available. The safe wrappers now
cover `u32`, `i32`, and `f32` sum reductions and prefix scans, plus `u32` radix
sort, flagged select, and transform-add over `DeviceBuffer`.

The async methods accept a `DeviceAlgorithmTemporaryStorage` value so callers can
keep rocPRIM temporary storage alive until the stream reaches the operation. The
sync convenience methods allocate the required temporary storage, launch on a
fresh stream, and synchronize before returning.

Because rocFFT and the rocPRIM/hipCUB shim are optional,
`RocmLibraryReport::query()` reports rocBLAS, rocFFT, and rocPRIM/hipCUB
availability independently while the rest of ROCm-Oxide continues to run.

### High-Level `gpu` Algorithms

`rocm_oxide::gpu` is the small host-side algorithms layer above the optional
rocPRIM and rocThrust wrappers. It is intended for users who want common GPU
operations before writing a custom kernel. The easiest path is `GpuArray<T>`,
which wraps `DeviceBuffer<T>` with method-oriented helpers:

```rust,ignore
use rocm_oxide::GpuArray;

let input = GpuArray::from_slice(&[1u32, 2, 3, 4])?;
let sum = input.sum()?;

let scan = input.exclusive_scan(0)?;
let mapped = input.map_add(8)?;

let mut sortable = GpuArray::from_slice(&[4u32, 1, 3, 2])?;
sortable.sort()?;
```

The free-function layer remains available when you already own
`DeviceBuffer<T>` values:

```rust,ignore
use rocm_oxide::{DeviceBuffer, gpu};

let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4])?;
let sum = gpu::reduce_sum(&input)?;

let scan = DeviceBuffer::<u32>::new(input.len())?;
gpu::exclusive_scan(&input, &scan, 0)?;

let mapped = DeviceBuffer::<u32>::new(input.len())?;
gpu::map_add_u32(&input, &mapped, 8)?;

let mut sortable = DeviceBuffer::from_slice(&[4u32, 1, 3, 2])?;
gpu::sort(&mut sortable)?;
```

Supported first-pass operations:

- `reduce_sum`, `inclusive_scan`, and `exclusive_scan` for `u32`, `i32`, and
  `f32`;
- in-place `sort`, out-of-place `sort_keys_u32`, `sort_by_key_u32`,
  `unique_u32`, and `count_eq_u32`;
- `select_flagged_u32` and `map_add_u32`;
- `fill_zero` and byte-pattern `fill_bytes`.

The helpers allocate their own temporary storage and synchronize before
returning. Use the lower-level `RocPrim` and `RocThrust` methods when the caller
needs explicit stream ordering, temporary-storage reuse, or APIs not exposed by
the high-level layer yet.

## COMGR

`Comgr::open()` loads `libamd_comgr.so` or `libamd_comgr.so.3` and resolves
the compiler-driver entry points needed to create data sets, action metadata,
logs, relocatable objects, and executable code objects.
`Comgr::compile_hip_source_to_code_object()` compiles HIP source to a
relocatable, links it to an executable HSACO payload, and feeds the same
persistent code-object cache shape as the HIPRTC backend.

## Matrix Candidates

`HipBlasLt::open()` loads `libhipblaslt.so` or `libhipblaslt.so.1`, resolves
handle create/destroy, version lookup, matmul descriptor/layout/preference
creation, `hipblasLtMatmulAlgoGetHeuristic`, and `hipblasLtMatmul`.
`HipBlasLtMatmulProblem::sgemm_nn()` validates a checked FP32 column-major
matmul shape before `HipBlasLtHandle::sgemm_nn_heuristics()` asks hipBLASLt for
candidate algorithms and summarizes the best workspace/state/wave estimate.
`HipBlasLtHandle::sgemm_nn()` chooses the best returned algorithm, allocates
temporary workspace when required, executes a synchronous SGEMM over
`DeviceBuffer<f32>` values, and validates the output buffer sizes before the
library call. `HipBlasLtHandle::sgemm_nn_on_stream()` exposes the same checked
launch on a caller-provided HIP stream when the caller owns the stream-ordered
buffer and workspace lifetimes.

`MatrixIntegrationReport::query()` reports three CUDA-like matrix replacement
candidates independently:

- hipBLASLt: dynamic library availability plus handle/version and SGEMM
  descriptor, heuristic, and execution smoke coverage.
- Composable Kernel: installed headers, device GEMM archive, and CMake package.
- rocWMMA: header availability. This local ROCm install does not include
  `rocwmma/rocwmma.hpp`, so ROCm-Oxide must not assume a rocWMMA path exists.

Primary references:
[rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/),
[rocFFT API](https://rocm.docs.amd.com/projects/rocFFT/en/latest/reference/api.html),
[rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/),
[hipCUB](https://rocm.docs.amd.com/projects/hipCUB/en/latest/),
[COMGR](https://rocm.docs.amd.com/projects/comgr/en/latest/),
[hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/),
[Composable Kernel](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/),
[HIP streams](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html).
