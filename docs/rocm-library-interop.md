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
`rocprim/rocprim.hpp` and `hipcub/hipcub.hpp` available. The first safe wrappers
cover `u32` sum reduction, inclusive sum scan, and exclusive sum scan over
`DeviceBuffer`.

The async methods accept a `DeviceAlgorithmTemporaryStorage` value so callers can
keep rocPRIM temporary storage alive until the stream reaches the operation. The
sync convenience methods allocate the required temporary storage, launch on a
fresh stream, and synchronize before returning.

Because rocFFT and the rocPRIM/hipCUB shim are optional,
`RocmLibraryReport::query()` reports rocBLAS, rocFFT, and rocPRIM/hipCUB
availability independently while the rest of ROCm-Oxide continues to run.

## COMGR

`Comgr::open()` loads `libamd_comgr.so` or `libamd_comgr.so.3` and resolves
`amd_comgr_get_version`. ROCm-Oxide does not yet compile through COMGR, but the
HIPRTC specialization cache keys code objects by compiler backend so a future
COMGR path can coexist with HIPRTC output without cache collisions.

## Matrix Candidates

`HipBlasLt::open()` loads `libhipblaslt.so` or `libhipblaslt.so.1`, resolves
handle create/destroy plus version lookup, and exposes `HipBlasLtHandle` as the
first low-risk handle for future matmul descriptors and heuristic selection.

`MatrixIntegrationReport::query()` reports three CUDA-like matrix replacement
candidates independently:

- hipBLASLt: dynamic library availability plus handle/version smoke coverage.
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
