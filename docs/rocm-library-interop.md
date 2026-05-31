# ROCm Library Interop

ROCm-Oxide now has an optional library interop layer in `src/libraries.rs`.
It uses `dlopen`/`dlsym` instead of hard links so a missing rocFFT install does
not prevent the core compiler/runtime from building.

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

The current workstation has rocBLAS installed under `/opt/rocm/lib` but no
rocFFT headers or library. Because the loader is optional,
`RocmLibraryReport::query()` can report rocFFT as unavailable while rocBLAS
interop and the rest of ROCm-Oxide continue to run.

Primary references:
[rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/),
[rocFFT API](https://rocm.docs.amd.com/projects/rocFFT/en/latest/reference/api.html),
[HIP streams](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html).
