//! Rust-first host/runtime APIs for AMD ROCm.
//!
//! Prefer the root `rocm_oxide::*` re-exports for application code. Public
//! modules are classified in `docs/api-stability.md`; low-level ROCm modules are
//! intentionally public for experiments and generated bindings, but are not all
//! stable API yet.

/// Experimental high-level GPU algorithm helpers backed by ROCm libraries.
pub mod gpu;
/// Experimental low-level HIP wrapper surface.
pub mod hip;
/// Experimental HIPRTC/COMGR runtime compilation and code-object cache surface.
pub mod hiprtc;
/// Experimental optional ROCm library interop surface.
pub mod libraries;
/// Experimental lazy stream and graph operation composition.
pub mod operation;
/// Experimental CUDA-to-ROCm feature planning helpers.
pub mod parity;
/// Experimental rocTX profiling marker/range helpers.
pub mod profiling;
mod runtime;
/// Experimental helpers for GPU-backed Rust tests.
pub mod testing;

pub use gpu::{GpuArray, GpuArray2D};
pub use hip::{
    DeviceBuffer, DevicePod, DeviceVirtualMemory, Event, Global, Graph, GraphExec,
    GraphMemoryAllocation, GraphNode, HipHostFn, ManagedBuffer, ManagedMemoryKind, MemAccessFlags,
    MemAllocationGranularity, MemLocation, MemPool, OwnedMemPool, PinnedHostBuffer, Stream,
};
pub use libraries::{
    Comgr, ComgrVersion, DeviceAlgorithmTemporaryStorage, HipBlasLt, HipBlasLtHandle,
    HipBlasLtHeuristicSummary, HipBlasLtMatmulProblem, HipBlasLtMatrixLayout, LibraryAvailability,
    MatrixIntegrationReport, RocBlas, RocBlasHandle, RocFft, RocFftComplexDirection,
    RocFftExecutionInfo, RocFftPlan, RocFftSession, RocPrim, RocThrust, RocmLibraryReport,
    SgemmLayout,
};
pub use operation::{
    CapturedGraph, DeviceCopyCompletion, DeviceFuture, DeviceMemset, DeviceMemsetCompletion,
    DeviceOperation, DeviceTileTransfer, DeviceToDeviceCopy, ExecutionContext, HostToDeviceCopy,
    KernelLaunchCompletion, StreamPool, Value, copy_device_to_device, copy_host_to_device,
    memset_device, tile_transfer_device_to_device, zero_device,
};
pub use parity::{
    CudaPortingConcept, MatrixMathBackend, RocmAdvancedHardwareRewritePlan,
    RocmCodeObjectInteropPlan, RocmFeaturePlan, RocmFeatureSet, RocmMatrixMathPlan,
    RocmSourceRewriteBoundary, RocmTileTransferPlan, RocmWorkgroupClusterPlan,
    rocm_advanced_hardware_rewrite_plan, rocm_code_object_interop_plan,
    rocm_feature_parity_for_device,
};
pub use profiling::{RocTx, RocTxScopedRange, RocTxVersion};
pub use runtime::{
    AtomicMemoryKind, CooperativeKernelLaunch, Device, DeviceLimits, DeviceProperties, DeviceSlice,
    DeviceSliceMut, Dim3, Error, HostReferenceCaptureVisibility, Kernel, KernelMetadata,
    KernelResource, LaunchConfig, LaunchRecommendation, Module, OccupancyActiveBlocks,
    OccupancyMaxPotentialBlockSize, Result, SystemScopeAtomicVisibility, validate_block_x,
    validate_buffer_len, validate_cooperative_launch_config,
    validate_cooperative_launch_for_device, validate_cooperative_multi_device_launch_for_device,
    validate_device_buffers_disjoint, validate_generated_artifact_metadata, validate_launch_config,
    validate_launch_config_for_limits,
};

/// Common host-side imports for ROCm-Oxide applications.
///
/// This prelude is intentionally conservative. It re-exports the stable SDK
/// path a new host application is most likely to need without pulling in every
/// low-level HIP or ROCm-library experiment.
///
/// ```rust,ignore
/// use rocm_oxide::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        Device, DeviceBuffer, DevicePod, DeviceSlice, DeviceSliceMut, Dim3, Error, Event, Global,
        GpuArray, GpuArray2D, Kernel, KernelMetadata, KernelResource, LaunchConfig, ManagedBuffer,
        Module, PinnedHostBuffer, Result, RocTx, RocTxScopedRange, Stream, gpu, launch, launch_1d,
        launch_1d_with_block, testing::GpuTestContext, validate_block_x, validate_buffer_len,
        validate_device_buffers_disjoint, validate_launch_config,
    };
}

/// Launch a GPU kernel with a typed argument list.
///
/// # Syntax
///
/// ```text
/// launch!(kernel, config, arg0, arg1, ..., argN)?
/// ```
///
/// - `kernel` — a [`Kernel`] handle returned by [`Module::kernel`] or the
///   generated `DeviceKernels` accessor.
/// - `config` — a [`LaunchConfig`] that sets grid/block dimensions and optional
///   dynamic shared memory.
/// - `arg0 .. argN` — kernel arguments **by value**. Types must match the
///   kernel's signature exactly. The macro takes the address of each local copy
///   and builds the `HIP_LAUNCH_PARAM` pointer array for `hipLaunchKernel`.
///
/// Returns `Result<()>`. Append `?` to propagate errors.
///
/// # Example
///
/// ```rust,ignore
/// let device = Device::first()?;
/// let module = device.load_code_object_file(env!("ROCM_OXIDE_DEVICE_HSACO"))?;
/// let kernel = module.kernel(c"vector_add")?;
///
/// let stream = hip::Stream::new()?;
/// let config = LaunchConfig::for_num_elems(n);
///
/// launch!(kernel, config, a_buf.as_device_slice(), b_buf.as_device_slice(), out.as_device_slice_mut(), n)?;
/// stream.synchronize()?;
/// ```
///
/// # Safety
///
/// `launch!` expands to a call to [`Kernel::launch_raw`], which is `unsafe`.
/// The macro invocation must be wrapped in an `unsafe` block:
///
/// ```rust,ignore
/// unsafe {
///     rocm_oxide::launch!(kernel, config, arg0, arg1, n as u64)?;
/// }
/// ```
///
/// You are responsible for ensuring that:
/// - The argument types match the kernel signature exactly.
/// - Pointer arguments point to valid, live device allocations.
/// - Output buffers are not aliased unless the kernel explicitly handles that.
#[macro_export]
macro_rules! launch {
    ($kernel:expr, $config:expr $(,)?) => {{
        let mut params: [*mut ::std::ffi::c_void; 0] = [];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut params = [$crate::__private::arg_ptr(&mut a0)];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut a10 = $a10;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
            $crate::__private::arg_ptr(&mut a10),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut a10 = $a10;
        let mut a11 = $a11;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
            $crate::__private::arg_ptr(&mut a10),
            $crate::__private::arg_ptr(&mut a11),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr, $a12:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut a10 = $a10;
        let mut a11 = $a11;
        let mut a12 = $a12;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
            $crate::__private::arg_ptr(&mut a10),
            $crate::__private::arg_ptr(&mut a11),
            $crate::__private::arg_ptr(&mut a12),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr, $a12:expr, $a13:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut a10 = $a10;
        let mut a11 = $a11;
        let mut a12 = $a12;
        let mut a13 = $a13;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
            $crate::__private::arg_ptr(&mut a10),
            $crate::__private::arg_ptr(&mut a11),
            $crate::__private::arg_ptr(&mut a12),
            $crate::__private::arg_ptr(&mut a13),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr, $a12:expr, $a13:expr, $a14:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut a10 = $a10;
        let mut a11 = $a11;
        let mut a12 = $a12;
        let mut a13 = $a13;
        let mut a14 = $a14;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
            $crate::__private::arg_ptr(&mut a10),
            $crate::__private::arg_ptr(&mut a11),
            $crate::__private::arg_ptr(&mut a12),
            $crate::__private::arg_ptr(&mut a13),
            $crate::__private::arg_ptr(&mut a14),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
    ($kernel:expr, $config:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr, $a12:expr, $a13:expr, $a14:expr, $a15:expr $(,)?) => {{
        let mut a0 = $a0;
        let mut a1 = $a1;
        let mut a2 = $a2;
        let mut a3 = $a3;
        let mut a4 = $a4;
        let mut a5 = $a5;
        let mut a6 = $a6;
        let mut a7 = $a7;
        let mut a8 = $a8;
        let mut a9 = $a9;
        let mut a10 = $a10;
        let mut a11 = $a11;
        let mut a12 = $a12;
        let mut a13 = $a13;
        let mut a14 = $a14;
        let mut a15 = $a15;
        let mut params = [
            $crate::__private::arg_ptr(&mut a0),
            $crate::__private::arg_ptr(&mut a1),
            $crate::__private::arg_ptr(&mut a2),
            $crate::__private::arg_ptr(&mut a3),
            $crate::__private::arg_ptr(&mut a4),
            $crate::__private::arg_ptr(&mut a5),
            $crate::__private::arg_ptr(&mut a6),
            $crate::__private::arg_ptr(&mut a7),
            $crate::__private::arg_ptr(&mut a8),
            $crate::__private::arg_ptr(&mut a9),
            $crate::__private::arg_ptr(&mut a10),
            $crate::__private::arg_ptr(&mut a11),
            $crate::__private::arg_ptr(&mut a12),
            $crate::__private::arg_ptr(&mut a13),
            $crate::__private::arg_ptr(&mut a14),
            $crate::__private::arg_ptr(&mut a15),
        ];
        $kernel.launch_raw($config, &mut params)
    }};
}

/// Launch a one-dimensional GPU kernel using the default block size.
///
/// `launch_1d!` is a small convenience wrapper around
/// [`LaunchConfig::for_num_elems`] and [`launch!`]. It is intended for raw
/// HIPRTC or loaded-code-object kernels whose thread index is computed from
/// `blockIdx.x * blockDim.x + threadIdx.x`.
///
/// # Syntax
///
/// ```text
/// launch_1d!(kernel, num_elems, arg0, arg1, ..., argN)?
/// ```
///
/// This expands to:
///
/// ```rust,ignore
/// let config = LaunchConfig::for_num_elems(num_elems);
/// launch!(kernel, config, arg0, arg1, ..., argN)?;
/// ```
///
/// # Safety
///
/// This macro has the same safety contract as [`launch!`]: the invocation must
/// be wrapped in an `unsafe` block, and the kernel arguments must exactly match
/// the kernel signature.
#[macro_export]
macro_rules! launch_1d {
    ($kernel:expr, $num_elems:expr $(, $arg:expr)* $(,)?) => {{
        let config = $crate::LaunchConfig::for_num_elems($num_elems);
        $crate::launch!($kernel, config $(, $arg)*)
    }};
}

/// Launch a one-dimensional GPU kernel using an explicit `block.x` size.
///
/// `launch_1d_with_block!` is the custom-block-size counterpart to
/// [`launch_1d!`]. It delegates to
/// [`LaunchConfig::for_num_elems_with_block_size`] and then to [`launch!`].
///
/// # Syntax
///
/// ```text
/// launch_1d_with_block!(kernel, num_elems, block_x, arg0, arg1, ..., argN)?
/// ```
///
/// # Safety
///
/// This macro has the same safety contract as [`launch!`].
#[macro_export]
macro_rules! launch_1d_with_block {
    ($kernel:expr, $num_elems:expr, $block_x:expr $(, $arg:expr)* $(,)?) => {{
        let config = $crate::LaunchConfig::for_num_elems_with_block_size($num_elems, $block_x);
        $crate::launch!($kernel, config $(, $arg)*)
    }};
}

#[doc(hidden)]
pub mod __private {
    pub fn arg_ptr<T>(value: &mut T) -> *mut std::ffi::c_void {
        (value as *mut T).cast::<std::ffi::c_void>()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::{Dim3, LaunchConfig, Result};

    #[derive(Default)]
    struct RecordingKernel {
        grid: Cell<Option<Dim3>>,
        block: Cell<Option<Dim3>>,
        arg_count: Cell<Option<usize>>,
    }

    impl RecordingKernel {
        unsafe fn launch_raw(
            &self,
            config: LaunchConfig,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.grid.set(Some(config.grid));
            self.block.set(Some(config.block));
            self.arg_count.set(Some(params.len()));
            Ok(())
        }
    }

    #[test]
    fn launch_1d_macro_uses_default_block_size() {
        let kernel = RecordingKernel::default();

        unsafe {
            crate::launch_1d!(kernel, 1_025usize, 1u32, 2u64)
                .expect("fake kernel launch should succeed");
        }

        assert_eq!(kernel.grid.get(), Some(Dim3::x(5)));
        assert_eq!(
            kernel.block.get(),
            Some(Dim3::x(LaunchConfig::DEFAULT_BLOCK_X))
        );
        assert_eq!(kernel.arg_count.get(), Some(2));
    }

    #[test]
    fn launch_1d_with_block_macro_uses_custom_block_size() {
        let kernel = RecordingKernel::default();

        unsafe {
            crate::launch_1d_with_block!(kernel, 1_025usize, 128u32, 1u32)
                .expect("fake kernel launch should succeed");
        }

        assert_eq!(kernel.grid.get(), Some(Dim3::x(9)));
        assert_eq!(kernel.block.get(), Some(Dim3::x(128)));
        assert_eq!(kernel.arg_count.get(), Some(1));
    }
}
