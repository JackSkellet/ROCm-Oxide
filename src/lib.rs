pub mod hip;
pub mod hiprtc;
pub mod libraries;
pub mod operation;
pub mod parity;
pub mod profiling;
mod runtime;

pub use hip::{
    DeviceBuffer, DeviceVirtualMemory, Event, Global, ManagedBuffer, ManagedMemoryKind,
    MemAccessFlags, MemAllocationGranularity, MemLocation, MemPool, OwnedMemPool, PinnedHostBuffer,
    Stream,
};
pub use libraries::{
    Comgr, ComgrVersion, DeviceAlgorithmTemporaryStorage, HipBlasLt, HipBlasLtHandle,
    HipBlasLtHeuristicSummary, HipBlasLtMatmulProblem, HipBlasLtMatrixLayout, LibraryAvailability,
    MatrixIntegrationReport, RocBlas, RocBlasHandle, RocFft, RocFftComplexDirection,
    RocFftExecutionInfo, RocFftPlan, RocFftSession, RocPrim, RocmLibraryReport, SgemmLayout,
};
pub use operation::{
    CapturedGraph, DeviceFuture, DeviceOperation, ExecutionContext, KernelLaunchCompletion,
    StreamPool, Value,
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
    AtomicMemoryKind, Device, DeviceLimits, DeviceProperties, DeviceSlice, DeviceSliceMut, Dim3,
    Error, HostReferenceCaptureVisibility, Kernel, KernelMetadata, KernelResource, LaunchConfig,
    LaunchRecommendation, Module, OccupancyActiveBlocks, OccupancyMaxPotentialBlockSize, Result,
    SystemScopeAtomicVisibility, validate_block_x, validate_buffer_len,
    validate_cooperative_launch_config, validate_device_buffers_disjoint, validate_launch_config,
    validate_launch_config_for_limits,
};

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

#[doc(hidden)]
pub mod __private {
    pub fn arg_ptr<T>(value: &mut T) -> *mut std::ffi::c_void {
        (value as *mut T).cast::<std::ffi::c_void>()
    }
}
