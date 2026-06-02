use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

pub type HipError = c_int;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;
pub type HipGraphNode = *mut c_void;
pub type HipGraphExecUpdateResult = c_int;
pub type HipMemPool = *mut c_void;
pub type HipMemGenericAllocationHandle = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIP_ERROR_INVALID_VALUE: HipError = 1;
pub const HIP_ERROR_NOT_SUPPORTED: HipError = 801;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const HIP_DEVICE_MALLOC_FINEGRAINED: c_uint = 0x1;
pub const HIP_HOST_MALLOC_MAPPED: c_uint = 0x2;
pub const HIP_HOST_MALLOC_COHERENT: c_uint = 0x4000_0000;
pub const HIP_MEM_ATTACH_GLOBAL: c_uint = 0x01;
pub const HIP_MEM_ADVISE_SET_COARSE_GRAIN: c_int = 100;
pub const HIP_MEM_ADVISE_UNSET_COARSE_GRAIN: c_int = 101;
// hipDeviceAttribute_t discriminants used through hipDeviceGetAttribute.
// Values match ROCm HIP 7.2 headers and the CUDA-compatible enum ordering.
pub const HIP_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: c_int = 2;
pub const HIP_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: c_int = 3;
pub const HIP_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: c_int = 4;
pub const HIP_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: c_int = 9;
pub const HIP_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: c_int = 10;
pub const HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH: c_int = 11;
pub const HIP_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: c_int = 13;
pub const HIP_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: c_int = 15;
pub const HIP_DEVICE_ATTRIBUTE_MANAGED_MEMORY: c_int = 24;
pub const HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 26;
pub const HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 27;
pub const HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 28;
pub const HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 56;
pub const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 63;
pub const HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: c_int = 65;
pub const HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: c_int = 66;
pub const HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 74;
pub const HIP_DEVICE_ATTRIBUTE_SHARED_MEM_PER_BLOCK_OPTIN: c_int = 75;
pub const HIP_DEVICE_ATTRIBUTE_SHARED_MEM_PER_MULTIPROCESSOR: c_int = 76;
pub const HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: c_int = 85;
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 87;
pub const HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: c_int = 88;
pub const HIP_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: c_int = 90;
pub const HIP_DEVICE_ATTRIBUTE_CLOCK_INSTRUCTION_RATE: c_int = 10000;
pub const HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE: c_int = 10017;
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;
pub const HIP_MEM_POOL_REUSE_FOLLOW_EVENT_DEPENDENCIES: c_int = 0x1;
pub const HIP_MEM_POOL_REUSE_ALLOW_OPPORTUNISTIC: c_int = 0x2;
pub const HIP_MEM_POOL_REUSE_ALLOW_INTERNAL_DEPENDENCIES: c_int = 0x3;
pub const HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD: c_int = 0x4;
pub const HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT: c_int = 0x5;
pub const HIP_MEM_POOL_ATTR_RESERVED_MEM_HIGH: c_int = 0x6;
pub const HIP_MEM_POOL_ATTR_USED_MEM_CURRENT: c_int = 0x7;
pub const HIP_MEM_POOL_ATTR_USED_MEM_HIGH: c_int = 0x8;
pub const HIP_GRAPH_EXEC_UPDATE_SUCCESS: HipGraphExecUpdateResult = 0;
pub const HIP_MEM_LOCATION_TYPE_DEVICE: c_int = 1;
pub const HIP_MEM_LOCATION_TYPE_HOST: c_int = 2;
pub const HIP_MEM_LOCATION_TYPE_HOST_NUMA: c_int = 3;
pub const HIP_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT: c_int = 4;
pub const HIP_MEM_ACCESS_FLAGS_PROT_NONE: c_int = 0;
pub const HIP_MEM_ACCESS_FLAGS_PROT_READ: c_int = 1;
pub const HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE: c_int = 3;
pub const HIP_MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
pub const HIP_MEM_HANDLE_TYPE_NONE: c_int = 0;
pub const HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM: c_int = 0;
pub const HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED: c_int = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct HipMemLocation {
    location_type: c_int,
    id: c_int,
}

#[repr(C)]
struct HipMemAccessDesc {
    location: HipMemLocation,
    flags: c_int,
}

#[repr(C)]
struct HipMemPoolProps {
    alloc_type: c_int,
    handle_types: c_int,
    location: HipMemLocation,
    win32_security_attributes: *mut c_void,
    max_size: usize,
    reserved: [u8; 56],
}

#[repr(C)]
struct HipMemAllocationFlags {
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
}

#[repr(C)]
struct HipMemAllocationProp {
    allocation_type: c_int,
    requested_handle_type: c_int,
    location: HipMemLocation,
    win32_handle_metadata: *mut c_void,
    alloc_flags: HipMemAllocationFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HipDim3 {
    x: c_uint,
    y: c_uint,
    z: c_uint,
}

impl HipDim3 {
    const fn new(dims: (u32, u32, u32)) -> Self {
        Self {
            x: dims.0,
            y: dims.1,
            z: dims.2,
        }
    }
}

#[repr(C)]
struct HipKernelNodeParams {
    block_dim: HipDim3,
    extra: *mut *mut c_void,
    func: *mut c_void,
    grid_dim: HipDim3,
    kernel_params: *mut *mut c_void,
    shared_mem_bytes: c_uint,
}

#[repr(C)]
struct HipMemsetParams {
    dst: *mut c_void,
    element_size: c_uint,
    height: usize,
    pitch: usize,
    value: c_uint,
    width: usize,
}

#[repr(C)]
struct HipMemAllocNodeParams {
    pool_props: HipMemPoolProps,
    access_descs: *const HipMemAccessDesc,
    access_desc_count: usize,
    bytesize: usize,
    dptr: *mut c_void,
}

unsafe extern "C" {
    fn hipGetErrorString(error: HipError) -> *const c_char;
    fn hipGetDeviceCount(count: *mut c_int) -> HipError;
    fn hipGetDevice(device_id: *mut c_int) -> HipError;
    fn hipSetDevice(device_id: c_int) -> HipError;
    fn hipDeviceGetAttribute(value: *mut c_int, attr: c_int, device_id: c_int) -> HipError;
    fn hipDeviceCanAccessPeer(
        can_access_peer: *mut c_int,
        device_id: c_int,
        peer_device_id: c_int,
    ) -> HipError;
    fn hipDeviceEnablePeerAccess(peer_device_id: c_int, flags: c_uint) -> HipError;
    fn hipDeviceDisablePeerAccess(peer_device_id: c_int) -> HipError;
    fn hipDeviceGetDefaultMemPool(mem_pool: *mut HipMemPool, device: c_int) -> HipError;
    fn hipDeviceGetMemPool(mem_pool: *mut HipMemPool, device: c_int) -> HipError;
    fn hipDeviceSetMemPool(device: c_int, mem_pool: HipMemPool) -> HipError;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    fn hipExtMallocWithFlags(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> HipError;
    fn hipMallocManaged(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> HipError;
    fn hipMallocAsync(ptr: *mut *mut c_void, size: usize, stream: HipStream) -> HipError;
    fn hipMallocFromPoolAsync(
        ptr: *mut *mut c_void,
        size: usize,
        mem_pool: HipMemPool,
        stream: HipStream,
    ) -> HipError;
    fn hipFree(ptr: *mut c_void) -> HipError;
    fn hipFreeAsync(ptr: *mut c_void, stream: HipStream) -> HipError;
    fn hipMemAdvise(ptr: *const c_void, count: usize, advice: c_int, device: c_int) -> HipError;
    fn hipMemPoolTrimTo(mem_pool: HipMemPool, min_bytes_to_hold: usize) -> HipError;
    fn hipMemPoolSetAttribute(mem_pool: HipMemPool, attr: c_int, value: *mut c_void) -> HipError;
    fn hipMemPoolGetAttribute(mem_pool: HipMemPool, attr: c_int, value: *mut c_void) -> HipError;
    fn hipMemPoolCreate(mem_pool: *mut HipMemPool, pool_props: *const HipMemPoolProps) -> HipError;
    fn hipMemPoolDestroy(mem_pool: HipMemPool) -> HipError;
    fn hipMemPoolSetAccess(
        mem_pool: HipMemPool,
        desc_list: *const HipMemAccessDesc,
        count: usize,
    ) -> HipError;
    fn hipMemPoolGetAccess(
        flags: *mut c_int,
        mem_pool: HipMemPool,
        location: *mut HipMemLocation,
    ) -> HipError;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> HipError;
    fn hipHostGetDevicePointer(
        device_ptr: *mut *mut c_void,
        host_ptr: *mut c_void,
        flags: c_uint,
    ) -> HipError;
    fn hipHostFree(ptr: *mut c_void) -> HipError;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> HipError;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
        stream: HipStream,
    ) -> HipError;
    fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> HipError;
    fn hipMemsetAsync(dst: *mut c_void, value: c_int, size: usize, stream: HipStream) -> HipError;
    fn hipDeviceSynchronize() -> HipError;
    fn hipStreamCreate(stream: *mut HipStream) -> HipError;
    fn hipStreamDestroy(stream: HipStream) -> HipError;
    fn hipStreamSynchronize(stream: HipStream) -> HipError;
    fn hipStreamBeginCapture(stream: HipStream, mode: c_int) -> HipError;
    fn hipStreamEndCapture(stream: HipStream, graph: *mut HipGraph) -> HipError;
    fn hipEventCreate(event: *mut HipEvent) -> HipError;
    fn hipEventDestroy(event: HipEvent) -> HipError;
    fn hipEventRecord(event: HipEvent, stream: HipStream) -> HipError;
    fn hipEventSynchronize(event: HipEvent) -> HipError;
    fn hipEventElapsedTime(ms: *mut f32, start: HipEvent, stop: HipEvent) -> HipError;
    fn hipModuleLoadData(module: *mut HipModule, image: *const c_void) -> HipError;
    fn hipModuleUnload(module: HipModule) -> HipError;
    fn hipModuleGetFunction(
        function: *mut HipFunction,
        module: HipModule,
        name: *const c_char,
    ) -> HipError;
    fn hipModuleGetGlobal(
        dptr: *mut *mut c_void,
        bytes: *mut usize,
        module: HipModule,
        name: *const c_char,
    ) -> HipError;
    fn hipModuleLaunchKernel(
        function: HipFunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
    fn hipModuleLaunchCooperativeKernel(
        function: HipFunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
    ) -> HipError;
    fn hipModuleOccupancyMaxPotentialBlockSize(
        grid_size: *mut c_int,
        block_size: *mut c_int,
        function: HipFunction,
        dynamic_shared_mem_per_block: usize,
        block_size_limit: c_int,
    ) -> HipError;
    fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks_per_multiprocessor: *mut c_int,
        function: HipFunction,
        block_size: c_int,
        dynamic_shared_mem_per_block: usize,
    ) -> HipError;
    fn hipGraphInstantiate(
        graph_exec: *mut HipGraphExec,
        graph: HipGraph,
        error_node: *mut HipGraphNode,
        log_buffer: *mut c_char,
        buffer_size: usize,
    ) -> HipError;
    fn hipGraphCreate(graph: *mut HipGraph, flags: c_uint) -> HipError;
    fn hipGraphAddDependencies(
        graph: HipGraph,
        from: *const HipGraphNode,
        to: *const HipGraphNode,
        num_dependencies: usize,
    ) -> HipError;
    fn hipGraphAddKernelNode(
        graph_node: *mut HipGraphNode,
        graph: HipGraph,
        dependencies: *const HipGraphNode,
        num_dependencies: usize,
        node_params: *const HipKernelNodeParams,
    ) -> HipError;
    fn hipGraphKernelNodeSetParams(
        node: HipGraphNode,
        node_params: *const HipKernelNodeParams,
    ) -> HipError;
    fn hipGraphAddMemcpyNode1D(
        graph_node: *mut HipGraphNode,
        graph: HipGraph,
        dependencies: *const HipGraphNode,
        num_dependencies: usize,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> HipError;
    fn hipGraphMemcpyNodeSetParams1D(
        node: HipGraphNode,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> HipError;
    fn hipGraphAddMemsetNode(
        graph_node: *mut HipGraphNode,
        graph: HipGraph,
        dependencies: *const HipGraphNode,
        num_dependencies: usize,
        node_params: *const HipMemsetParams,
    ) -> HipError;
    fn hipGraphMemsetNodeSetParams(
        node: HipGraphNode,
        node_params: *const HipMemsetParams,
    ) -> HipError;
    fn hipGraphAddMemAllocNode(
        graph_node: *mut HipGraphNode,
        graph: HipGraph,
        dependencies: *const HipGraphNode,
        num_dependencies: usize,
        node_params: *mut HipMemAllocNodeParams,
    ) -> HipError;
    fn hipGraphAddMemFreeNode(
        graph_node: *mut HipGraphNode,
        graph: HipGraph,
        dependencies: *const HipGraphNode,
        num_dependencies: usize,
        dev_ptr: *mut c_void,
    ) -> HipError;
    fn hipGraphAddEmptyNode(
        graph_node: *mut HipGraphNode,
        graph: HipGraph,
        dependencies: *const HipGraphNode,
        num_dependencies: usize,
    ) -> HipError;
    fn hipGraphExecUpdate(
        graph_exec: HipGraphExec,
        graph: HipGraph,
        error_node: *mut HipGraphNode,
        update_result: *mut HipGraphExecUpdateResult,
    ) -> HipError;
    fn hipGraphLaunch(graph_exec: HipGraphExec, stream: HipStream) -> HipError;
    fn hipGraphDestroy(graph: HipGraph) -> HipError;
    fn hipGraphExecDestroy(graph_exec: HipGraphExec) -> HipError;
    fn hipMemAddressReserve(
        ptr: *mut *mut c_void,
        size: usize,
        alignment: usize,
        addr: *mut c_void,
        flags: u64,
    ) -> HipError;
    fn hipMemAddressFree(ptr: *mut c_void, size: usize) -> HipError;
    fn hipMemCreate(
        handle: *mut HipMemGenericAllocationHandle,
        size: usize,
        props: *const HipMemAllocationProp,
        flags: u64,
    ) -> HipError;
    fn hipMemRelease(handle: HipMemGenericAllocationHandle) -> HipError;
    fn hipMemGetAllocationGranularity(
        granularity: *mut usize,
        props: *const HipMemAllocationProp,
        option: c_int,
    ) -> HipError;
    fn hipMemMap(
        ptr: *mut c_void,
        size: usize,
        offset: usize,
        handle: HipMemGenericAllocationHandle,
        flags: u64,
    ) -> HipError;
    fn hipMemUnmap(ptr: *mut c_void, size: usize) -> HipError;
    fn hipMemSetAccess(
        ptr: *mut c_void,
        size: usize,
        desc: *const HipMemAccessDesc,
        count: usize,
    ) -> HipError;
    fn hipMemGetAccess(
        flags: *mut u64,
        location: *const HipMemLocation,
        ptr: *mut c_void,
    ) -> HipError;
}

#[derive(Debug, Clone)]
pub struct Error {
    code: Option<HipError>,
    message: String,
}

impl Error {
    fn from_code(code: HipError) -> Self {
        let message = unsafe {
            let ptr = hipGetErrorString(code);
            if ptr.is_null() {
                format!("HIP error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Self {
            code: Some(code),
            message,
        }
    }

    pub(crate) fn invalid_value(message: impl Into<String>) -> Self {
        Self {
            code: None,
            message: message.into(),
        }
    }

    pub const fn code(&self) -> Option<HipError> {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = self.code {
            write!(f, "{} ({})", self.message, code)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn check(code: HipError) -> Result<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_code(code))
    }
}

pub fn device_attribute(device_id: i32, attribute: c_int) -> Result<u32> {
    let mut value = 0;
    unsafe {
        check(hipDeviceGetAttribute(&mut value, attribute, device_id))?;
    }
    u32_from_hip_int(&format!("HIP device attribute {attribute}"), value)
}

pub fn device_attribute_bool(device_id: i32, attribute: c_int) -> Result<bool> {
    Ok(device_attribute(device_id, attribute)? != 0)
}

pub fn can_access_peer(device_id: i32, peer_device_id: i32) -> Result<bool> {
    let mut value = 0;
    unsafe {
        check(hipDeviceCanAccessPeer(
            &mut value,
            device_id,
            peer_device_id,
        ))?;
    }
    Ok(value != 0)
}

pub fn enable_peer_access(peer_device_id: i32) -> Result<()> {
    unsafe { check(hipDeviceEnablePeerAccess(peer_device_id, 0)) }
}

pub fn disable_peer_access(peer_device_id: i32) -> Result<()> {
    unsafe { check(hipDeviceDisablePeerAccess(peer_device_id)) }
}

fn u32_from_hip_int(label: &str, value: c_int) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::invalid_value(format!("{label} returned negative value {value}")))
}

fn c_int_from_u32(label: &str, value: u32) -> Result<c_int> {
    c_int::try_from(value).map_err(|_| {
        Error::invalid_value(format!(
            "{label} value {value} exceeds HIP int parameter range"
        ))
    })
}

fn checked_allocation_bytes<T>(len: usize, label: &str) -> Result<usize> {
    len.checked_mul(size_of::<T>()).ok_or_else(|| {
        Error::invalid_value(format!(
            "{label} allocation size overflow for {len} elements"
        ))
    })
}

fn validate_slice_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::invalid_value(format!(
            "{label} length mismatch: got {actual}, expected {expected}"
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagedMemoryKind {
    FineGrain,
    CoarseGrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemLocation {
    Device(i32),
    Host,
    HostNuma(i32),
    HostNumaCurrent,
}

impl MemLocation {
    const fn as_raw(self) -> HipMemLocation {
        match self {
            Self::Device(id) => HipMemLocation {
                location_type: HIP_MEM_LOCATION_TYPE_DEVICE,
                id,
            },
            Self::Host => HipMemLocation {
                location_type: HIP_MEM_LOCATION_TYPE_HOST,
                id: 0,
            },
            Self::HostNuma(id) => HipMemLocation {
                location_type: HIP_MEM_LOCATION_TYPE_HOST_NUMA,
                id,
            },
            Self::HostNumaCurrent => HipMemLocation {
                location_type: HIP_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
                id: 0,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAccessFlags {
    None,
    Read,
    ReadWrite,
}

impl MemAccessFlags {
    const fn as_raw(self) -> c_int {
        match self {
            Self::None => HIP_MEM_ACCESS_FLAGS_PROT_NONE,
            Self::Read => HIP_MEM_ACCESS_FLAGS_PROT_READ,
            Self::ReadWrite => HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE,
        }
    }

    fn from_raw(raw: c_int) -> Result<Self> {
        match raw {
            HIP_MEM_ACCESS_FLAGS_PROT_NONE => Ok(Self::None),
            HIP_MEM_ACCESS_FLAGS_PROT_READ => Ok(Self::Read),
            HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE => Ok(Self::ReadWrite),
            _ => Err(Error::invalid_value(format!(
                "HIP returned unknown memory access flags {raw}"
            ))),
        }
    }

    fn from_u64(raw: u64) -> Result<Self> {
        let raw = c_int::try_from(raw).map_err(|_| {
            Error::invalid_value(format!(
                "HIP returned out-of-range memory access flags {raw}"
            ))
        })?;
        Self::from_raw(raw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAllocationGranularity {
    Minimum,
    Recommended,
}

impl MemAllocationGranularity {
    const fn as_raw(self) -> c_int {
        match self {
            Self::Minimum => HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM,
            Self::Recommended => HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
        }
    }
}

#[derive(Clone, Copy)]
pub struct MemPool {
    raw: HipMemPool,
}

pub struct OwnedMemPool {
    raw: HipMemPool,
}

unsafe impl Send for MemPool {}
unsafe impl Sync for MemPool {}
unsafe impl Send for OwnedMemPool {}
unsafe impl Sync for OwnedMemPool {}

impl OwnedMemPool {
    pub fn new_for_device(device_id: i32) -> Result<Self> {
        let props = hip_mem_pool_props_for_device(device_id);
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipMemPoolCreate(&mut raw, &props))?;
        }
        Ok(Self { raw })
    }

    pub const fn as_pool(&self) -> MemPool {
        MemPool { raw: self.raw }
    }

    pub const fn as_raw(&self) -> HipMemPool {
        self.raw
    }
}

impl Drop for OwnedMemPool {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipMemPoolDestroy(self.raw);
            }
        }
    }
}

impl MemPool {
    pub fn default_for_device(device_id: i32) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipDeviceGetDefaultMemPool(&mut raw, device_id))?;
        }
        Ok(Self { raw })
    }

    pub fn current_for_device(device_id: i32) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipDeviceGetMemPool(&mut raw, device_id))?;
        }
        Ok(Self { raw })
    }

    pub fn set_current_for_device(self, device_id: i32) -> Result<()> {
        unsafe { check(hipDeviceSetMemPool(device_id, self.raw)) }
    }

    pub fn create_for_device(device_id: i32) -> Result<OwnedMemPool> {
        OwnedMemPool::new_for_device(device_id)
    }

    pub fn trim_to(self, min_bytes_to_hold: usize) -> Result<()> {
        unsafe { check(hipMemPoolTrimTo(self.raw, min_bytes_to_hold)) }
    }

    pub fn release_threshold(self) -> Result<u64> {
        self.get_u64_attr(HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD)
    }

    pub fn set_release_threshold(self, bytes: u64) -> Result<()> {
        self.set_u64_attr(HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD, bytes)
    }

    pub fn reserved_mem_current(self) -> Result<u64> {
        self.get_u64_attr(HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT)
    }

    pub fn reserved_mem_high(self) -> Result<u64> {
        self.get_u64_attr(HIP_MEM_POOL_ATTR_RESERVED_MEM_HIGH)
    }

    pub fn used_mem_current(self) -> Result<u64> {
        self.get_u64_attr(HIP_MEM_POOL_ATTR_USED_MEM_CURRENT)
    }

    pub fn used_mem_high(self) -> Result<u64> {
        self.get_u64_attr(HIP_MEM_POOL_ATTR_USED_MEM_HIGH)
    }

    pub fn set_reuse_follow_event_dependencies(self, enabled: bool) -> Result<()> {
        self.set_i32_attr(
            HIP_MEM_POOL_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            if enabled { 1 } else { 0 },
        )
    }

    pub fn reuse_follow_event_dependencies(self) -> Result<bool> {
        Ok(self.get_i32_attr(HIP_MEM_POOL_REUSE_FOLLOW_EVENT_DEPENDENCIES)? != 0)
    }

    pub fn set_reuse_allow_opportunistic(self, enabled: bool) -> Result<()> {
        self.set_i32_attr(
            HIP_MEM_POOL_REUSE_ALLOW_OPPORTUNISTIC,
            if enabled { 1 } else { 0 },
        )
    }

    pub fn reuse_allow_opportunistic(self) -> Result<bool> {
        Ok(self.get_i32_attr(HIP_MEM_POOL_REUSE_ALLOW_OPPORTUNISTIC)? != 0)
    }

    pub fn set_reuse_allow_internal_dependencies(self, enabled: bool) -> Result<()> {
        self.set_i32_attr(
            HIP_MEM_POOL_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            if enabled { 1 } else { 0 },
        )
    }

    pub fn reuse_allow_internal_dependencies(self) -> Result<bool> {
        Ok(self.get_i32_attr(HIP_MEM_POOL_REUSE_ALLOW_INTERNAL_DEPENDENCIES)? != 0)
    }

    pub fn set_access(self, location: MemLocation, flags: MemAccessFlags) -> Result<()> {
        let desc = HipMemAccessDesc {
            location: location.as_raw(),
            flags: flags.as_raw(),
        };
        unsafe { check(hipMemPoolSetAccess(self.raw, &desc, 1)) }
    }

    pub fn access(self, location: MemLocation) -> Result<MemAccessFlags> {
        let mut location = location.as_raw();
        let mut flags = 0;
        unsafe {
            check(hipMemPoolGetAccess(&mut flags, self.raw, &mut location))?;
        }
        MemAccessFlags::from_raw(flags)
    }

    pub const fn as_raw(self) -> HipMemPool {
        self.raw
    }

    fn get_u64_attr(self, attr: c_int) -> Result<u64> {
        let mut value = 0u64;
        unsafe {
            check(hipMemPoolGetAttribute(
                self.raw,
                attr,
                (&mut value as *mut u64).cast::<c_void>(),
            ))?;
        }
        Ok(value)
    }

    fn set_u64_attr(self, attr: c_int, mut value: u64) -> Result<()> {
        unsafe {
            check(hipMemPoolSetAttribute(
                self.raw,
                attr,
                (&mut value as *mut u64).cast::<c_void>(),
            ))
        }
    }

    fn get_i32_attr(self, attr: c_int) -> Result<i32> {
        let mut value = 0i32;
        unsafe {
            check(hipMemPoolGetAttribute(
                self.raw,
                attr,
                (&mut value as *mut i32).cast::<c_void>(),
            ))?;
        }
        Ok(value)
    }

    fn set_i32_attr(self, attr: c_int, mut value: i32) -> Result<()> {
        unsafe {
            check(hipMemPoolSetAttribute(
                self.raw,
                attr,
                (&mut value as *mut i32).cast::<c_void>(),
            ))
        }
    }
}

fn hip_mem_pool_props_for_device(device_id: i32) -> HipMemPoolProps {
    HipMemPoolProps {
        alloc_type: HIP_MEM_ALLOCATION_TYPE_PINNED,
        handle_types: HIP_MEM_HANDLE_TYPE_NONE,
        location: MemLocation::Device(device_id).as_raw(),
        win32_security_attributes: ptr::null_mut(),
        max_size: 0,
        reserved: [0; 56],
    }
}

pub struct DeviceVirtualMemory {
    ptr: *mut c_void,
    size: usize,
    requested_size: usize,
    handle: HipMemGenericAllocationHandle,
}

unsafe impl Send for DeviceVirtualMemory {}
unsafe impl Sync for DeviceVirtualMemory {}

impl DeviceVirtualMemory {
    pub fn new_for_device(device_id: i32, requested_size: usize) -> Result<Self> {
        if requested_size == 0 {
            return Err(Error::invalid_value(
                "HIP virtual memory reservations must be nonzero",
            ));
        }
        let granularity =
            Self::allocation_granularity(device_id, MemAllocationGranularity::Recommended)?;
        let size =
            round_up_to_multiple(requested_size, granularity, "HIP virtual memory allocation")?;
        let props = hip_mem_allocation_props_for_device(device_id);
        let mut ptr = ptr::null_mut();
        let mut handle = ptr::null_mut();

        unsafe {
            if let Err(err) = check(hipMemAddressReserve(
                &mut ptr,
                size,
                granularity,
                ptr::null_mut(),
                0,
            )) {
                return Err(err);
            }
            if let Err(err) = check(hipMemCreate(&mut handle, size, &props, 0)) {
                let _ = hipMemAddressFree(ptr, size);
                return Err(err);
            }
            if let Err(err) = check(hipMemMap(ptr, size, 0, handle, 0)) {
                let _ = hipMemRelease(handle);
                let _ = hipMemAddressFree(ptr, size);
                return Err(err);
            }
            let desc = HipMemAccessDesc {
                location: MemLocation::Device(device_id).as_raw(),
                flags: MemAccessFlags::ReadWrite.as_raw(),
            };
            if let Err(err) = check(hipMemSetAccess(ptr, size, &desc, 1)) {
                let _ = hipMemUnmap(ptr, size);
                let _ = hipMemRelease(handle);
                let _ = hipMemAddressFree(ptr, size);
                return Err(err);
            }
        }

        Ok(Self {
            ptr,
            size,
            requested_size,
            handle,
        })
    }

    pub fn allocation_granularity(
        device_id: i32,
        granularity: MemAllocationGranularity,
    ) -> Result<usize> {
        let props = hip_mem_allocation_props_for_device(device_id);
        let mut value = 0usize;
        unsafe {
            check(hipMemGetAllocationGranularity(
                &mut value,
                &props,
                granularity.as_raw(),
            ))?;
        }
        Ok(value)
    }

    pub fn set_access(&self, location: MemLocation, flags: MemAccessFlags) -> Result<()> {
        let desc = HipMemAccessDesc {
            location: location.as_raw(),
            flags: flags.as_raw(),
        };
        unsafe { check(hipMemSetAccess(self.ptr, self.size, &desc, 1)) }
    }

    pub fn access(&self, location: MemLocation) -> Result<MemAccessFlags> {
        let location = location.as_raw();
        let mut flags = 0u64;
        unsafe {
            check(hipMemGetAccess(&mut flags, &location, self.ptr))?;
        }
        MemAccessFlags::from_u64(flags)
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr.cast::<T>()
    }

    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.ptr.cast::<T>()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn requested_size(&self) -> usize {
        self.requested_size
    }
}

impl Drop for DeviceVirtualMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = hipMemUnmap(self.ptr, self.size);
                if !self.handle.is_null() {
                    let _ = hipMemRelease(self.handle);
                }
                let _ = hipMemAddressFree(self.ptr, self.size);
            }
        }
    }
}

fn hip_mem_allocation_props_for_device(device_id: i32) -> HipMemAllocationProp {
    HipMemAllocationProp {
        allocation_type: HIP_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_type: HIP_MEM_HANDLE_TYPE_NONE,
        location: MemLocation::Device(device_id).as_raw(),
        win32_handle_metadata: ptr::null_mut(),
        alloc_flags: HipMemAllocationFlags {
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
        },
    }
}

fn round_up_to_multiple(value: usize, granularity: usize, label: &str) -> Result<usize> {
    if granularity == 0 {
        return Err(Error::invalid_value(format!(
            "{label} granularity must be nonzero"
        )));
    }
    let remainder = value % granularity;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(granularity - remainder)
            .ok_or_else(|| Error::invalid_value(format!("{label} size overflow")))
    }
}

pub struct Stream {
    raw: HipStream,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipStreamCreate(&mut raw))?;
        }
        Ok(Self { raw })
    }

    pub const fn null() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(hipStreamSynchronize(self.raw)) }
    }

    pub fn begin_capture(&self, mode: StreamCaptureMode) -> Result<()> {
        unsafe { check(hipStreamBeginCapture(self.raw, mode.as_raw())) }
    }

    pub fn end_capture(&self) -> Result<Graph> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipStreamEndCapture(self.raw, &mut raw))?;
        }
        Ok(Graph { raw })
    }

    pub fn as_raw(&self) -> HipStream {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipStreamDestroy(self.raw);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCaptureMode {
    Global,
    ThreadLocal,
    Relaxed,
}

impl StreamCaptureMode {
    const fn as_raw(self) -> c_int {
        match self {
            Self::Global => HIP_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            Self::Relaxed => HIP_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

pub struct Graph {
    raw: HipGraph,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

/// Non-owning handle to a node inside a HIP graph.
///
/// The owning `Graph` must outlive any `GraphNode` values created from it.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphNode {
    raw: HipGraphNode,
}

unsafe impl Send for GraphNode {}
unsafe impl Sync for GraphNode {}

impl GraphNode {
    pub const fn as_raw(self) -> HipGraphNode {
        self.raw
    }

    /// Retargets a 1D memcpy node.
    ///
    /// # Safety
    ///
    /// `self` must be a memcpy node. `dst` and `src` must be valid for
    /// `bytes` bytes in the address space described by `kind` whenever an
    /// executable graph built from this node is launched.
    pub unsafe fn set_memcpy_1d(
        self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> Result<()> {
        unsafe {
            check(hipGraphMemcpyNodeSetParams1D(
                self.raw, dst, src, bytes, kind,
            ))
        }
    }

    /// Retargets a byte-pattern memset node.
    ///
    /// # Safety
    ///
    /// `self` must be a memset node. `dst` must be valid and writable for
    /// `bytes` bytes whenever an executable graph built from this node is
    /// launched.
    pub unsafe fn set_memset_1d(self, dst: *mut c_void, value: u8, bytes: usize) -> Result<()> {
        let params = hip_memset_params_1d(dst, value, bytes);
        unsafe { check(hipGraphMemsetNodeSetParams(self.raw, &params)) }
    }

    /// Retargets a kernel node.
    ///
    /// # Safety
    ///
    /// `self` must be a kernel node. `function` and all pointed-to kernel
    /// arguments must remain valid whenever an executable graph built from this
    /// node is launched.
    pub unsafe fn set_kernel_params(
        self,
        function: &Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let params = hip_kernel_node_params(function, grid, block, shared_mem_bytes, params);
        unsafe { check(hipGraphKernelNodeSetParams(self.raw, &params)) }
    }
}

/// Non-owning handle to memory allocated by a HIP graph allocation node.
///
/// Dropping this value does not free device memory. Add an explicit graph memory
/// free node after the last graph node that uses the returned pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphMemoryAllocation {
    allocation_node: GraphNode,
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for GraphMemoryAllocation {}
unsafe impl Sync for GraphMemoryAllocation {}

impl GraphMemoryAllocation {
    pub const fn allocation_node(self) -> GraphNode {
        self.allocation_node
    }

    pub const fn as_ptr<T>(self) -> *const T {
        self.ptr.cast::<T>()
    }

    pub const fn as_mut_ptr<T>(self) -> *mut T {
        self.ptr.cast::<T>()
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }

    /// Adds a graph node that frees this graph-managed allocation.
    ///
    /// # Safety
    ///
    /// `dependencies` must order this free after every node that reads or writes
    /// the allocation. The allocation must belong to `graph`.
    pub unsafe fn add_free_node(
        self,
        graph: &Graph,
        dependencies: &[GraphNode],
    ) -> Result<GraphNode> {
        unsafe { graph.add_mem_free_node(dependencies, self.ptr) }
    }
}

impl Graph {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphCreate(&mut raw, 0))?;
        }
        Ok(Self { raw })
    }

    pub fn as_raw(&self) -> HipGraph {
        self.raw
    }

    pub fn instantiate(&self) -> Result<GraphExec> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphInstantiate(
                &mut raw,
                self.raw,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ))?;
        }
        Ok(GraphExec { raw })
    }

    pub fn add_empty_node(&self, dependencies: &[GraphNode]) -> Result<GraphNode> {
        let (dependencies, dependency_count) = graph_dependency_slice(dependencies);
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphAddEmptyNode(
                &mut raw,
                self.raw,
                dependencies,
                dependency_count,
            ))?;
        }
        Ok(GraphNode { raw })
    }

    pub fn add_dependency(&self, from: GraphNode, to: GraphNode) -> Result<()> {
        let from = [from.raw];
        let to = [to.raw];
        unsafe {
            check(hipGraphAddDependencies(
                self.raw,
                from.as_ptr(),
                to.as_ptr(),
                1,
            ))
        }
    }

    pub fn add_dependencies(&self, edges: &[(GraphNode, GraphNode)]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let from = edges.iter().map(|(from, _)| from.raw).collect::<Vec<_>>();
        let to = edges.iter().map(|(_, to)| to.raw).collect::<Vec<_>>();
        unsafe {
            check(hipGraphAddDependencies(
                self.raw,
                from.as_ptr(),
                to.as_ptr(),
                edges.len(),
            ))
        }
    }

    /// Adds a 1D memcpy node to the graph.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid for `bytes` bytes in the address space
    /// described by `kind` whenever an executable graph built from this graph
    /// is launched.
    pub unsafe fn add_memcpy_node_1d(
        &self,
        dependencies: &[GraphNode],
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> Result<GraphNode> {
        let (dependencies, dependency_count) = graph_dependency_slice(dependencies);
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphAddMemcpyNode1D(
                &mut raw,
                self.raw,
                dependencies,
                dependency_count,
                dst,
                src,
                bytes,
                kind,
            ))?;
        }
        Ok(GraphNode { raw })
    }

    /// Adds a byte-pattern memset node to the graph.
    ///
    /// # Safety
    ///
    /// `dst` must be valid and writable for `bytes` bytes whenever an
    /// executable graph built from this graph is launched.
    pub unsafe fn add_memset_node_1d(
        &self,
        dependencies: &[GraphNode],
        dst: *mut c_void,
        value: u8,
        bytes: usize,
    ) -> Result<GraphNode> {
        let (dependencies, dependency_count) = graph_dependency_slice(dependencies);
        let params = hip_memset_params_1d(dst, value, bytes);
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphAddMemsetNode(
                &mut raw,
                self.raw,
                dependencies,
                dependency_count,
                &params,
            ))?;
        }
        Ok(GraphNode { raw })
    }

    /// Adds a graph memory allocation node and returns the graph-managed pointer.
    pub fn add_mem_alloc_node(
        &self,
        dependencies: &[GraphNode],
        device_id: i32,
        bytes: usize,
    ) -> Result<GraphMemoryAllocation> {
        if bytes == 0 {
            return Err(Error::invalid_value(
                "HIP graph memory allocation nodes must request nonzero bytes",
            ));
        }
        let (dependencies, dependency_count) = graph_dependency_slice(dependencies);
        let access_desc = HipMemAccessDesc {
            location: MemLocation::Device(device_id).as_raw(),
            flags: MemAccessFlags::ReadWrite.as_raw(),
        };
        let mut params = HipMemAllocNodeParams {
            pool_props: hip_mem_pool_props_for_device(device_id),
            access_descs: &access_desc,
            access_desc_count: 1,
            bytesize: bytes,
            dptr: ptr::null_mut(),
        };
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphAddMemAllocNode(
                &mut raw,
                self.raw,
                dependencies,
                dependency_count,
                &mut params,
            ))?;
        }
        if params.dptr.is_null() {
            return Err(Error::invalid_value(
                "HIP graph memory allocation node returned a null device pointer",
            ));
        }
        Ok(GraphMemoryAllocation {
            allocation_node: GraphNode { raw },
            ptr: params.dptr,
            bytes,
        })
    }

    /// Adds a graph memory free node.
    ///
    /// # Safety
    ///
    /// `ptr` must identify a graph-managed allocation, and `dependencies` must
    /// order the free after every graph node that uses `ptr`.
    pub unsafe fn add_mem_free_node(
        &self,
        dependencies: &[GraphNode],
        ptr: *mut c_void,
    ) -> Result<GraphNode> {
        if ptr.is_null() {
            return Err(Error::invalid_value(
                "HIP graph memory free node pointer is null",
            ));
        }
        let (dependencies, dependency_count) = graph_dependency_slice(dependencies);
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphAddMemFreeNode(
                &mut raw,
                self.raw,
                dependencies,
                dependency_count,
                ptr,
            ))?;
        }
        Ok(GraphNode { raw })
    }

    /// Adds a kernel launch node to the graph.
    ///
    /// # Safety
    ///
    /// `function` and all pointed-to kernel arguments must remain valid
    /// whenever an executable graph built from this graph is launched.
    pub unsafe fn add_kernel_node(
        &self,
        dependencies: &[GraphNode],
        function: &Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<GraphNode> {
        let (dependencies, dependency_count) = graph_dependency_slice(dependencies);
        let params = hip_kernel_node_params(function, grid, block, shared_mem_bytes, params);
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipGraphAddKernelNode(
                &mut raw,
                self.raw,
                dependencies,
                dependency_count,
                &params,
            ))?;
        }
        Ok(GraphNode { raw })
    }
}

fn graph_dependency_slice(dependencies: &[GraphNode]) -> (*const HipGraphNode, usize) {
    if dependencies.is_empty() {
        (ptr::null(), 0)
    } else {
        (
            dependencies.as_ptr().cast::<HipGraphNode>(),
            dependencies.len(),
        )
    }
}

fn hip_memset_params_1d(dst: *mut c_void, value: u8, bytes: usize) -> HipMemsetParams {
    HipMemsetParams {
        dst,
        element_size: 1,
        height: 1,
        pitch: bytes,
        value: value as c_uint,
        width: bytes,
    }
}

fn hip_kernel_node_params(
    function: &Function,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
    params: &mut [*mut c_void],
) -> HipKernelNodeParams {
    let kernel_params = if params.is_empty() {
        ptr::null_mut()
    } else {
        params.as_mut_ptr()
    };
    HipKernelNodeParams {
        block_dim: HipDim3::new(block),
        extra: ptr::null_mut(),
        func: function.raw.cast::<c_void>(),
        grid_dim: HipDim3::new(grid),
        kernel_params,
        shared_mem_bytes,
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipGraphDestroy(self.raw);
            }
        }
    }
}

pub struct GraphExec {
    raw: HipGraphExec,
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        unsafe { check(hipGraphLaunch(self.raw, stream.as_raw())) }
    }

    pub fn update(&self, graph: &Graph) -> Result<()> {
        let mut error_node = ptr::null_mut();
        let mut update_result = HIP_GRAPH_EXEC_UPDATE_SUCCESS;
        unsafe {
            check(hipGraphExecUpdate(
                self.raw,
                graph.raw,
                &mut error_node,
                &mut update_result,
            ))?;
        }
        if update_result == HIP_GRAPH_EXEC_UPDATE_SUCCESS {
            Ok(())
        } else {
            Err(Error::invalid_value(format!(
                "HIP graph exec update failed with result {update_result}"
            )))
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipGraphExecDestroy(self.raw);
            }
        }
    }
}

pub struct Event {
    raw: HipEvent,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipEventCreate(&mut raw))?;
        }
        Ok(Self { raw })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { check(hipEventRecord(self.raw, stream.as_raw())) }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(hipEventSynchronize(self.raw)) }
    }

    pub fn elapsed_ms_until(&self, stop: &Event) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            check(hipEventElapsedTime(&mut ms, self.raw, stop.raw))?;
        }
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipEventDestroy(self.raw);
            }
        }
    }
}

pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

/// Plain-old-data element type that can be safely represented by zeroed host
/// memory and copied to/from device memory as raw bytes.
///
/// # Safety
///
/// Implementors must guarantee that every bit pattern produced by zeroed memory
/// and GPU byte copies is a valid value of the type, and that the type has no
/// destructor, references, or hidden ownership/lifetime invariants.
pub unsafe trait DevicePod: Copy + Send + Sync + 'static {}

macro_rules! impl_device_pod {
    ($($ty:ty),* $(,)?) => {
        $(
            unsafe impl DevicePod for $ty {}
        )*
    };
}

impl_device_pod!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64,
);

unsafe impl<T: DevicePod, const N: usize> DevicePod for [T; N] {}

pub struct ManagedBuffer<T> {
    ptr: *mut T,
    len: usize,
    kind: ManagedMemoryKind,
}

unsafe impl<T: Send> Send for ManagedBuffer<T> {}
unsafe impl<T: Sync> Sync for ManagedBuffer<T> {}

impl<T: DevicePod> ManagedBuffer<T> {
    pub fn new_zeroed(len: usize) -> Result<Self> {
        Self::new_zeroed_with_kind(len, ManagedMemoryKind::FineGrain)
    }

    pub fn new_zeroed_coarse_grained(len: usize) -> Result<Self> {
        Self::new_zeroed_with_kind(len, ManagedMemoryKind::CoarseGrain)
    }

    fn new_zeroed_with_kind(len: usize, kind: ManagedMemoryKind) -> Result<Self> {
        let bytes = checked_allocation_bytes::<T>(len, "managed")?;
        let coarse_grain_device = if matches!(kind, ManagedMemoryKind::CoarseGrain) {
            Some(current_device()?)
        } else {
            None
        };
        if bytes == 0 {
            return Ok(Self {
                ptr: NonNull::<T>::dangling().as_ptr(),
                len,
                kind,
            });
        }

        let mut ptr = ptr::null_mut();
        unsafe {
            if let Err(err) = check(hipMallocManaged(&mut ptr, bytes, HIP_MEM_ATTACH_GLOBAL)) {
                if !ptr.is_null() {
                    let _ = hipFree(ptr);
                }
                return Err(err);
            }
            if matches!(kind, ManagedMemoryKind::CoarseGrain)
                && let Err(err) = check(hipMemAdvise(
                    ptr.cast::<c_void>(),
                    bytes,
                    HIP_MEM_ADVISE_SET_COARSE_GRAIN,
                    coarse_grain_device.expect("coarse-grain device must be queried"),
                ))
            {
                let _ = hipFree(ptr);
                return Err(err);
            }
            ptr::write_bytes(ptr.cast::<u8>(), 0, bytes);
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len,
            kind,
        })
    }

    pub fn kind(&self) -> ManagedMemoryKind {
        self.kind
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: DevicePod> ManagedBuffer<T> {
    pub fn from_slice(input: &[T]) -> Result<Self> {
        let mut buffer = Self::new_zeroed(input.len())?;
        buffer.as_mut_slice().copy_from_slice(input);
        Ok(buffer)
    }
}

impl<T> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        if self.len != 0 && !self.ptr.is_null() {
            unsafe {
                let _ = hipFree(self.ptr.cast::<c_void>());
            }
        }
    }
}

impl<T> DeviceBuffer<T> {
    pub fn new(len: usize) -> Result<Self> {
        let bytes = checked_allocation_bytes::<T>(len, "device")?;
        if bytes == 0 {
            return Ok(Self {
                ptr: NonNull::<T>::dangling().as_ptr(),
                len,
            });
        }

        let mut ptr = ptr::null_mut();
        unsafe {
            if let Err(err) = check(hipMalloc(&mut ptr, bytes)) {
                if !ptr.is_null() {
                    let _ = hipFree(ptr);
                }
                return Err(err);
            }
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len,
        })
    }

    pub fn new_fine_grained(len: usize) -> Result<Self> {
        let bytes = checked_allocation_bytes::<T>(len, "fine-grained device")?;
        if bytes == 0 {
            return Ok(Self {
                ptr: NonNull::<T>::dangling().as_ptr(),
                len,
            });
        }

        let mut ptr = ptr::null_mut();
        unsafe {
            if let Err(err) = check(hipExtMallocWithFlags(
                &mut ptr,
                bytes,
                HIP_DEVICE_MALLOC_FINEGRAINED,
            )) {
                if !ptr.is_null() {
                    let _ = hipFree(ptr);
                }
                return Err(err);
            }
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len,
        })
    }

    /// Enqueues a stream-ordered device allocation.
    ///
    /// # Safety
    ///
    /// The returned allocation must only be used by work ordered after this
    /// allocation on `stream` until the stream reaches the allocation. The
    /// stream and any memory pool selected by HIP for async allocation must
    /// remain valid until that point.
    pub unsafe fn new_async(stream: &Stream, len: usize) -> Result<Self> {
        let bytes = checked_allocation_bytes::<T>(len, "device")?;
        if bytes == 0 {
            return Ok(Self {
                ptr: NonNull::<T>::dangling().as_ptr(),
                len,
            });
        }

        let mut ptr = ptr::null_mut();
        unsafe {
            if let Err(err) = check(hipMallocAsync(&mut ptr, bytes, stream.as_raw())) {
                if !ptr.is_null() {
                    let _ = hipFreeAsync(ptr, stream.as_raw());
                }
                return Err(err);
            }
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len,
        })
    }

    /// Enqueues a stream-ordered device allocation from `pool`.
    ///
    /// # Safety
    ///
    /// The returned allocation must only be used by work ordered after this
    /// allocation on `stream` until the stream reaches the allocation. `stream`
    /// and `pool` must remain valid until that point.
    pub unsafe fn new_from_pool_async(stream: &Stream, pool: MemPool, len: usize) -> Result<Self> {
        let bytes = checked_allocation_bytes::<T>(len, "pooled device")?;
        if bytes == 0 {
            return Ok(Self {
                ptr: NonNull::<T>::dangling().as_ptr(),
                len,
            });
        }

        let mut ptr = ptr::null_mut();
        unsafe {
            if let Err(err) = check(hipMallocFromPoolAsync(
                &mut ptr,
                bytes,
                pool.as_raw(),
                stream.as_raw(),
            )) {
                if !ptr.is_null() {
                    let _ = hipFreeAsync(ptr, stream.as_raw());
                }
                return Err(err);
            }
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len,
        })
    }

    pub fn copy_from_host(&self, input: &[T]) -> Result<()> {
        validate_slice_len("host-to-device source", input.len(), self.len)?;
        let bytes = std::mem::size_of_val(input);
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemcpy(
                self.ptr.cast::<c_void>(),
                input.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ))
        }
    }

    /// Enqueues a host-to-device copy from borrowed host memory.
    ///
    /// # Safety
    ///
    /// `self`, `stream`, and `input` must remain valid until `stream` reaches
    /// this copy. `input` must not be mutated for that duration.
    pub unsafe fn copy_from_host_async(&self, stream: &Stream, input: &[T]) -> Result<()> {
        validate_slice_len("async host-to-device source", input.len(), self.len)?;
        let bytes = std::mem::size_of_val(input);
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemcpyAsync(
                self.ptr.cast::<c_void>(),
                input.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
                stream.as_raw(),
            ))
        }
    }

    pub fn copy_to_host(&self, output: &mut [T]) -> Result<()> {
        validate_slice_len("device-to-host destination", output.len(), self.len)?;
        let bytes = std::mem::size_of_val(output);
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemcpy(
                output.as_mut_ptr().cast::<c_void>(),
                self.ptr.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_HOST,
            ))
        }
    }

    /// Enqueues a device-to-host copy into borrowed host memory.
    ///
    /// # Safety
    ///
    /// `self`, `stream`, and `output` must remain valid until `stream` reaches
    /// this copy. `output` must not be read, written, or aliased for that
    /// duration.
    pub unsafe fn copy_to_host_async(&self, stream: &Stream, output: &mut [T]) -> Result<()> {
        validate_slice_len("async device-to-host destination", output.len(), self.len)?;
        let bytes = std::mem::size_of_val(output);
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemcpyAsync(
                output.as_mut_ptr().cast::<c_void>(),
                self.ptr.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_HOST,
                stream.as_raw(),
            ))
        }
    }

    /// Copies another device buffer into this buffer without staging through host memory.
    pub fn copy_from_device(&self, input: &DeviceBuffer<T>) -> Result<()> {
        validate_slice_len("device-to-device source", input.len, self.len)?;
        let bytes = checked_allocation_bytes::<T>(self.len, "device-to-device copy")?;
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemcpy(
                self.ptr.cast::<c_void>(),
                input.ptr.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            ))
        }
    }

    /// Enqueues a device-to-device copy into this buffer.
    ///
    /// The source and destination buffers must stay alive until `stream`
    /// reaches this copy.
    /// Enqueues a device-to-device copy into this buffer.
    ///
    /// # Safety
    ///
    /// The source buffer, destination buffer, and `stream` must stay alive
    /// until `stream` reaches this copy. The source and destination must not
    /// alias.
    pub unsafe fn copy_from_device_async(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<T>,
    ) -> Result<()> {
        validate_slice_len("async device-to-device source", input.len, self.len)?;
        let bytes = checked_allocation_bytes::<T>(self.len, "async device-to-device copy")?;
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemcpyAsync(
                self.ptr.cast::<c_void>(),
                input.ptr.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream.as_raw(),
            ))
        }
    }

    /// Copies this buffer into another device buffer without staging through host memory.
    pub fn copy_to_device(&self, output: &DeviceBuffer<T>) -> Result<()> {
        output.copy_from_device(self)
    }

    /// Enqueues a copy from this buffer into another device buffer.
    ///
    /// # Safety
    ///
    /// The source buffer, destination buffer, and `stream` must stay alive
    /// until `stream` reaches this copy. The source and destination must not
    /// alias.
    pub unsafe fn copy_to_device_async(
        &self,
        stream: &Stream,
        output: &DeviceBuffer<T>,
    ) -> Result<()> {
        unsafe { output.copy_from_device_async(stream, self) }
    }

    /// Fills the device allocation with a byte pattern.
    ///
    /// Prefer `set_zero` for typed zero initialization; nonzero byte patterns
    /// are intended for byte-addressed buffers and debugging sentinels.
    pub fn memset(&self, value: u8) -> Result<()> {
        let bytes = checked_allocation_bytes::<T>(self.len, "device memset")?;
        if bytes == 0 {
            return Ok(());
        }
        unsafe { check(hipMemset(self.ptr.cast::<c_void>(), value as c_int, bytes)) }
    }

    /// Enqueues a byte-pattern fill of the device allocation.
    ///
    /// # Safety
    ///
    /// `self` and `stream` must remain valid until `stream` reaches this
    /// memset.
    pub unsafe fn memset_async(&self, stream: &Stream, value: u8) -> Result<()> {
        let bytes = checked_allocation_bytes::<T>(self.len, "async device memset")?;
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            check(hipMemsetAsync(
                self.ptr.cast::<c_void>(),
                value as c_int,
                bytes,
                stream.as_raw(),
            ))
        }
    }

    /// Fills the device allocation with zero bytes.
    pub fn set_zero(&self) -> Result<()> {
        self.memset(0)
    }

    /// Enqueues a zero-byte fill of the device allocation.
    ///
    /// # Safety
    ///
    /// `self` and `stream` must remain valid until `stream` reaches this
    /// memset.
    pub unsafe fn set_zero_async(&self, stream: &Stream) -> Result<()> {
        unsafe { self.memset_async(stream, 0) }
    }

    /// Copies this buffer into another device-visible pointer.
    ///
    /// This is intended for interop destinations such as graphics resources
    /// that HIP maps to a raw device pointer.
    ///
    /// # Safety
    ///
    /// `output` must point to at least `len` valid `T` elements in device
    /// address space, must be writable for the duration of the copy, and must
    /// not alias this buffer.
    pub unsafe fn copy_to_device_ptr(&self, output: *mut T, len: usize) -> Result<()> {
        validate_slice_len("device-to-device destination", len, self.len)?;
        let bytes = checked_allocation_bytes::<T>(len, "device-to-device copy")?;
        if bytes == 0 {
            return Ok(());
        }
        if output.is_null() {
            return Err(Error::invalid_value(
                "device-to-device destination pointer is null",
            ));
        }
        unsafe {
            check(hipMemcpy(
                output.cast::<c_void>(),
                self.ptr.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            ))
        }
    }

    /// Copies from another device-visible pointer into this buffer.
    ///
    /// This is intended for interop sources such as HIP virtual-memory
    /// reservations or graphics resources that expose raw device pointers.
    ///
    /// # Safety
    ///
    /// `input` must point to at least `len` valid `T` elements in device
    /// address space, must be readable for the duration of the copy, and must
    /// not alias this buffer.
    pub unsafe fn copy_from_device_ptr(&self, input: *const T, len: usize) -> Result<()> {
        validate_slice_len("device-to-device source", len, self.len)?;
        let bytes = checked_allocation_bytes::<T>(len, "device-to-device copy")?;
        if bytes == 0 {
            return Ok(());
        }
        if input.is_null() {
            return Err(Error::invalid_value(
                "device-to-device source pointer is null",
            ));
        }
        unsafe {
            check(hipMemcpy(
                self.ptr.cast::<c_void>(),
                input.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            ))
        }
    }

    /// Enqueues a copy from this buffer into another device-visible pointer.
    ///
    /// # Safety
    ///
    /// `output` must point to at least `len` valid `T` elements in device
    /// address space, must be writable until `stream` reaches this copy, and
    /// must not alias this buffer.
    pub unsafe fn copy_to_device_ptr_async(
        &self,
        stream: &Stream,
        output: *mut T,
        len: usize,
    ) -> Result<()> {
        validate_slice_len("async device-to-device destination", len, self.len)?;
        let bytes = checked_allocation_bytes::<T>(len, "async device-to-device copy")?;
        if bytes == 0 {
            return Ok(());
        }
        if output.is_null() {
            return Err(Error::invalid_value(
                "async device-to-device destination pointer is null",
            ));
        }
        unsafe {
            check(hipMemcpyAsync(
                output.cast::<c_void>(),
                self.ptr.cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream.as_raw(),
            ))
        }
    }

    /// Enqueues a host-to-device copy from pinned host memory.
    ///
    /// # Safety
    ///
    /// The input pinned buffer must not be dropped, freed, mutated, or aliased
    /// until the stream reaches this copy.
    pub unsafe fn copy_from_pinned_host_async(
        &self,
        stream: &Stream,
        input: &PinnedHostBuffer<T>,
    ) -> Result<()>
    where
        T: DevicePod,
    {
        unsafe { self.copy_from_host_async(stream, input.as_slice()) }
    }

    /// Enqueues a device-to-host copy into pinned host memory.
    ///
    /// # Safety
    ///
    /// The output pinned buffer must not be dropped, freed, read, or aliased
    /// until the stream reaches this copy.
    pub unsafe fn copy_to_pinned_host_async(
        &self,
        stream: &Stream,
        output: &mut PinnedHostBuffer<T>,
    ) -> Result<()>
    where
        T: DevicePod,
    {
        unsafe { self.copy_to_host_async(stream, output.as_mut_slice()) }
    }

    pub fn copy_from_pinned_host(&self, input: &PinnedHostBuffer<T>) -> Result<()>
    where
        T: DevicePod,
    {
        self.copy_from_host(input.as_slice())
    }

    pub fn copy_to_pinned_host(&self, output: &mut PinnedHostBuffer<T>) -> Result<()>
    where
        T: DevicePod,
    {
        self.copy_to_host(output.as_mut_slice())
    }

    pub unsafe fn free_async(mut self, stream: &Stream) -> Result<()> {
        if self.len != 0 && !self.ptr.is_null() {
            let ptr = self.ptr.cast::<c_void>();
            self.ptr = ptr::null_mut();
            unsafe {
                check(hipFreeAsync(ptr, stream.as_raw()))?;
            }
        }
        Ok(())
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct PinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedHostBuffer<T> {}

impl<T: DevicePod> PinnedHostBuffer<T> {
    pub fn new_zeroed(len: usize) -> Result<Self> {
        Self::new_zeroed_with_flags(len, 0)
    }

    pub fn new_zeroed_mapped_coherent(len: usize) -> Result<Self> {
        Self::new_zeroed_with_flags(len, HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_COHERENT)
    }

    fn new_zeroed_with_flags(len: usize, flags: c_uint) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
                len,
            });
        }

        let mut ptr = ptr::null_mut();
        let bytes = checked_allocation_bytes::<T>(len, "pinned host")?;
        unsafe {
            check(hipHostMalloc(&mut ptr, bytes, flags))?;
            ptr::write_bytes(ptr.cast::<u8>(), 0, bytes);
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len,
        })
    }

    pub fn device_ptr(&self) -> Result<*mut T> {
        if self.len == 0 {
            return Ok(std::ptr::NonNull::<T>::dangling().as_ptr());
        }
        let mut ptr = ptr::null_mut();
        unsafe {
            check(hipHostGetDevicePointer(
                &mut ptr,
                self.ptr.cast::<c_void>(),
                0,
            ))?;
        }
        Ok(ptr.cast::<T>())
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: DevicePod> PinnedHostBuffer<T> {
    pub fn from_slice(input: &[T]) -> Result<Self> {
        let mut buffer = Self::new_zeroed(input.len())?;
        buffer.as_mut_slice().copy_from_slice(input);
        Ok(buffer)
    }
}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.len != 0 && !self.ptr.is_null() {
            unsafe {
                let _ = hipHostFree(self.ptr.cast::<c_void>());
            }
        }
    }
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn from_slice(input: &[T]) -> Result<Self> {
        let buffer = Self::new(input.len())?;
        buffer.copy_from_host(input)?;
        Ok(buffer)
    }
}

impl<T: Copy + Default> DeviceBuffer<T> {
    pub fn copy_to_vec(&self) -> Result<Vec<T>> {
        let mut output = vec![T::default(); self.len];
        self.copy_to_host(&mut output)?;
        Ok(output)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.len != 0 && !self.ptr.is_null() {
            unsafe {
                let _ = hipFree(self.ptr.cast::<c_void>());
            }
        }
    }
}

pub struct Module {
    raw: HipModule,
}

// HIP module handles are immutable after load and launches bind the target
// device before use through `ExecutionContext`.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub fn from_code_object(bytes: &[u8]) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipModuleLoadData(&mut raw, bytes.as_ptr().cast::<c_void>()))?;
        }
        Ok(Self { raw })
    }

    pub fn function(&self, name: &CStr) -> Result<Function> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hipModuleGetFunction(&mut raw, self.raw, name.as_ptr()))?;
        }
        Ok(Function { raw })
    }

    pub fn global<T>(&self, name: &CStr) -> Result<Global<T>> {
        let mut ptr = ptr::null_mut();
        let mut bytes = 0usize;
        unsafe {
            check(hipModuleGetGlobal(
                &mut ptr,
                &mut bytes,
                self.raw,
                name.as_ptr(),
            ))?;
        }
        Global::new(ptr.cast::<T>(), bytes, name.to_string_lossy().into_owned())
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipModuleUnload(self.raw);
            }
        }
    }
}

pub struct Function {
    raw: HipFunction,
}

// HIP function handles are immutable module entry-point references. Launches
// bind device/context at the caller layer before use.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub fn occupancy_max_potential_block_size(
        &self,
        dynamic_shared_mem_per_block: u32,
        block_size_limit: u32,
    ) -> Result<(u32, u32)> {
        let mut grid_size = 0;
        let mut block_size = 0;
        let block_size_limit = c_int_from_u32("occupancy block-size limit", block_size_limit)?;
        unsafe {
            check(hipModuleOccupancyMaxPotentialBlockSize(
                &mut grid_size,
                &mut block_size,
                self.raw,
                dynamic_shared_mem_per_block as usize,
                block_size_limit,
            ))?;
        }
        Ok((
            u32_from_hip_int("occupancy grid size", grid_size)?,
            u32_from_hip_int("occupancy block size", block_size)?,
        ))
    }

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_shared_mem_per_block: u32,
    ) -> Result<u32> {
        let mut blocks_per_multiprocessor = 0;
        let block_size = c_int_from_u32("occupancy block size", block_size)?;
        unsafe {
            check(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut blocks_per_multiprocessor,
                self.raw,
                block_size,
                dynamic_shared_mem_per_block as usize,
            ))?;
        }
        u32_from_hip_int(
            "occupancy active blocks per multiprocessor",
            blocks_per_multiprocessor,
        )
    }

    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe { self.launch_on_stream(grid, block, shared_mem_bytes, ptr::null_mut(), params) }
    }

    pub unsafe fn launch_on_stream(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: HipStream,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        check(unsafe {
            hipModuleLaunchKernel(
                self.raw,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        })
    }

    pub unsafe fn launch_cooperative_on_stream(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: HipStream,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        check(unsafe {
            hipModuleLaunchCooperativeKernel(
                self.raw,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream,
                params.as_mut_ptr(),
            )
        })
    }
}

pub struct Global<T> {
    ptr: *mut T,
    bytes: usize,
    name: String,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for Global<T> {}
unsafe impl<T: Sync> Sync for Global<T> {}

impl<T> Global<T> {
    fn new(ptr: *mut T, bytes: usize, name: String) -> Result<Self> {
        let element_size = size_of::<T>();
        if element_size == 0 {
            return Err(Error::invalid_value(format!(
                "module global `{name}` cannot be viewed as a zero-sized Rust type"
            )));
        }
        if bytes % element_size != 0 {
            return Err(Error::invalid_value(format!(
                "module global `{name}` has {bytes} bytes, which is not a multiple of {}",
                element_size
            )));
        }
        Ok(Self {
            ptr,
            bytes,
            name,
            _marker: PhantomData,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes / size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn copy_from_slice(&self, input: &[T]) -> Result<()> {
        let input_bytes = std::mem::size_of_val(input);
        if input_bytes != self.bytes {
            return Err(Error::invalid_value(format!(
                "module global `{}` copy size mismatch: got {input_bytes} bytes, expected {}",
                self.name, self.bytes
            )));
        }
        unsafe {
            check(hipMemcpy(
                self.ptr.cast::<c_void>(),
                input.as_ptr().cast::<c_void>(),
                input_bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ))
        }
    }

    pub fn copy_to_slice(&self, output: &mut [T]) -> Result<()> {
        let output_bytes = std::mem::size_of_val(output);
        if output_bytes != self.bytes {
            return Err(Error::invalid_value(format!(
                "module global `{}` copy size mismatch: got {output_bytes} bytes, expected {}",
                self.name, self.bytes
            )));
        }
        unsafe {
            check(hipMemcpy(
                output.as_mut_ptr().cast::<c_void>(),
                self.ptr.cast::<c_void>(),
                output_bytes,
                HIP_MEMCPY_DEVICE_TO_HOST,
            ))
        }
    }
}

impl<T: Copy> Global<T> {
    pub fn set(&self, value: T) -> Result<()> {
        self.copy_from_slice(std::slice::from_ref(&value))
    }
}

impl<T: Copy + Default> Global<T> {
    pub fn copy_to_vec(&self) -> Result<Vec<T>> {
        let mut output = vec![T::default(); self.len()];
        self.copy_to_slice(&mut output)?;
        Ok(output)
    }
}

pub fn synchronize() -> Result<()> {
    unsafe { check(hipDeviceSynchronize()) }
}

pub fn device_count() -> Result<i32> {
    let mut count = 0;
    unsafe {
        check(hipGetDeviceCount(&mut count))?;
    }
    Ok(count)
}

pub fn current_device() -> Result<i32> {
    let mut device_id = 0;
    unsafe {
        check(hipGetDevice(&mut device_id))?;
    }
    Ok(device_id)
}

pub fn set_device(device_id: i32) -> Result<()> {
    unsafe { check(hipSetDevice(device_id)) }
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceBuffer, DeviceVirtualMemory, Global, Graph, HIP_ERROR_NOT_SUPPORTED,
        HIP_MEMCPY_DEVICE_TO_DEVICE, ManagedBuffer, MemAccessFlags, MemLocation, MemPool,
        PinnedHostBuffer, Stream, current_device,
    };
    use std::ffi::c_void;

    fn is_not_supported(err: &super::Error) -> bool {
        err.code() == Some(HIP_ERROR_NOT_SUPPORTED)
    }

    #[test]
    fn device_allocation_size_overflow_is_error() {
        let Err(err) = DeviceBuffer::<u16>::new(usize::MAX) else {
            panic!("overflow should fail");
        };
        assert!(err.to_string().contains("allocation size overflow"));
    }

    #[test]
    fn pinned_allocation_size_overflow_is_error() {
        let Err(err) = PinnedHostBuffer::<u16>::new_zeroed(usize::MAX) else {
            panic!("overflow should fail");
        };
        assert!(err.to_string().contains("allocation size overflow"));
    }

    #[test]
    fn managed_allocation_size_overflow_is_error() {
        let Err(err) = ManagedBuffer::<u16>::new_zeroed(usize::MAX) else {
            panic!("overflow should fail");
        };
        assert!(err.to_string().contains("allocation size overflow"));
    }

    #[test]
    fn copy_length_mismatch_is_error() {
        let buffer = DeviceBuffer::<u8>::new(4).expect("small allocation should work");
        let err = buffer
            .copy_from_host(&[1, 2])
            .expect_err("short host copy should fail");
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn device_copy_length_mismatch_is_error() {
        let output = DeviceBuffer::<u8>::new(4).expect("small allocation should work");
        let input = DeviceBuffer::<u8>::new(2).expect("small allocation should work");
        let err = output
            .copy_from_device(&input)
            .expect_err("short device copy should fail");
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn device_to_device_copy_round_trips() {
        let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4]).expect("input upload should work");
        let output = DeviceBuffer::<u32>::new(4).expect("output allocation should work");
        output
            .copy_from_device(&input)
            .expect("device-to-device copy should work");
        assert_eq!(
            output.copy_to_vec().expect("download should work"),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn device_set_zero_round_trips() {
        let buffer = DeviceBuffer::from_slice(&[7u32, 8, 9]).expect("upload should work");
        buffer.set_zero().expect("device memset should work");
        assert_eq!(
            buffer.copy_to_vec().expect("download should work"),
            [0, 0, 0]
        );
    }

    #[test]
    fn explicit_graph_memset_and_copy_nodes_round_trip() {
        let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4]).expect("input upload should work");
        let scratch = DeviceBuffer::<u32>::new(4).expect("scratch allocation should work");
        let output = DeviceBuffer::<u32>::new(4).expect("output allocation should work");
        let stream = Stream::new().expect("stream should be created");
        let graph = Graph::new().expect("graph should be created");
        let bytes = std::mem::size_of::<u32>() * input.len();

        let gate = graph.add_empty_node(&[]).expect("empty node should work");
        let zero_output = unsafe {
            graph.add_memset_node_1d(&[gate], output.as_mut_ptr().cast::<c_void>(), 0, bytes)
        }
        .expect("memset node should work");
        let copy_to_scratch = unsafe {
            graph.add_memcpy_node_1d(
                &[zero_output],
                scratch.as_mut_ptr().cast::<c_void>(),
                input.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        }
        .expect("device-to-device memcpy node should work");
        let copy_to_output = unsafe {
            graph.add_memcpy_node_1d(
                &[],
                output.as_mut_ptr().cast::<c_void>(),
                scratch.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        }
        .expect("dependent memcpy node should work");
        graph
            .add_dependency(copy_to_scratch, copy_to_output)
            .expect("explicit dependency should work");

        let exec = graph.instantiate().expect("graph instantiate should work");
        exec.launch(&stream).expect("graph launch should work");
        stream.synchronize().expect("graph stream should finish");
        assert_eq!(
            output.copy_to_vec().expect("download should work"),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn explicit_graph_mem_alloc_and_free_nodes_round_trip_if_supported() {
        let device_id = current_device().expect("current device should be visible");
        let output = DeviceBuffer::<u8>::new(16).expect("output allocation should work");
        let stream = Stream::new().expect("stream should be created");
        let graph = Graph::new().expect("graph should be created");
        let bytes = output.len();

        let allocation = match graph.add_mem_alloc_node(&[], device_id, bytes) {
            Ok(allocation) => allocation,
            Err(err) if is_not_supported(&err) => return,
            Err(err) => {
                panic!("graph memory allocation nodes should work or be unsupported: {err}")
            }
        };
        assert_eq!(allocation.bytes(), bytes);
        assert!(!allocation.as_mut_ptr::<u8>().is_null());

        let fill = unsafe {
            graph.add_memset_node_1d(
                &[allocation.allocation_node()],
                allocation.as_mut_ptr::<u8>().cast::<c_void>(),
                0x5a,
                bytes,
            )
        }
        .expect("graph memset on graph allocation should work");
        let copy_to_output = unsafe {
            graph.add_memcpy_node_1d(
                &[fill],
                output.as_mut_ptr().cast::<c_void>(),
                allocation.as_ptr::<u8>().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        }
        .expect("graph copy from graph allocation should work");
        unsafe {
            allocation
                .add_free_node(&graph, &[copy_to_output])
                .expect("graph memory free node should work");
        }

        let exec = match graph.instantiate() {
            Ok(exec) => exec,
            Err(err) if is_not_supported(&err) => return,
            Err(err) => panic!("graph memory allocation instantiate should work: {err}"),
        };
        exec.launch(&stream)
            .expect("graph memory allocation launch should work");
        stream
            .synchronize()
            .expect("graph memory allocation stream should finish");
        assert_eq!(
            output.copy_to_vec().expect("download should work"),
            [0x5a; 16]
        );
    }

    #[test]
    fn graph_exec_update_retargets_memcpy_node() {
        let input_a = DeviceBuffer::from_slice(&[5u32, 6, 7, 8]).expect("upload A should work");
        let input_b = DeviceBuffer::from_slice(&[9u32, 10, 11, 12]).expect("upload B should work");
        let output = DeviceBuffer::<u32>::new(4).expect("output allocation should work");
        let stream = Stream::new().expect("stream should be created");
        let bytes = std::mem::size_of::<u32>() * input_a.len();

        let graph_a = Graph::new().expect("graph A should be created");
        unsafe {
            graph_a.add_memcpy_node_1d(
                &[],
                output.as_mut_ptr().cast::<c_void>(),
                input_a.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        }
        .expect("graph A memcpy node should work");
        let exec = graph_a
            .instantiate()
            .expect("graph A instantiate should work");
        exec.launch(&stream).expect("graph A launch should work");
        stream.synchronize().expect("graph A stream should finish");
        assert_eq!(
            output.copy_to_vec().expect("download A should work"),
            [5, 6, 7, 8]
        );

        let graph_b = Graph::new().expect("graph B should be created");
        unsafe {
            graph_b.add_memcpy_node_1d(
                &[],
                output.as_mut_ptr().cast::<c_void>(),
                input_b.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        }
        .expect("graph B memcpy node should work");
        exec.update(&graph_b).expect("graph update should work");
        exec.launch(&stream)
            .expect("updated graph launch should work");
        stream
            .synchronize()
            .expect("updated graph stream should finish");
        assert_eq!(
            output.copy_to_vec().expect("download B should work"),
            [9, 10, 11, 12]
        );
    }

    #[test]
    fn owned_memory_pool_access_flags_round_trip_if_supported() {
        let device_id = current_device().expect("current device should be visible");
        let pool = match MemPool::create_for_device(device_id) {
            Ok(pool) => pool,
            Err(err) if is_not_supported(&err) => return,
            Err(err) => panic!("custom pool creation should work or be unsupported: {err}"),
        };
        let pool = pool.as_pool();
        pool.set_release_threshold(4096)
            .expect("custom pool release threshold should be set");
        assert_eq!(
            pool.release_threshold()
                .expect("custom pool release threshold should be queried"),
            4096
        );
        pool.set_access(MemLocation::Device(device_id), MemAccessFlags::ReadWrite)
            .expect("custom pool device access should be set");
        assert_eq!(
            pool.access(MemLocation::Device(device_id))
                .expect("custom pool device access should be queried"),
            MemAccessFlags::ReadWrite
        );
    }

    #[test]
    fn virtual_memory_round_trips_through_device_copies_if_supported() {
        let device_id = current_device().expect("current device should be visible");
        let bytes = std::mem::size_of::<u32>() * 4;
        let memory = match DeviceVirtualMemory::new_for_device(device_id, bytes) {
            Ok(memory) => memory,
            Err(err) if is_not_supported(&err) => return,
            Err(err) => panic!("device virtual memory should work or be unsupported: {err}"),
        };
        assert!(memory.size() >= bytes);
        assert_eq!(memory.requested_size(), bytes);
        assert_eq!(
            memory
                .access(MemLocation::Device(device_id))
                .expect("virtual memory access should be queried"),
            MemAccessFlags::ReadWrite
        );

        let input = DeviceBuffer::from_slice(&[13u32, 14, 15, 16]).expect("upload should work");
        unsafe {
            input
                .copy_to_device_ptr(memory.as_mut_ptr::<u32>(), 4)
                .expect("copy into virtual memory should work");
        }
        let output = DeviceBuffer::<u32>::new(4).expect("output allocation should work");
        unsafe {
            output
                .copy_from_device_ptr(memory.as_ptr::<u32>(), 4)
                .expect("copy from virtual memory should work");
        }
        assert_eq!(
            output.copy_to_vec().expect("download should work"),
            [13, 14, 15, 16]
        );
    }

    #[test]
    fn zero_length_device_buffer_does_not_allocate() {
        let buffer = DeviceBuffer::<u8>::new(0).expect("zero-sized allocation should work");
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        buffer
            .set_zero()
            .expect("zero-length memset should be a no-op");
    }

    #[test]
    fn global_size_mismatch_is_error_before_copy() {
        let global = Global::<u32>::new(std::ptr::null_mut(), 8, "coeffs".into())
            .expect("u32 view of eight bytes should be valid");
        let err = global
            .copy_from_slice(&[1])
            .expect_err("short global copy should fail");
        assert!(err.to_string().contains("copy size mismatch"));
    }
}
