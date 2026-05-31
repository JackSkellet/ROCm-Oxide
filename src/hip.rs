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
pub type HipMemPool = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
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
    fn hipGraphLaunch(graph_exec: HipGraphExec, stream: HipStream) -> HipError;
    fn hipGraphDestroy(graph: HipGraph) -> HipError;
    fn hipGraphExecDestroy(graph_exec: HipGraphExec) -> HipError;
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

#[derive(Clone, Copy)]
pub struct MemPool {
    raw: HipMemPool,
}

unsafe impl Send for MemPool {}
unsafe impl Sync for MemPool {}

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

impl Graph {
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

pub struct ManagedBuffer<T> {
    ptr: *mut T,
    len: usize,
    kind: ManagedMemoryKind,
}

unsafe impl<T: Send> Send for ManagedBuffer<T> {}
unsafe impl<T: Sync> Sync for ManagedBuffer<T> {}

impl<T> ManagedBuffer<T> {
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

impl<T: Copy> ManagedBuffer<T> {
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

    pub fn new_async(stream: &Stream, len: usize) -> Result<Self> {
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

    pub fn new_from_pool_async(stream: &Stream, pool: MemPool, len: usize) -> Result<Self> {
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

    pub fn copy_from_host_async(&self, stream: &Stream, input: &[T]) -> Result<()> {
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

    pub fn copy_to_host_async(&self, stream: &Stream, output: &mut [T]) -> Result<()> {
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
    ) -> Result<()> {
        self.copy_from_host_async(stream, input.as_slice())
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
    ) -> Result<()> {
        self.copy_to_host_async(stream, output.as_mut_slice())
    }

    pub fn copy_from_pinned_host(&self, input: &PinnedHostBuffer<T>) -> Result<()> {
        self.copy_from_host(input.as_slice())
    }

    pub fn copy_to_pinned_host(&self, output: &mut PinnedHostBuffer<T>) -> Result<()> {
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

impl<T> PinnedHostBuffer<T> {
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

impl<T: Copy> PinnedHostBuffer<T> {
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
    use super::{DeviceBuffer, Global, ManagedBuffer, PinnedHostBuffer};

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
    fn zero_length_device_buffer_does_not_allocate() {
        let buffer = DeviceBuffer::<u8>::new(0).expect("zero-sized allocation should work");
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
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
