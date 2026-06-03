use crate::{hip, hiprtc};
use std::ffi::{CStr, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub enum Error {
    Hip(hip::Error),
    Hiprtc(hiprtc::Error),
    Io(std::io::Error),
    InvalidLaunch(String),
    Async(String),
    Library(String),
    NoDevice,
    MissingArchitecture,
}

impl From<hip::Error> for Error {
    fn from(value: hip::Error) -> Self {
        Self::Hip(value)
    }
}

impl From<hiprtc::Error> for Error {
    fn from(value: hiprtc::Error) -> Self {
        Self::Hiprtc(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(err) => write!(f, "{err}"),
            Self::Hiprtc(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
            Self::InvalidLaunch(message) => write!(f, "invalid kernel launch: {message}"),
            Self::Async(message) => write!(f, "async device operation failed: {message}"),
            Self::Library(message) => write!(f, "ROCm library interop failed: {message}"),
            Self::NoDevice => write!(f, "no HIP devices are visible"),
            Self::MissingArchitecture => write!(
                f,
                "could not detect a ROCm GPU architecture; set ROCM_OXIDE_ARCH=gfx..."
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Device {
    ordinal: i32,
    arch: String,
    limits: DeviceLimits,
}

impl Device {
    pub fn count() -> Result<i32> {
        Ok(hip::device_count()?)
    }

    pub fn first() -> Result<Self> {
        Self::at(0)
    }

    pub fn at(ordinal: i32) -> Result<Self> {
        let count = hip::device_count()?;
        if count == 0 {
            return Err(Error::NoDevice);
        }
        if ordinal < 0 || ordinal >= count {
            return Err(Error::InvalidLaunch(format!(
                "device ordinal {ordinal} is outside visible HIP device range 0..{count}"
            )));
        }

        hip::set_device(ordinal)?;
        let arch = detect_arch().ok_or(Error::MissingArchitecture)?;
        let limits = DeviceLimits::query(ordinal)?;
        Ok(Self {
            ordinal,
            arch,
            limits,
        })
    }

    pub fn all() -> Result<Vec<Self>> {
        let count = Self::count()?;
        let mut devices = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            devices.push(Self::at(ordinal)?);
        }
        Ok(devices)
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn limits(&self) -> DeviceLimits {
        self.limits
    }

    pub fn properties(&self) -> Result<DeviceProperties> {
        DeviceProperties::query(self.ordinal)
    }

    pub fn compile_hip_source(&self, source: &str) -> Result<Module> {
        let code_object = hiprtc::compile_code_object_cached(source, &self.arch)?;
        self.load_code_object(code_object.as_ref())
    }

    pub fn compile_hip_source_specialized(
        &self,
        source: &str,
        extra_options: &[&str],
        launch_metadata: &str,
    ) -> Result<Module> {
        let code_object = hiprtc::compile_code_object_cached_with_metadata(
            source,
            &self.arch,
            extra_options,
            launch_metadata,
        )?;
        self.load_code_object(code_object.as_ref())
    }

    pub fn compile_hip_source_comgr(&self, source: &str) -> Result<Module> {
        let code_object = hiprtc::compile_code_object_cached_comgr(source, &self.arch)?;
        self.load_code_object(code_object.as_ref())
    }

    pub fn compile_hip_source_specialized_comgr(
        &self,
        source: &str,
        extra_options: &[&str],
        launch_metadata: &str,
    ) -> Result<Module> {
        let code_object = hiprtc::compile_code_object_cached_comgr_with_metadata(
            source,
            &self.arch,
            extra_options,
            launch_metadata,
        )?;
        self.load_code_object(code_object.as_ref())
    }

    pub fn load_code_object(&self, code_object: &[u8]) -> Result<Module> {
        let module = hip::Module::from_code_object(code_object)?;
        Ok(Module {
            module,
            limits: self.limits,
            device_ordinal: self.ordinal,
        })
    }

    pub fn load_code_object_file(&self, path: impl AsRef<Path>) -> Result<Module> {
        let code_object = std::fs::read(path)?;
        self.load_code_object(&code_object)
    }

    pub fn execution_context(&self) -> Result<crate::ExecutionContext> {
        crate::ExecutionContext::new(self)
    }

    pub fn default_mem_pool(&self) -> Result<hip::MemPool> {
        Ok(hip::MemPool::default_for_device(self.ordinal)?)
    }

    pub fn current_mem_pool(&self) -> Result<hip::MemPool> {
        Ok(hip::MemPool::current_for_device(self.ordinal)?)
    }

    pub fn set_mem_pool(&self, pool: hip::MemPool) -> Result<()> {
        Ok(pool.set_current_for_device(self.ordinal)?)
    }

    pub fn create_mem_pool(&self) -> Result<hip::OwnedMemPool> {
        Ok(hip::OwnedMemPool::new_for_device(self.ordinal)?)
    }

    pub fn virtual_memory_granularity(
        &self,
        granularity: hip::MemAllocationGranularity,
    ) -> Result<usize> {
        Ok(hip::DeviceVirtualMemory::allocation_granularity(
            self.ordinal,
            granularity,
        )?)
    }

    pub fn reserve_virtual_memory(
        &self,
        requested_size: usize,
    ) -> Result<hip::DeviceVirtualMemory> {
        Ok(hip::DeviceVirtualMemory::new_for_device(
            self.ordinal,
            requested_size,
        )?)
    }

    pub fn can_access_peer(&self, peer: &Device) -> Result<bool> {
        Ok(hip::can_access_peer(self.ordinal, peer.ordinal)?)
    }

    pub fn enable_peer_access(&self, peer: &Device) -> Result<()> {
        hip::set_device(self.ordinal)?;
        Ok(hip::enable_peer_access(peer.ordinal)?)
    }

    pub fn disable_peer_access(&self, peer: &Device) -> Result<()> {
        hip::set_device(self.ordinal)?;
        Ok(hip::disable_peer_access(peer.ordinal)?)
    }

    pub fn supports_cooperative_launch(&self) -> Result<bool> {
        Ok(hip::device_attribute_bool(
            self.ordinal,
            hip::HIP_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        )?)
    }

    pub fn supports_cooperative_multi_device_launch(&self) -> Result<bool> {
        Ok(hip::device_attribute_bool(
            self.ordinal,
            hip::HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
        )?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProperties {
    pub ordinal: i32,
    pub managed_memory: bool,
    pub concurrent_managed_access: bool,
    pub cooperative_launch: bool,
    pub cooperative_multi_device_launch: bool,
    pub direct_managed_mem_access_from_host: bool,
    pub can_map_host_memory: bool,
    pub can_use_host_pointer_for_registered_mem: bool,
    pub host_native_atomic_supported: bool,
    pub pageable_memory_access: bool,
    pub pageable_memory_access_uses_host_page_tables: bool,
    pub memory_pools_supported: bool,
    pub unified_addressing: bool,
    pub host_register_supported: bool,
    pub async_engine_count: u32,
    pub multiprocessor_count: u32,
    pub warp_size: u32,
    pub clock_instruction_rate_khz: u32,
    pub wall_clock_rate_khz: u32,
}

impl DeviceProperties {
    fn query(ordinal: i32) -> Result<Self> {
        Ok(Self {
            ordinal,
            managed_memory: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
            )?,
            concurrent_managed_access: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
            )?,
            cooperative_launch: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
            )?,
            cooperative_multi_device_launch: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
            )?,
            direct_managed_mem_access_from_host: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
            )?,
            can_map_host_memory: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
            )?,
            can_use_host_pointer_for_registered_mem: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
            )?,
            host_native_atomic_supported: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
            )?,
            pageable_memory_access: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            )?,
            pageable_memory_access_uses_host_page_tables: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
            )?,
            memory_pools_supported: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )?,
            unified_addressing: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            )?,
            host_register_supported: hip::device_attribute_bool(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
            )?,
            async_engine_count: hip::device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
            )?,
            multiprocessor_count: hip::device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?,
            warp_size: hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_WARP_SIZE)?,
            clock_instruction_rate_khz: optional_device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_CLOCK_INSTRUCTION_RATE,
            )?,
            wall_clock_rate_khz: optional_device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE,
            )?,
        })
    }

    pub const fn has_clock_rate_metadata(self) -> bool {
        self.clock_instruction_rate_khz != 0 || self.wall_clock_rate_khz != 0
    }

    pub const fn mapped_host_memory_kind(self) -> Option<AtomicMemoryKind> {
        if self.can_map_host_memory && self.host_native_atomic_supported {
            Some(AtomicMemoryKind::MappedCoherentHost)
        } else {
            None
        }
    }

    pub const fn mapped_host_reference_capture_kind(self) -> Option<AtomicMemoryKind> {
        self.mapped_host_memory_kind()
    }

    pub const fn managed_memory_kind(
        self,
        requested: hip::ManagedMemoryKind,
    ) -> Option<AtomicMemoryKind> {
        if !self.managed_memory {
            return None;
        }
        match requested {
            hip::ManagedMemoryKind::FineGrain
                if self.concurrent_managed_access && self.host_native_atomic_supported =>
            {
                Some(AtomicMemoryKind::ManagedFineGrain)
            }
            hip::ManagedMemoryKind::FineGrain | hip::ManagedMemoryKind::CoarseGrain => {
                Some(AtomicMemoryKind::ManagedCoarseGrain)
            }
        }
    }

    pub const fn managed_host_reference_capture_kind(
        self,
        requested: hip::ManagedMemoryKind,
    ) -> Option<AtomicMemoryKind> {
        self.managed_memory_kind(requested)
    }
}

fn optional_device_attribute(ordinal: i32, attribute: std::ffi::c_int) -> Result<u32> {
    match hip::device_attribute(ordinal, attribute) {
        Ok(value) => Ok(value),
        Err(err)
            if matches!(
                err.code(),
                Some(hip::HIP_ERROR_NOT_SUPPORTED | hip::HIP_ERROR_INVALID_VALUE)
            ) =>
        {
            Ok(0)
        }
        Err(err) => Err(err.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    pub const fn as_tuple(self) -> (u32, u32, u32) {
        (self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_threads_per_block: u32,
    pub max_block_dim: Dim3,
    pub max_grid_dim: Dim3,
    pub max_shared_mem_per_block: u32,
    pub max_shared_mem_per_block_optin: u32,
    pub max_shared_mem_per_multiprocessor: u32,
}

impl DeviceLimits {
    pub const fn prototype() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_block_dim: Dim3::new(1024, 1024, 1024),
            max_grid_dim: Dim3::new(u32::MAX, u32::MAX, u32::MAX),
            max_shared_mem_per_block: 64 * 1024,
            max_shared_mem_per_block_optin: 64 * 1024,
            max_shared_mem_per_multiprocessor: 64 * 1024,
        }
    }

    fn query(ordinal: i32) -> Result<Self> {
        Ok(Self {
            max_threads_per_block: hip::device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            )?,
            max_block_dim: Dim3::new(
                hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
                hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
                hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
            ),
            max_grid_dim: Dim3::new(
                hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
                hip::device_attribute(ordinal, hip::HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
            ),
            max_shared_mem_per_block: hip::device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )?,
            max_shared_mem_per_block_optin: hip::device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_SHARED_MEM_PER_BLOCK_OPTIN,
            )?,
            max_shared_mem_per_multiprocessor: hip::device_attribute(
                ordinal,
                hip::HIP_DEVICE_ATTRIBUTE_SHARED_MEM_PER_MULTIPROCESSOR,
            )?,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelMetadata {
    pub max_flat_workgroup_size: Option<u32>,
    pub static_shared_mem_bytes: u32,
    pub uses_dynamic_shared_mem: bool,
    pub wavefront_size: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelResource {
    pub name: &'static str,
    pub kernarg_segment_size: Option<u32>,
    pub kernarg_segment_align: Option<u32>,
    pub max_flat_workgroup_size: Option<u32>,
    pub group_segment_fixed_size: Option<u32>,
    pub private_segment_fixed_size: Option<u32>,
    pub sgpr_count: Option<u32>,
    pub vgpr_count: Option<u32>,
    pub sgpr_spill_count: Option<u32>,
    pub vgpr_spill_count: Option<u32>,
    pub wavefront_size: Option<u32>,
    pub uses_dynamic_shared_mem: bool,
    pub uses_dynamic_stack: Option<bool>,
}

impl KernelResource {
    pub const fn launch_metadata(self) -> KernelMetadata {
        KernelMetadata {
            max_flat_workgroup_size: self.max_flat_workgroup_size,
            static_shared_mem_bytes: match self.group_segment_fixed_size {
                Some(value) => value,
                None => 0,
            },
            uses_dynamic_shared_mem: self.uses_dynamic_shared_mem,
            wavefront_size: self.wavefront_size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicMemoryKind {
    DefaultDevice,
    FineGrainedDevice,
    MappedCoherentHost,
    ManagedFineGrain,
    ManagedCoarseGrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemScopeAtomicVisibility {
    DeviceOnly,
    HostVisibleAfterSynchronization,
    HostVisibleDuringKernel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostReferenceCaptureVisibility {
    DeviceOnly,
    HostVisibleAfterSynchronization,
    HostVisibleDuringKernel,
}

impl AtomicMemoryKind {
    pub const fn system_scope_visibility(self) -> SystemScopeAtomicVisibility {
        match self {
            Self::DefaultDevice | Self::FineGrainedDevice => {
                SystemScopeAtomicVisibility::DeviceOnly
            }
            Self::MappedCoherentHost | Self::ManagedFineGrain => {
                SystemScopeAtomicVisibility::HostVisibleDuringKernel
            }
            Self::ManagedCoarseGrain => {
                SystemScopeAtomicVisibility::HostVisibleAfterSynchronization
            }
        }
    }

    pub const fn allows_host_concurrent_system_scope(self) -> bool {
        matches!(
            self.system_scope_visibility(),
            SystemScopeAtomicVisibility::HostVisibleDuringKernel
        )
    }

    pub const fn host_reference_capture_visibility(self) -> HostReferenceCaptureVisibility {
        match self {
            Self::DefaultDevice | Self::FineGrainedDevice => {
                HostReferenceCaptureVisibility::DeviceOnly
            }
            Self::MappedCoherentHost | Self::ManagedFineGrain => {
                HostReferenceCaptureVisibility::HostVisibleDuringKernel
            }
            Self::ManagedCoarseGrain => {
                HostReferenceCaptureVisibility::HostVisibleAfterSynchronization
            }
        }
    }

    pub const fn allows_host_reference_capture_during_kernel(self) -> bool {
        matches!(
            self.host_reference_capture_visibility(),
            HostReferenceCaptureVisibility::HostVisibleDuringKernel
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyMaxPotentialBlockSize {
    pub min_grid_size: u32,
    pub block_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyActiveBlocks {
    pub blocks_per_multiprocessor: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchRecommendation {
    pub config: LaunchConfig,
    pub min_grid_size: u32,
    pub block_size: u32,
    pub dynamic_shared_mem_bytes: u32,
    pub active_blocks_per_multiprocessor: u32,
    pub waves_per_block: Option<u32>,
    pub waves_per_multiprocessor: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub const DEFAULT_BLOCK_X: u32 = 256;

    pub const fn new(grid: Dim3, block: Dim3) -> Self {
        Self {
            grid,
            block,
            shared_mem_bytes: 0,
        }
    }

    pub fn try_new(grid: Dim3, block: Dim3) -> Result<Self> {
        let config = Self::new(grid, block);
        if config.grid.x == 0
            || config.grid.y == 0
            || config.grid.z == 0
            || config.block.x == 0
            || config.block.y == 0
            || config.block.z == 0
        {
            return Err(Error::InvalidLaunch(format!(
                "grid and block dimensions must be nonzero, got grid=({}, {}, {}) block=({}, {}, {})",
                config.grid.x,
                config.grid.y,
                config.grid.z,
                config.block.x,
                config.block.y,
                config.block.z
            )));
        }
        Ok(config)
    }

    pub fn for_num_elems(num_elems: usize) -> Self {
        Self::for_num_elems_with_block_size(num_elems, Self::DEFAULT_BLOCK_X)
    }

    pub fn try_for_num_elems(num_elems: usize) -> Result<Self> {
        Self::try_for_num_elems_with_block_size(num_elems, Self::DEFAULT_BLOCK_X)
    }

    pub fn for_num_elems_with_block_size(num_elems: usize, block_x: u32) -> Self {
        Self::try_for_num_elems_with_block_size(num_elems, block_x)
            .expect("invalid one-dimensional launch configuration")
    }

    pub fn try_for_num_elems_with_block_size(num_elems: usize, block_x: u32) -> Result<Self> {
        if block_x == 0 {
            return Err(Error::InvalidLaunch(
                "one-dimensional launch block size must be nonzero".to_string(),
            ));
        }
        let grid_x = num_elems.div_ceil(block_x as usize);
        let grid_x = u32::try_from(grid_x).map_err(|_| {
            Error::InvalidLaunch(format!(
                "one-dimensional launch for {num_elems} elements with block.x={block_x} requires grid.x={grid_x}, exceeding u32 launch limit"
            ))
        })?;
        Self::try_new(Dim3::x(grid_x), Dim3::x(block_x))
    }

    pub fn for_2d(width: u32, height: u32, block_x: u32, block_y: u32) -> Self {
        Self::try_for_2d(width, height, block_x, block_y)
            .expect("invalid two-dimensional launch configuration")
    }

    pub fn try_for_2d(width: u32, height: u32, block_x: u32, block_y: u32) -> Result<Self> {
        if block_x == 0 || block_y == 0 {
            return Err(Error::InvalidLaunch(format!(
                "two-dimensional launch block dimensions must be nonzero, got block=({block_x}, {block_y})"
            )));
        }
        let grid_x = width.div_ceil(block_x);
        let grid_y = height.div_ceil(block_y);
        Self::try_new(Dim3::new(grid_x, grid_y, 1), Dim3::new(block_x, block_y, 1))
    }

    pub const fn with_shared_mem_bytes(mut self, shared_mem_bytes: u32) -> Self {
        self.shared_mem_bytes = shared_mem_bytes;
        self
    }

    pub fn try_with_dynamic_shared_mem<T>(self, elements: usize) -> Result<Self> {
        let bytes = elements
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                Error::InvalidLaunch(format!(
                    "dynamic shared memory size overflows usize for {elements} elements"
                ))
            })?;
        let bytes = u32::try_from(bytes).map_err(|_| {
            Error::InvalidLaunch(format!(
                "dynamic shared memory request is {bytes} bytes, exceeding u32 launch limit"
            ))
        })?;
        Ok(self.with_shared_mem_bytes(bytes))
    }
}

pub fn validate_launch_config(config: LaunchConfig) -> Result<()> {
    validate_launch_config_for_limits(config, DeviceLimits::prototype(), KernelMetadata::default())
}

pub fn validate_launch_config_for_limits(
    config: LaunchConfig,
    limits: DeviceLimits,
    metadata: KernelMetadata,
) -> Result<()> {
    if config.grid.x == 0
        || config.grid.y == 0
        || config.grid.z == 0
        || config.block.x == 0
        || config.block.y == 0
        || config.block.z == 0
    {
        return Err(Error::InvalidLaunch(format!(
            "grid and block dimensions must be nonzero, got grid=({}, {}, {}) block=({}, {}, {})",
            config.grid.x,
            config.grid.y,
            config.grid.z,
            config.block.x,
            config.block.y,
            config.block.z
        )));
    }

    let block_threads = config.block.x as u64 * config.block.y as u64 * config.block.z as u64;
    let max_threads = metadata
        .max_flat_workgroup_size
        .unwrap_or(limits.max_threads_per_block)
        .min(limits.max_threads_per_block);
    if block_threads > max_threads as u64 {
        return Err(Error::InvalidLaunch(format!(
            "block has {block_threads} threads, but this kernel/device supports at most {max_threads}"
        )));
    }

    if config.block.x > limits.max_block_dim.x
        || config.block.y > limits.max_block_dim.y
        || config.block.z > limits.max_block_dim.z
    {
        return Err(Error::InvalidLaunch(format!(
            "block dimensions ({}, {}, {}) exceed device maximum ({}, {}, {})",
            config.block.x,
            config.block.y,
            config.block.z,
            limits.max_block_dim.x,
            limits.max_block_dim.y,
            limits.max_block_dim.z
        )));
    }

    if config.grid.x > limits.max_grid_dim.x
        || config.grid.y > limits.max_grid_dim.y
        || config.grid.z > limits.max_grid_dim.z
    {
        return Err(Error::InvalidLaunch(format!(
            "grid dimensions ({}, {}, {}) exceed device maximum ({}, {}, {})",
            config.grid.x,
            config.grid.y,
            config.grid.z,
            limits.max_grid_dim.x,
            limits.max_grid_dim.y,
            limits.max_grid_dim.z
        )));
    }

    let total_shared_mem = metadata.static_shared_mem_bytes as u64 + config.shared_mem_bytes as u64;
    if metadata.uses_dynamic_shared_mem && config.shared_mem_bytes == 0 {
        return Err(Error::InvalidLaunch(
            "kernel uses dynamic LDS/shared memory, but launch requested 0 dynamic bytes"
                .to_string(),
        ));
    }
    if total_shared_mem > limits.max_shared_mem_per_block as u64 {
        return Err(Error::InvalidLaunch(format!(
            "kernel requests {total_shared_mem} bytes of LDS/shared memory ({} static + {} dynamic), but device limit is {} bytes per block",
            metadata.static_shared_mem_bytes,
            config.shared_mem_bytes,
            limits.max_shared_mem_per_block
        )));
    }

    Ok(())
}

pub fn validate_cooperative_launch_config(config: LaunchConfig) -> Result<()> {
    const HIP_COOPERATIVE_WORK_ITEMS_LIMIT: u64 = 1u64 << 32;
    let dims = [
        ("x", config.grid.x as u64, config.block.x as u64),
        ("y", config.grid.y as u64, config.block.y as u64),
        ("z", config.grid.z as u64, config.block.z as u64),
    ];
    for (axis, grid, block) in dims {
        let work_items = grid * block;
        if work_items >= HIP_COOPERATIVE_WORK_ITEMS_LIMIT {
            return Err(Error::InvalidLaunch(format!(
                "cooperative launch has grid.{axis} * block.{axis} = {work_items}, but HIP requires each dimension to stay below 2^32 work-items"
            )));
        }
    }
    Ok(())
}

pub fn validate_cooperative_launch_for_device(
    config: LaunchConfig,
    properties: DeviceProperties,
    occupancy: OccupancyActiveBlocks,
) -> Result<()> {
    validate_cooperative_launch_config(config)?;
    if !properties.cooperative_launch {
        return Err(Error::InvalidLaunch(format!(
            "device {} does not support cooperative launch",
            properties.ordinal
        )));
    }

    let grid_blocks = cooperative_grid_blocks(config)?;
    let resident_blocks = (properties.multiprocessor_count as u64)
        .checked_mul(occupancy.blocks_per_multiprocessor as u64)
        .ok_or_else(|| {
            Error::InvalidLaunch(
                "cooperative launch resident block capacity overflows u64".to_string(),
            )
        })?;
    if resident_blocks == 0 {
        return Err(Error::InvalidLaunch(format!(
            "cooperative launch has no resident block capacity: multiprocessors={} blocks_per_multiprocessor={}",
            properties.multiprocessor_count, occupancy.blocks_per_multiprocessor
        )));
    }
    if grid_blocks > resident_blocks {
        return Err(Error::InvalidLaunch(format!(
            "cooperative launch requests {grid_blocks} blocks, but the device can keep at most {resident_blocks} resident blocks ({} multiprocessors * {} blocks per multiprocessor)",
            properties.multiprocessor_count, occupancy.blocks_per_multiprocessor
        )));
    }
    Ok(())
}

pub fn validate_cooperative_multi_device_launch_for_device(
    config: LaunchConfig,
    properties: DeviceProperties,
    occupancy: OccupancyActiveBlocks,
) -> Result<()> {
    validate_cooperative_launch_config(config)?;
    if !properties.cooperative_multi_device_launch {
        return Err(Error::InvalidLaunch(format!(
            "device {} does not support cooperative multi-device launch",
            properties.ordinal
        )));
    }

    let grid_blocks = cooperative_grid_blocks(config)?;
    let resident_blocks = (properties.multiprocessor_count as u64)
        .checked_mul(occupancy.blocks_per_multiprocessor as u64)
        .ok_or_else(|| {
            Error::InvalidLaunch(
                "cooperative multi-device resident block capacity overflows u64".to_string(),
            )
        })?;
    if resident_blocks == 0 {
        return Err(Error::InvalidLaunch(format!(
            "cooperative multi-device launch has no resident block capacity: multiprocessors={} blocks_per_multiprocessor={}",
            properties.multiprocessor_count, occupancy.blocks_per_multiprocessor
        )));
    }
    if grid_blocks > resident_blocks {
        return Err(Error::InvalidLaunch(format!(
            "cooperative multi-device launch requests {grid_blocks} blocks, but the device can keep at most {resident_blocks} resident blocks ({} multiprocessors * {} blocks per multiprocessor)",
            properties.multiprocessor_count, occupancy.blocks_per_multiprocessor
        )));
    }
    Ok(())
}

fn cooperative_grid_blocks(config: LaunchConfig) -> Result<u64> {
    if config.grid.x == 0
        || config.grid.y == 0
        || config.grid.z == 0
        || config.block.x == 0
        || config.block.y == 0
        || config.block.z == 0
    {
        return Err(Error::InvalidLaunch(format!(
            "cooperative launch grid and block dimensions must be nonzero, got grid=({}, {}, {}) block=({}, {}, {})",
            config.grid.x,
            config.grid.y,
            config.grid.z,
            config.block.x,
            config.block.y,
            config.block.z
        )));
    }
    (config.grid.x as u64)
        .checked_mul(config.grid.y as u64)
        .and_then(|value| value.checked_mul(config.grid.z as u64))
        .ok_or_else(|| {
            Error::InvalidLaunch("cooperative launch grid block count overflows u64".to_string())
        })
}

pub fn validate_buffer_len(name: &str, actual: usize, required: usize) -> Result<()> {
    if actual < required {
        Err(Error::InvalidLaunch(format!(
            "buffer `{name}` has length {actual}, but kernel requires at least {required}"
        )))
    } else {
        Ok(())
    }
}

pub fn validate_block_x(config: LaunchConfig, block_x: u32) -> Result<()> {
    if config.block.x != block_x {
        Err(Error::InvalidLaunch(format!(
            "`block_x` argument is {block_x}, but launch config block.x is {}",
            config.block.x
        )))
    } else {
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceSlice<T> {
    pub ptr: *const T,
    pub len: usize,
}

impl<T> DeviceSlice<T> {
    pub fn from_buffer(buffer: &hip::DeviceBuffer<T>) -> Self {
        Self {
            ptr: buffer.as_ptr(),
            len: buffer.len(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceSliceMut<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> DeviceSliceMut<T> {
    pub fn from_buffer(buffer: &hip::DeviceBuffer<T>) -> Self {
        Self {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len(),
        }
    }

    pub const fn as_const(self) -> DeviceSlice<T> {
        DeviceSlice {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

pub fn validate_device_buffers_disjoint<T, U>(
    lhs_name: &str,
    lhs: &hip::DeviceBuffer<T>,
    rhs_name: &str,
    rhs: &hip::DeviceBuffer<U>,
) -> Result<()> {
    let Some((lhs_start, lhs_end)) = device_buffer_byte_range(lhs_name, lhs)? else {
        return Ok(());
    };
    let Some((rhs_start, rhs_end)) = device_buffer_byte_range(rhs_name, rhs)? else {
        return Ok(());
    };

    if lhs_start < rhs_end && rhs_start < lhs_end {
        Err(Error::InvalidLaunch(format!(
            "mutable buffer `{lhs_name}` aliases `{rhs_name}`; generated bindings require disjoint device buffers"
        )))
    } else {
        Ok(())
    }
}

fn device_buffer_byte_range<T>(
    name: &str,
    buffer: &hip::DeviceBuffer<T>,
) -> Result<Option<(usize, usize)>> {
    let bytes = buffer
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            Error::InvalidLaunch(format!(
                "buffer `{name}` byte length overflows usize for disjointness validation"
            ))
        })?;
    if bytes == 0 {
        return Ok(None);
    }
    let start = buffer.as_ptr() as usize;
    let end = start.checked_add(bytes).ok_or_else(|| {
        Error::InvalidLaunch(format!(
            "buffer `{name}` address range overflows usize for disjointness validation"
        ))
    })?;
    Ok(Some((start, end)))
}

pub struct Module {
    module: hip::Module,
    limits: DeviceLimits,
    device_ordinal: i32,
}

impl Module {
    /// Returns the underlying HIP module handle without transferring ownership.
    ///
    /// This is an interop escape hatch for ROCm APIs that need the raw module.
    /// The handle remains valid only while this `Module` or kernels/globals
    /// derived from it keep the module alive. Callers must not unload it and
    /// must make `device_ordinal()` current before using it with foreign HIP
    /// APIs.
    pub unsafe fn as_raw_hip_module(&self) -> hip::HipModule {
        unsafe { self.module.as_raw() }
    }

    /// Returns the HIP device ordinal that owns this module.
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub fn kernel(&self, name: &CStr) -> Result<Kernel> {
        self.kernel_with_metadata(name, KernelMetadata::default())
    }

    pub fn kernel_with_metadata(&self, name: &CStr, metadata: KernelMetadata) -> Result<Kernel> {
        Ok(Kernel {
            function: self.module.function(name)?,
            limits: self.limits,
            device_ordinal: self.device_ordinal,
            metadata,
        })
    }

    pub fn global<T>(&self, name: &CStr) -> Result<hip::Global<T>> {
        Ok(self.module.global(name)?)
    }
}

pub struct Kernel {
    function: hip::Function,
    limits: DeviceLimits,
    device_ordinal: i32,
    metadata: KernelMetadata,
}

/// Checked runtime launch entry for cooperative multi-device module launches.
///
/// `Kernel` methods validate the launch shape and device support for each entry
/// before delegating to the low-level HIP wrapper. Raw ABI and pointed-to
/// argument lifetimes remain the caller's responsibility.
pub struct CooperativeKernelLaunch<'a> {
    pub kernel: &'a Kernel,
    pub stream: &'a hip::Stream,
    pub config: LaunchConfig,
    pub params: &'a mut [*mut c_void],
}

impl<'a> CooperativeKernelLaunch<'a> {
    pub fn new(
        kernel: &'a Kernel,
        stream: &'a hip::Stream,
        config: LaunchConfig,
        params: &'a mut [*mut c_void],
    ) -> Self {
        Self {
            kernel,
            stream,
            config,
            params,
        }
    }
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Returns the underlying HIP function handle without transferring ownership.
    ///
    /// This is an interop escape hatch for ROCm APIs that need the raw function.
    /// The handle remains valid only while this `Kernel` is alive. Callers must
    /// make `device_ordinal()` current before using it with foreign HIP APIs.
    pub unsafe fn as_raw_hip_function(&self) -> hip::HipFunction {
        unsafe { self.function.as_raw() }
    }

    /// Returns the HIP device ordinal that owns this function.
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub const fn metadata(&self) -> KernelMetadata {
        self.metadata
    }

    pub const fn limits(&self) -> DeviceLimits {
        self.limits
    }

    pub fn validate_launch_config(&self, config: LaunchConfig) -> Result<()> {
        validate_launch_config_for_limits(config, self.limits, self.metadata)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        dynamic_shared_mem_per_block: u32,
        block_size_limit: u32,
    ) -> Result<OccupancyMaxPotentialBlockSize> {
        self.validate_occupancy_shared_mem(dynamic_shared_mem_per_block)?;
        let (min_grid_size, block_size) = self
            .function
            .occupancy_max_potential_block_size(dynamic_shared_mem_per_block, block_size_limit)?;
        Ok(OccupancyMaxPotentialBlockSize {
            min_grid_size,
            block_size,
        })
    }

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_shared_mem_per_block: u32,
    ) -> Result<OccupancyActiveBlocks> {
        if block_size == 0 {
            return Err(Error::InvalidLaunch(
                "occupancy block size must be nonzero".to_string(),
            ));
        }
        self.validate_occupancy_shared_mem(dynamic_shared_mem_per_block)?;
        Ok(OccupancyActiveBlocks {
            blocks_per_multiprocessor: self
                .function
                .occupancy_max_active_blocks_per_multiprocessor(
                    block_size,
                    dynamic_shared_mem_per_block,
                )?,
        })
    }

    pub fn occupancy_for_config(&self, config: LaunchConfig) -> Result<OccupancyActiveBlocks> {
        validate_launch_config_for_limits(config, self.limits, self.metadata)?;
        self.occupancy_max_active_blocks_per_multiprocessor(
            config.block.x * config.block.y * config.block.z,
            config.shared_mem_bytes,
        )
    }

    pub fn validate_cooperative_launch_config(
        &self,
        config: LaunchConfig,
    ) -> Result<OccupancyActiveBlocks> {
        validate_launch_config_for_limits(config, self.limits, self.metadata)?;
        let occupancy = self.occupancy_for_config(config)?;
        let properties = DeviceProperties::query(self.device_ordinal)?;
        validate_cooperative_launch_for_device(config, properties, occupancy)?;
        Ok(occupancy)
    }

    pub fn recommend_1d_launch(
        &self,
        num_elems: usize,
        dynamic_shared_mem_per_block: u32,
        block_size_limit: u32,
    ) -> Result<LaunchRecommendation> {
        let potential = self
            .occupancy_max_potential_block_size(dynamic_shared_mem_per_block, block_size_limit)?;
        if potential.block_size == 0 {
            return Err(Error::InvalidLaunch(
                "HIP occupancy returned a zero block-size recommendation".to_string(),
            ));
        }
        let config =
            LaunchConfig::try_for_num_elems_with_block_size(num_elems, potential.block_size)?
                .with_shared_mem_bytes(dynamic_shared_mem_per_block);
        validate_launch_config_for_limits(config, self.limits, self.metadata)?;
        let active = self.occupancy_for_config(config)?;
        let waves_per_block = self
            .metadata
            .wavefront_size
            .filter(|wavefront| *wavefront > 0)
            .map(|wavefront| potential.block_size.div_ceil(wavefront));
        let waves_per_multiprocessor =
            waves_per_block.map(|waves| waves * active.blocks_per_multiprocessor);

        Ok(LaunchRecommendation {
            config,
            min_grid_size: potential.min_grid_size,
            block_size: potential.block_size,
            dynamic_shared_mem_bytes: dynamic_shared_mem_per_block,
            active_blocks_per_multiprocessor: active.blocks_per_multiprocessor,
            waves_per_block,
            waves_per_multiprocessor,
        })
    }

    fn validate_occupancy_shared_mem(&self, dynamic_shared_mem_per_block: u32) -> Result<()> {
        let total_shared_mem =
            self.metadata.static_shared_mem_bytes as u64 + dynamic_shared_mem_per_block as u64;
        if total_shared_mem > self.limits.max_shared_mem_per_block as u64 {
            return Err(Error::InvalidLaunch(format!(
                "occupancy query requests {total_shared_mem} bytes of LDS/shared memory ({} static + {} dynamic), but device limit is {} bytes per block",
                self.metadata.static_shared_mem_bytes,
                dynamic_shared_mem_per_block,
                self.limits.max_shared_mem_per_block
            )));
        }
        Ok(())
    }

    /// Validates the launch shape, then launches this kernel on the default
    /// stream with raw ABI parameters.
    ///
    /// # Safety
    ///
    /// Launch-shape validation does not validate the kernel ABI. `params` must
    /// contain exactly the ABI expected by this kernel, and every pointer
    /// referenced by those arguments must remain valid until the launch has
    /// completed.
    pub unsafe fn launch_raw(
        &self,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.validate_launch_config(config)?;
        unsafe { self.launch_raw_unchecked(config, params) }
    }

    /// Launches this kernel without validating the launch configuration.
    ///
    /// # Safety
    ///
    /// The caller must ensure `config` is valid for this kernel and device,
    /// `params` contains exactly the ABI expected by the kernel, and every
    /// pointer in `params` remains valid until the launch has completed.
    pub unsafe fn launch_raw_unchecked(
        &self,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe {
            self.function.launch(
                config.grid.as_tuple(),
                config.block.as_tuple(),
                config.shared_mem_bytes,
                params,
            )?;
        }
        Ok(())
    }

    /// Validates the launch shape, then adds this kernel to an explicit HIP
    /// graph with raw ABI parameters.
    ///
    /// # Safety
    ///
    /// Launch-shape validation does not validate the kernel ABI or graph
    /// resource lifetimes. `params` must contain exactly the ABI expected by
    /// this kernel, every buffer or pointer referenced by those arguments must
    /// remain valid until any graph execution using this node completes, and
    /// `graph` must be associated with the kernel's device/context.
    pub unsafe fn add_graph_node_raw(
        &self,
        graph: &hip::Graph,
        dependencies: &[hip::GraphNode],
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<hip::GraphNode> {
        self.validate_launch_config(config)?;
        unsafe { self.add_graph_node_raw_unchecked(graph, dependencies, config, params) }
    }

    /// Adds this kernel to an explicit HIP graph without validating the launch
    /// configuration.
    ///
    /// # Safety
    ///
    /// The caller must ensure `config` is valid for this kernel and device,
    /// `params` contains exactly the ABI expected by the kernel, every buffer
    /// or pointer referenced by those arguments remains valid until any graph
    /// execution using this node completes, and the graph is associated with
    /// the kernel's device/context.
    pub unsafe fn add_graph_node_raw_unchecked(
        &self,
        graph: &hip::Graph,
        dependencies: &[hip::GraphNode],
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<hip::GraphNode> {
        Ok(unsafe {
            graph.add_kernel_node(
                dependencies,
                &self.function,
                config.grid.as_tuple(),
                config.block.as_tuple(),
                config.shared_mem_bytes,
                params,
            )?
        })
    }

    /// Validates the launch shape, then launches this kernel on `stream` with
    /// raw ABI parameters.
    ///
    /// # Safety
    ///
    /// Launch-shape validation does not validate the kernel ABI. `stream` must
    /// be valid for the kernel's device/context, `params` must contain exactly
    /// the ABI expected by this kernel, and every pointer referenced by those
    /// arguments must remain valid until `stream` reaches the launch.
    pub unsafe fn launch_raw_on_stream(
        &self,
        stream: &hip::Stream,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.validate_launch_config(config)?;
        unsafe { self.launch_raw_on_stream_unchecked(stream, config, params) }
    }

    /// Launches this kernel on `stream` without validating the launch
    /// configuration.
    ///
    /// # Safety
    ///
    /// The caller must ensure `config` is valid for this kernel and device,
    /// `params` contains exactly the ABI expected by the kernel, every pointer
    /// in `params` remains valid until `stream` reaches the launch, and the
    /// stream is associated with the kernel's device/context.
    pub unsafe fn launch_raw_on_stream_unchecked(
        &self,
        stream: &hip::Stream,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe {
            self.function.launch_on_stream(
                config.grid.as_tuple(),
                config.block.as_tuple(),
                config.shared_mem_bytes,
                stream.as_raw(),
                params,
            )?;
        }
        Ok(())
    }

    /// Validates launch shape and cooperative limits, then launches this
    /// kernel cooperatively on `stream` with raw ABI parameters.
    ///
    /// # Safety
    ///
    /// Launch-shape and cooperative-limit validation do not validate the
    /// kernel ABI. `stream` must be valid for the kernel's device/context, the
    /// kernel must be eligible for cooperative launch, `params` must contain
    /// exactly the ABI expected by this kernel, and every pointer referenced by
    /// those arguments must remain valid until `stream` reaches the launch.
    pub unsafe fn launch_cooperative_raw_on_stream(
        &self,
        stream: &hip::Stream,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.validate_cooperative_launch_config(config)?;
        unsafe {
            self.function.launch_cooperative_on_stream(
                config.grid.as_tuple(),
                config.block.as_tuple(),
                config.shared_mem_bytes,
                stream.as_raw(),
                params,
            )?;
        }
        Ok(())
    }

    /// Validates and launches cooperative kernels across multiple devices.
    ///
    /// # Safety
    ///
    /// Validation covers launch shape, reported cooperative multi-device
    /// support, and resident block capacity for each entry. It does not
    /// validate the raw kernel ABI, cross-device topology, peer-memory
    /// visibility, or the lifetime of pointed-to kernel arguments. Each
    /// `params` entry must match its kernel ABI exactly, every pointed-to
    /// argument must remain valid until the corresponding stream reaches the
    /// launch, and each stream must belong to the intended device/context.
    pub unsafe fn launch_cooperative_multi_device_raw(
        launches: &mut [CooperativeKernelLaunch<'_>],
    ) -> Result<()> {
        unsafe { Self::launch_cooperative_multi_device_raw_with_flags(launches, 0) }
    }

    /// Validates and launches cooperative kernels across multiple devices with
    /// HIP launch flags.
    ///
    /// # Safety
    ///
    /// Has the same caller obligations as
    /// [`Kernel::launch_cooperative_multi_device_raw`]. `flags` is passed
    /// through to HIP and must be valid for
    /// `hipModuleLaunchCooperativeKernelMultiDevice`.
    pub unsafe fn launch_cooperative_multi_device_raw_with_flags(
        launches: &mut [CooperativeKernelLaunch<'_>],
        flags: u32,
    ) -> Result<()> {
        if launches.len() < 2 {
            return Err(Error::InvalidLaunch(
                "cooperative multi-device launch requires at least two launch entries".to_string(),
            ));
        }
        for launch in launches.iter() {
            validate_launch_config_for_limits(
                launch.config,
                launch.kernel.limits,
                launch.kernel.metadata,
            )?;
            let occupancy = launch.kernel.occupancy_for_config(launch.config)?;
            let properties = DeviceProperties::query(launch.kernel.device_ordinal)?;
            validate_cooperative_multi_device_launch_for_device(
                launch.config,
                properties,
                occupancy,
            )?;
        }
        let mut raw_launches = launches
            .iter_mut()
            .map(|launch| {
                hip::CooperativeFunctionLaunch::new(
                    &launch.kernel.function,
                    launch.config.grid.as_tuple(),
                    launch.config.block.as_tuple(),
                    launch.config.shared_mem_bytes,
                    launch.stream,
                    launch.params,
                )
            })
            .collect::<Vec<_>>();
        unsafe { hip::launch_cooperative_multi_device(&mut raw_launches, flags)? };
        Ok(())
    }
}

fn detect_arch() -> Option<String> {
    if let Ok(arch) = std::env::var("ROCM_OXIDE_ARCH") {
        if !arch.trim().is_empty() {
            return Some(arch);
        }
    }

    let output = Command::new(rocminfo_path()).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(|line| {
        let (_, value) = line.split_once("Name:")?;
        let value = value.trim();
        if value.starts_with("gfx") && !value.contains('-') {
            Some(value.to_string())
        } else {
            None
        }
    })
}

fn rocminfo_path() -> PathBuf {
    if let Some(path) = std::env::var_os("ROCMINFO").filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    std::env::var_os("ROCM_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
        .join("bin/rocminfo")
}

#[cfg(test)]
mod tests {
    use super::{
        AtomicMemoryKind, Device, DeviceLimits, DeviceProperties, Dim3,
        HostReferenceCaptureVisibility, KernelMetadata, LaunchConfig, SystemScopeAtomicVisibility,
    };
    use crate::hip::{DeviceBuffer, ManagedMemoryKind, Stream};

    fn cooperative_props() -> DeviceProperties {
        DeviceProperties {
            ordinal: 0,
            managed_memory: true,
            concurrent_managed_access: true,
            cooperative_launch: true,
            cooperative_multi_device_launch: false,
            direct_managed_mem_access_from_host: false,
            can_map_host_memory: true,
            can_use_host_pointer_for_registered_mem: false,
            host_native_atomic_supported: true,
            pageable_memory_access: false,
            pageable_memory_access_uses_host_page_tables: false,
            memory_pools_supported: true,
            unified_addressing: true,
            host_register_supported: true,
            async_engine_count: 2,
            multiprocessor_count: 4,
            warp_size: 32,
            clock_instruction_rate_khz: 100_000,
            wall_clock_rate_khz: 100_000,
        }
    }

    #[test]
    fn one_dimensional_launch_config_rounds_up() {
        let config = LaunchConfig::for_num_elems(1_025);
        assert_eq!(config.grid, Dim3::x(5));
        assert_eq!(config.block, Dim3::x(256));
        assert_eq!(config.shared_mem_bytes, 0);
    }

    #[test]
    fn module_and_kernel_expose_non_owning_raw_hip_handles() {
        let device = Device::first().expect("device should be visible");
        let module = device
            .compile_hip_source(
                r#"
extern "C" __global__
void raw_handle_probe(unsigned int* out) {
    out[0] = 7;
}
"#,
            )
            .expect("HIP source should compile");

        assert_eq!(module.device_ordinal(), device.ordinal());
        let raw_module = unsafe { module.as_raw_hip_module() };
        assert!(!raw_module.is_null());

        let kernel = module
            .kernel(c"raw_handle_probe")
            .expect("kernel should load");
        assert_eq!(kernel.device_ordinal(), device.ordinal());
        let raw_function = unsafe { kernel.as_raw_hip_function() };
        assert!(!raw_function.is_null());
    }

    #[test]
    fn custom_one_dimensional_block_size_rounds_up() {
        let config = LaunchConfig::for_num_elems_with_block_size(1_025, 128);
        assert_eq!(config.grid, Dim3::x(9));
        assert_eq!(config.block, Dim3::x(128));
    }

    #[test]
    fn fallible_one_dimensional_launch_rejects_zero_block_size() {
        let err = LaunchConfig::try_for_num_elems_with_block_size(1_025, 0)
            .expect_err("zero block size should fail");
        assert!(err.to_string().contains("must be nonzero"));
    }

    #[test]
    fn fallible_one_dimensional_launch_rejects_zero_elements() {
        let err = LaunchConfig::try_for_num_elems(0).expect_err("zero elements should fail");
        assert!(err.to_string().contains("nonzero"));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn fallible_one_dimensional_launch_rejects_grid_overflow() {
        let num_elems = usize::try_from(u64::from(u32::MAX) + 1)
            .expect("u32::MAX + 1 should fit on 64-bit targets");
        let err = LaunchConfig::try_for_num_elems_with_block_size(num_elems, 1)
            .expect_err("grid.x beyond u32 should fail");
        assert!(err.to_string().contains("exceeding u32 launch limit"));
    }

    #[test]
    fn two_dimensional_launch_config_rounds_up() {
        let config = LaunchConfig::for_2d(1_025, 513, 16, 16);
        assert_eq!(config.grid, Dim3::new(65, 33, 1));
        assert_eq!(config.block, Dim3::new(16, 16, 1));
    }

    #[test]
    fn fallible_two_dimensional_launch_rejects_zero_block_size() {
        let err = LaunchConfig::try_for_2d(1_025, 513, 0, 16)
            .expect_err("zero 2D block size should fail");
        assert!(err.to_string().contains("must be nonzero"));
    }

    #[test]
    fn fallible_two_dimensional_launch_rejects_zero_extent() {
        let err = LaunchConfig::try_for_2d(0, 513, 16, 16).expect_err("zero 2D extent should fail");
        assert!(err.to_string().contains("nonzero"));
    }

    #[test]
    fn typed_dynamic_shared_memory_sets_byte_count() {
        let config = LaunchConfig::for_num_elems_with_block_size(128, 128)
            .try_with_dynamic_shared_mem::<f32>(128)
            .expect("f32 LDS byte count should fit");
        assert_eq!(config.shared_mem_bytes, 512);
    }

    #[test]
    fn dim3_keeps_axes_explicit() {
        assert_eq!(Dim3::new(2, 3, 4).as_tuple(), (2, 3, 4));
    }

    #[test]
    fn launch_config_rejects_zero_dimensions() {
        let err = super::validate_launch_config(LaunchConfig::new(Dim3::x(0), Dim3::x(256)))
            .expect_err("zero grid should fail");
        assert!(err.to_string().contains("nonzero"));
    }

    #[test]
    fn launch_config_rejects_kernel_workgroup_limit() {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::x(256));
        let metadata = KernelMetadata {
            max_flat_workgroup_size: Some(128),
            ..KernelMetadata::default()
        };
        let err =
            super::validate_launch_config_for_limits(config, DeviceLimits::prototype(), metadata)
                .expect_err("kernel workgroup limit should fail");
        assert!(err.to_string().contains("at most 128"));
    }

    #[test]
    fn launch_config_rejects_excess_shared_memory() {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::x(256)).with_shared_mem_bytes(512);
        let metadata = KernelMetadata {
            static_shared_mem_bytes: 64 * 1024,
            ..KernelMetadata::default()
        };
        let err =
            super::validate_launch_config_for_limits(config, DeviceLimits::prototype(), metadata)
                .expect_err("total LDS over device limit should fail");
        assert!(err.to_string().contains("LDS/shared memory"));
    }

    #[test]
    fn launch_config_rejects_excess_grid_dimensions() {
        let config = LaunchConfig::new(Dim3::new(3, 1, 1), Dim3::x(256));
        let limits = DeviceLimits {
            max_grid_dim: Dim3::new(2, 1, 1),
            ..DeviceLimits::prototype()
        };
        let err =
            super::validate_launch_config_for_limits(config, limits, KernelMetadata::default())
                .expect_err("grid beyond device limit should fail");
        assert!(err.to_string().contains("grid dimensions"));
    }

    #[test]
    fn launch_config_rejects_missing_dynamic_shared_memory() {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::x(256));
        let metadata = KernelMetadata {
            uses_dynamic_shared_mem: true,
            ..KernelMetadata::default()
        };
        let err =
            super::validate_launch_config_for_limits(config, DeviceLimits::prototype(), metadata)
                .expect_err("dynamic LDS kernel should need dynamic bytes");
        assert!(err.to_string().contains("requested 0 dynamic bytes"));
    }

    #[test]
    fn cooperative_launch_validation_requires_device_support() {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::x(64));
        let mut properties = cooperative_props();
        properties.cooperative_launch = false;
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        let err = super::validate_cooperative_launch_for_device(config, properties, occupancy)
            .expect_err("device without cooperative launch support should fail");
        assert!(err.to_string().contains("does not support cooperative"));
    }

    #[test]
    fn cooperative_launch_validation_rejects_nonresident_grid() {
        let config = LaunchConfig::new(Dim3::x(5), Dim3::x(64));
        let properties = cooperative_props();
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        let err = super::validate_cooperative_launch_for_device(config, properties, occupancy)
            .expect_err("grid larger than resident capacity should fail");
        assert!(err.to_string().contains("resident blocks"));
    }

    #[test]
    fn cooperative_launch_validation_accepts_resident_grid() {
        let config = LaunchConfig::new(Dim3::x(4), Dim3::x(64));
        let properties = cooperative_props();
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        super::validate_cooperative_launch_for_device(config, properties, occupancy)
            .expect("resident cooperative launch should validate");
    }

    #[test]
    fn cooperative_multi_device_launch_validation_requires_device_support() {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::x(64));
        let properties = cooperative_props();
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        let err = super::validate_cooperative_multi_device_launch_for_device(
            config, properties, occupancy,
        )
        .expect_err("device without cooperative multi-device support should fail");
        assert!(err.to_string().contains("multi-device"));
    }

    #[test]
    fn cooperative_multi_device_launch_validation_rejects_nonresident_grid() {
        let config = LaunchConfig::new(Dim3::x(5), Dim3::x(64));
        let mut properties = cooperative_props();
        properties.cooperative_multi_device_launch = true;
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        let err = super::validate_cooperative_multi_device_launch_for_device(
            config, properties, occupancy,
        )
        .expect_err("multi-device grid larger than resident capacity should fail");
        assert!(err.to_string().contains("resident blocks"));
    }

    #[test]
    fn cooperative_multi_device_launch_validation_accepts_resident_grid() {
        let config = LaunchConfig::new(Dim3::x(4), Dim3::x(64));
        let mut properties = cooperative_props();
        properties.cooperative_multi_device_launch = true;
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        super::validate_cooperative_multi_device_launch_for_device(config, properties, occupancy)
            .expect("resident cooperative multi-device launch should validate");
    }

    #[test]
    fn cooperative_launch_validation_rejects_zero_block_dimension() {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::new(0, 1, 1));
        let properties = cooperative_props();
        let occupancy = super::OccupancyActiveBlocks {
            blocks_per_multiprocessor: 1,
        };
        let err = super::validate_cooperative_launch_for_device(config, properties, occupancy)
            .expect_err("zero block dimension should fail");
        assert!(err.to_string().contains("must be nonzero"));
    }

    #[test]
    fn cooperative_raw_launch_round_trips_if_supported() {
        let device = Device::first().expect("device should be visible");
        if !device
            .supports_cooperative_launch()
            .expect("cooperative launch support should be queryable")
        {
            return;
        }
        let module = device
            .compile_hip_source(
                r#"
extern "C" __global__
void cooperative_probe(unsigned int* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 19;
    }
}
"#,
            )
            .expect("cooperative probe should compile");
        let kernel = module
            .kernel(c"cooperative_probe")
            .expect("cooperative probe kernel should load");
        let output = DeviceBuffer::<u32>::new(1).expect("output allocation should work");
        let stream = Stream::new().expect("stream should be created");
        let mut out_ptr = output.as_mut_ptr();
        let mut params = [crate::__private::arg_ptr(&mut out_ptr)];
        unsafe {
            kernel
                .launch_cooperative_raw_on_stream(
                    &stream,
                    LaunchConfig::new(Dim3::x(1), Dim3::x(64)),
                    &mut params,
                )
                .expect("resident cooperative launch should execute");
        }
        stream.synchronize().expect("stream should finish");
        assert_eq!(output.copy_to_vec().expect("download should work"), [19]);
    }

    #[test]
    fn system_scope_visibility_does_not_promote_device_memory_to_host_concurrent() {
        assert_eq!(
            AtomicMemoryKind::DefaultDevice.system_scope_visibility(),
            SystemScopeAtomicVisibility::DeviceOnly
        );
        assert_eq!(
            AtomicMemoryKind::FineGrainedDevice.system_scope_visibility(),
            SystemScopeAtomicVisibility::DeviceOnly
        );
        assert!(!AtomicMemoryKind::DefaultDevice.allows_host_concurrent_system_scope());
    }

    #[test]
    fn coarse_managed_memory_limits_system_scope_to_synchronization_boundaries() {
        assert_eq!(
            AtomicMemoryKind::ManagedCoarseGrain.system_scope_visibility(),
            SystemScopeAtomicVisibility::HostVisibleAfterSynchronization
        );
        assert!(!AtomicMemoryKind::ManagedCoarseGrain.allows_host_concurrent_system_scope());
        assert!(AtomicMemoryKind::MappedCoherentHost.allows_host_concurrent_system_scope());
    }

    #[test]
    fn host_reference_capture_visibility_matches_memory_visibility() {
        assert_eq!(
            AtomicMemoryKind::DefaultDevice.host_reference_capture_visibility(),
            HostReferenceCaptureVisibility::DeviceOnly
        );
        assert_eq!(
            AtomicMemoryKind::FineGrainedDevice.host_reference_capture_visibility(),
            HostReferenceCaptureVisibility::DeviceOnly
        );
        assert_eq!(
            AtomicMemoryKind::MappedCoherentHost.host_reference_capture_visibility(),
            HostReferenceCaptureVisibility::HostVisibleDuringKernel
        );
        assert_eq!(
            AtomicMemoryKind::ManagedFineGrain.host_reference_capture_visibility(),
            HostReferenceCaptureVisibility::HostVisibleDuringKernel
        );
        assert_eq!(
            AtomicMemoryKind::ManagedCoarseGrain.host_reference_capture_visibility(),
            HostReferenceCaptureVisibility::HostVisibleAfterSynchronization
        );
        assert!(AtomicMemoryKind::MappedCoherentHost.allows_host_reference_capture_during_kernel());
        assert!(
            !AtomicMemoryKind::ManagedCoarseGrain.allows_host_reference_capture_during_kernel()
        );
    }

    #[test]
    fn device_properties_classify_host_visible_memory() {
        let props = DeviceProperties {
            ordinal: 0,
            managed_memory: true,
            concurrent_managed_access: true,
            cooperative_launch: true,
            cooperative_multi_device_launch: false,
            direct_managed_mem_access_from_host: true,
            can_map_host_memory: true,
            can_use_host_pointer_for_registered_mem: true,
            host_native_atomic_supported: true,
            pageable_memory_access: true,
            pageable_memory_access_uses_host_page_tables: true,
            memory_pools_supported: true,
            unified_addressing: true,
            host_register_supported: true,
            async_engine_count: 2,
            multiprocessor_count: 64,
            warp_size: 32,
            clock_instruction_rate_khz: 100_000,
            wall_clock_rate_khz: 100_000,
        };
        assert!(props.has_clock_rate_metadata());
        assert_eq!(
            props.managed_memory_kind(ManagedMemoryKind::FineGrain),
            Some(AtomicMemoryKind::ManagedFineGrain)
        );
        assert_eq!(
            props.managed_memory_kind(ManagedMemoryKind::CoarseGrain),
            Some(AtomicMemoryKind::ManagedCoarseGrain)
        );
        assert_eq!(
            props.mapped_host_memory_kind(),
            Some(AtomicMemoryKind::MappedCoherentHost)
        );
        assert_eq!(
            props.managed_host_reference_capture_kind(ManagedMemoryKind::FineGrain),
            Some(AtomicMemoryKind::ManagedFineGrain)
        );
        assert_eq!(
            props.mapped_host_reference_capture_kind(),
            Some(AtomicMemoryKind::MappedCoherentHost)
        );
    }

    #[test]
    fn device_properties_downgrade_managed_without_concurrent_access() {
        let props = DeviceProperties {
            ordinal: 0,
            managed_memory: true,
            concurrent_managed_access: false,
            cooperative_launch: false,
            cooperative_multi_device_launch: false,
            direct_managed_mem_access_from_host: false,
            can_map_host_memory: false,
            can_use_host_pointer_for_registered_mem: false,
            host_native_atomic_supported: false,
            pageable_memory_access: false,
            pageable_memory_access_uses_host_page_tables: false,
            memory_pools_supported: false,
            unified_addressing: false,
            host_register_supported: false,
            async_engine_count: 0,
            multiprocessor_count: 1,
            warp_size: 32,
            clock_instruction_rate_khz: 0,
            wall_clock_rate_khz: 0,
        };
        assert!(!props.has_clock_rate_metadata());
        assert_eq!(
            props.managed_memory_kind(ManagedMemoryKind::FineGrain),
            Some(AtomicMemoryKind::ManagedCoarseGrain)
        );
        assert_eq!(props.mapped_host_memory_kind(), None);
    }

    #[test]
    fn mapped_host_memory_needs_host_native_atomics_for_concurrent_system_scope() {
        let props = DeviceProperties {
            ordinal: 0,
            managed_memory: true,
            concurrent_managed_access: true,
            cooperative_launch: true,
            cooperative_multi_device_launch: false,
            direct_managed_mem_access_from_host: false,
            can_map_host_memory: true,
            can_use_host_pointer_for_registered_mem: false,
            host_native_atomic_supported: false,
            pageable_memory_access: false,
            pageable_memory_access_uses_host_page_tables: false,
            memory_pools_supported: true,
            unified_addressing: true,
            host_register_supported: true,
            async_engine_count: 2,
            multiprocessor_count: 64,
            warp_size: 32,
            clock_instruction_rate_khz: 100_000,
            wall_clock_rate_khz: 100_000,
        };
        assert_eq!(props.mapped_host_memory_kind(), None);
    }

    #[test]
    fn fine_grain_managed_memory_needs_host_native_atomics_for_concurrent_system_scope() {
        let props = DeviceProperties {
            ordinal: 0,
            managed_memory: true,
            concurrent_managed_access: true,
            cooperative_launch: true,
            cooperative_multi_device_launch: false,
            direct_managed_mem_access_from_host: false,
            can_map_host_memory: true,
            can_use_host_pointer_for_registered_mem: false,
            host_native_atomic_supported: false,
            pageable_memory_access: false,
            pageable_memory_access_uses_host_page_tables: false,
            memory_pools_supported: true,
            unified_addressing: true,
            host_register_supported: true,
            async_engine_count: 2,
            multiprocessor_count: 64,
            warp_size: 32,
            clock_instruction_rate_khz: 100_000,
            wall_clock_rate_khz: 100_000,
        };
        assert_eq!(
            props.managed_memory_kind(ManagedMemoryKind::FineGrain),
            Some(AtomicMemoryKind::ManagedCoarseGrain)
        );
    }

    #[test]
    fn buffer_len_validation_reports_name() {
        let err = super::validate_buffer_len("input", 3, 4).expect_err("short buffer should fail");
        assert!(err.to_string().contains("input"));
    }

    #[test]
    fn disjoint_validation_rejects_same_buffer() {
        let buffer = DeviceBuffer::<u32>::new(4).expect("small allocation should work");
        let err = super::validate_device_buffers_disjoint("out", &buffer, "input", &buffer)
            .expect_err("same buffer should alias");
        assert!(err.to_string().contains("aliases"));
    }

    #[test]
    fn disjoint_validation_accepts_distinct_buffers() {
        let out = DeviceBuffer::<u32>::new(4).expect("small allocation should work");
        let input = DeviceBuffer::<u32>::new(4).expect("small allocation should work");
        super::validate_device_buffers_disjoint("out", &out, "input", &input)
            .expect("distinct allocations should not alias");
    }
}
