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
}

impl Device {
    pub fn first() -> Result<Self> {
        let count = hip::device_count()?;
        if count == 0 {
            return Err(Error::NoDevice);
        }

        hip::set_device(0)?;
        let arch = detect_arch().ok_or(Error::MissingArchitecture)?;
        Ok(Self { ordinal: 0, arch })
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn compile_hip_source(&self, source: &str) -> Result<Module> {
        let code_object = hiprtc::compile_code_object(source, &self.arch)?;
        self.load_code_object(&code_object)
    }

    pub fn load_code_object(&self, code_object: &[u8]) -> Result<Module> {
        let module = hip::Module::from_code_object(&code_object)?;
        Ok(Module { module })
    }

    pub fn load_code_object_file(&self, path: impl AsRef<Path>) -> Result<Module> {
        let code_object = std::fs::read(path)?;
        self.load_code_object(&code_object)
    }

    pub fn execution_context(&self) -> Result<crate::ExecutionContext> {
        crate::ExecutionContext::new(self)
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

    pub fn for_num_elems(num_elems: usize) -> Self {
        Self::for_num_elems_with_block_size(num_elems, Self::DEFAULT_BLOCK_X)
    }

    pub fn for_num_elems_with_block_size(num_elems: usize, block_x: u32) -> Self {
        let grid_x = (num_elems as u32).div_ceil(block_x);
        Self::new(Dim3::x(grid_x), Dim3::x(block_x))
    }

    pub fn for_2d(width: u32, height: u32, block_x: u32, block_y: u32) -> Self {
        let grid_x = width.div_ceil(block_x);
        let grid_y = height.div_ceil(block_y);
        Self::new(Dim3::new(grid_x, grid_y, 1), Dim3::new(block_x, block_y, 1))
    }

    pub const fn with_shared_mem_bytes(mut self, shared_mem_bytes: u32) -> Self {
        self.shared_mem_bytes = shared_mem_bytes;
        self
    }
}

pub fn validate_launch_config(config: LaunchConfig) -> Result<()> {
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
    if block_threads > 1024 {
        return Err(Error::InvalidLaunch(format!(
            "block has {block_threads} threads, maximum supported by this prototype is 1024"
        )));
    }

    Ok(())
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
}

impl Module {
    pub fn kernel(&self, name: &CStr) -> Result<Kernel> {
        Ok(Kernel {
            function: self.module.function(name)?,
        })
    }

    pub fn global<T>(&self, name: &CStr) -> Result<hip::Global<T>> {
        Ok(self.module.global(name)?)
    }
}

pub struct Kernel {
    function: hip::Function,
}

impl Kernel {
    pub unsafe fn launch_raw(
        &self,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        Ok(unsafe {
            self.function.launch(
                config.grid.as_tuple(),
                config.block.as_tuple(),
                config.shared_mem_bytes,
                params,
            )?;
        })
    }

    pub unsafe fn launch_raw_on_stream(
        &self,
        stream: &hip::Stream,
        config: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        Ok(unsafe {
            self.function.launch_on_stream(
                config.grid.as_tuple(),
                config.block.as_tuple(),
                config.shared_mem_bytes,
                stream.as_raw(),
                params,
            )?;
        })
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
    use super::{Dim3, LaunchConfig};
    use crate::hip::DeviceBuffer;

    #[test]
    fn one_dimensional_launch_config_rounds_up() {
        let config = LaunchConfig::for_num_elems(1_025);
        assert_eq!(config.grid, Dim3::x(5));
        assert_eq!(config.block, Dim3::x(256));
        assert_eq!(config.shared_mem_bytes, 0);
    }

    #[test]
    fn custom_one_dimensional_block_size_rounds_up() {
        let config = LaunchConfig::for_num_elems_with_block_size(1_025, 128);
        assert_eq!(config.grid, Dim3::x(9));
        assert_eq!(config.block, Dim3::x(128));
    }

    #[test]
    fn two_dimensional_launch_config_rounds_up() {
        let config = LaunchConfig::for_2d(1_025, 513, 16, 16);
        assert_eq!(config.grid, Dim3::new(65, 33, 1));
        assert_eq!(config.block, Dim3::new(16, 16, 1));
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
