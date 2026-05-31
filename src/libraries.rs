use crate::hip::{DeviceBuffer, Stream};
use crate::{Error, Result, validate_buffer_len};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

const RTLD_NOW: c_int = 2;
const ROCBLAS_STATUS_SUCCESS: c_int = 0;
const ROCBLAS_OPERATION_NONE: c_int = 111;
const ROCFFT_STATUS_SUCCESS: c_int = 0;

type RocBlasStatus = c_int;
type RocBlasOperation = c_int;
type RocBlasHandleRaw = *mut c_void;
type RocBlasCreateHandle = unsafe extern "C" fn(*mut RocBlasHandleRaw) -> RocBlasStatus;
type RocBlasDestroyHandle = unsafe extern "C" fn(RocBlasHandleRaw) -> RocBlasStatus;
type RocBlasSetStream = unsafe extern "C" fn(RocBlasHandleRaw, *mut c_void) -> RocBlasStatus;
type RocBlasSgemm = unsafe extern "C" fn(
    RocBlasHandleRaw,
    RocBlasOperation,
    RocBlasOperation,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> RocBlasStatus;

type RocFftStatus = c_int;
type RocFftPlanRaw = *mut c_void;
type RocFftPlanDescriptionRaw = *mut c_void;
type RocFftExecutionInfoRaw = *mut c_void;
type RocFftSetup = unsafe extern "C" fn() -> RocFftStatus;
type RocFftCleanup = unsafe extern "C" fn() -> RocFftStatus;
type RocFftPlanCreate = unsafe extern "C" fn(
    *mut RocFftPlanRaw,
    c_int,
    c_int,
    c_int,
    usize,
    *const usize,
    usize,
    RocFftPlanDescriptionRaw,
) -> RocFftStatus;
type RocFftPlanDestroy = unsafe extern "C" fn(RocFftPlanRaw) -> RocFftStatus;
type RocFftExecute = unsafe extern "C" fn(
    RocFftPlanRaw,
    *mut *mut c_void,
    *mut *mut c_void,
    RocFftExecutionInfoRaw,
) -> RocFftStatus;
type RocFftExecutionInfoCreate = unsafe extern "C" fn(*mut RocFftExecutionInfoRaw) -> RocFftStatus;
type RocFftExecutionInfoDestroy = unsafe extern "C" fn(RocFftExecutionInfoRaw) -> RocFftStatus;
type RocFftExecutionInfoSetStream =
    unsafe extern "C" fn(RocFftExecutionInfoRaw, *mut c_void) -> RocFftStatus;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryAvailability {
    pub available: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmLibraryReport {
    pub rocblas: LibraryAvailability,
    pub rocfft: LibraryAvailability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SgemmLayout {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocFftComplexDirection {
    Forward = 0,
    Inverse = 1,
}

pub struct RocBlas {
    funcs: Arc<RocBlasFunctions>,
}

pub struct RocBlasHandle {
    raw: RocBlasHandleRaw,
    funcs: Arc<RocBlasFunctions>,
}

pub struct RocFft {
    funcs: Arc<RocFftFunctions>,
}

pub struct RocFftSession {
    inner: Arc<RocFftSessionInner>,
}

pub struct RocFftPlan {
    raw: RocFftPlanRaw,
    required_complex_elements: usize,
    inner: Arc<RocFftSessionInner>,
}

pub struct RocFftExecutionInfo {
    raw: RocFftExecutionInfoRaw,
    inner: Arc<RocFftSessionInner>,
}

struct RocBlasFunctions {
    _lib: Arc<DynamicLibrary>,
    create_handle: RocBlasCreateHandle,
    destroy_handle: RocBlasDestroyHandle,
    set_stream: RocBlasSetStream,
    sgemm: RocBlasSgemm,
}

struct RocFftFunctions {
    _lib: Arc<DynamicLibrary>,
    setup: RocFftSetup,
    cleanup: RocFftCleanup,
    plan_create: RocFftPlanCreate,
    plan_destroy: RocFftPlanDestroy,
    execute: RocFftExecute,
    execution_info_create: RocFftExecutionInfoCreate,
    execution_info_destroy: RocFftExecutionInfoDestroy,
    execution_info_set_stream: RocFftExecutionInfoSetStream,
}

struct RocFftSessionInner {
    funcs: Arc<RocFftFunctions>,
}

struct DynamicLibrary {
    handle: *mut c_void,
    name: String,
}

unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}
unsafe impl Send for RocBlasFunctions {}
unsafe impl Sync for RocBlasFunctions {}
unsafe impl Send for RocFftFunctions {}
unsafe impl Sync for RocFftFunctions {}
unsafe impl Send for RocBlasHandle {}
unsafe impl Sync for RocBlasHandle {}
unsafe impl Send for RocFftSessionInner {}
unsafe impl Sync for RocFftSessionInner {}
unsafe impl Send for RocFftPlan {}
unsafe impl Sync for RocFftPlan {}
unsafe impl Send for RocFftExecutionInfo {}
unsafe impl Sync for RocFftExecutionInfo {}

impl LibraryAvailability {
    fn available(detail: impl Into<String>) -> Self {
        Self {
            available: true,
            detail: detail.into(),
        }
    }

    fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            available: false,
            detail: detail.into(),
        }
    }
}

impl RocmLibraryReport {
    pub fn query() -> Self {
        Self {
            rocblas: match RocBlas::open() {
                Ok(blas) => {
                    LibraryAvailability::available(format!("loaded {}", blas.funcs._lib.name))
                }
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
            rocfft: match RocFft::open() {
                Ok(fft) => {
                    LibraryAvailability::available(format!("loaded {}", fft.funcs._lib.name))
                }
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
        }
    }
}

impl SgemmLayout {
    pub fn column_major(m: u32, n: u32, k: u32) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(Error::Library(
                "SGEMM dimensions m, n, and k must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            m,
            n,
            k,
            lda: m,
            ldb: k,
            ldc: m,
        })
    }

    fn validate(self, a_len: usize, b_len: usize, c_len: usize) -> Result<()> {
        if self.lda < self.m || self.ldb < self.k || self.ldc < self.m {
            return Err(Error::Library(format!(
                "invalid SGEMM leading dimensions lda={}, ldb={}, ldc={} for m={}, n={}, k={}",
                self.lda, self.ldb, self.ldc, self.m, self.n, self.k
            )));
        }
        validate_buffer_len("rocBLAS SGEMM A", a_len, matrix_elements(self.lda, self.k)?)?;
        validate_buffer_len("rocBLAS SGEMM B", b_len, matrix_elements(self.ldb, self.n)?)?;
        validate_buffer_len("rocBLAS SGEMM C", c_len, matrix_elements(self.ldc, self.n)?)?;
        Ok(())
    }
}

impl RocBlas {
    pub fn open() -> Result<Self> {
        let lib = Arc::new(DynamicLibrary::open(&library_candidates(&[
            "librocblas.so",
            "librocblas.so.5",
        ]))?);
        let funcs = unsafe { RocBlasFunctions::load(lib)? };
        Ok(Self {
            funcs: Arc::new(funcs),
        })
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    pub fn create_handle(&self) -> Result<RocBlasHandle> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_rocblas(
                (self.funcs.create_handle)(&mut raw),
                "rocblas_create_handle",
            )?;
        }
        Ok(RocBlasHandle {
            raw,
            funcs: Arc::clone(&self.funcs),
        })
    }
}

impl RocBlasHandle {
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe {
            check_rocblas(
                (self.funcs.set_stream)(self.raw, stream.as_raw()),
                "rocblas_set_stream",
            )
        }
    }

    pub fn sgemm_nn(
        &self,
        layout: SgemmLayout,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        c: &DeviceBuffer<f32>,
    ) -> Result<()> {
        layout.validate(a.len(), b.len(), c.len())?;
        unsafe {
            check_rocblas(
                (self.funcs.sgemm)(
                    self.raw,
                    ROCBLAS_OPERATION_NONE,
                    ROCBLAS_OPERATION_NONE,
                    c_int_from_u32("SGEMM m", layout.m)?,
                    c_int_from_u32("SGEMM n", layout.n)?,
                    c_int_from_u32("SGEMM k", layout.k)?,
                    &alpha,
                    a.as_ptr(),
                    c_int_from_u32("SGEMM lda", layout.lda)?,
                    b.as_ptr(),
                    c_int_from_u32("SGEMM ldb", layout.ldb)?,
                    &beta,
                    c.as_mut_ptr(),
                    c_int_from_u32("SGEMM ldc", layout.ldc)?,
                ),
                "rocblas_sgemm",
            )
        }
    }
}

impl Drop for RocBlasHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.funcs.destroy_handle)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl RocFft {
    pub fn open() -> Result<Self> {
        let lib = Arc::new(DynamicLibrary::open(&library_candidates(&[
            "librocfft.so",
            "librocfft.so.0",
        ]))?);
        let funcs = unsafe { RocFftFunctions::load(lib)? };
        Ok(Self {
            funcs: Arc::new(funcs),
        })
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    pub fn setup(&self) -> Result<RocFftSession> {
        unsafe {
            check_rocfft((self.funcs.setup)(), "rocfft_setup")?;
        }
        Ok(RocFftSession {
            inner: Arc::new(RocFftSessionInner {
                funcs: Arc::clone(&self.funcs),
            }),
        })
    }
}

impl RocFftSession {
    pub fn create_1d_complex_f32_plan(
        &self,
        length: usize,
        batch: usize,
        direction: RocFftComplexDirection,
    ) -> Result<RocFftPlan> {
        if length == 0 || batch == 0 {
            return Err(Error::Library(
                "rocFFT length and batch must be nonzero".to_string(),
            ));
        }
        let required_complex_elements = length.checked_mul(batch).ok_or_else(|| {
            Error::Library("rocFFT plan element count overflows usize".to_string())
        })?;
        let lengths = [length];
        let mut raw = ptr::null_mut();
        unsafe {
            check_rocfft(
                (self.inner.funcs.plan_create)(
                    &mut raw,
                    0,
                    direction as c_int,
                    0,
                    1,
                    lengths.as_ptr(),
                    batch,
                    ptr::null_mut(),
                ),
                "rocfft_plan_create",
            )?;
        }
        Ok(RocFftPlan {
            raw,
            required_complex_elements,
            inner: Arc::clone(&self.inner),
        })
    }

    pub fn execution_info_for_stream(&self, stream: &Stream) -> Result<RocFftExecutionInfo> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_rocfft(
                (self.inner.funcs.execution_info_create)(&mut raw),
                "rocfft_execution_info_create",
            )?;
            if let Err(err) = check_rocfft(
                (self.inner.funcs.execution_info_set_stream)(raw, stream.as_raw()),
                "rocfft_execution_info_set_stream",
            ) {
                let _ = (self.inner.funcs.execution_info_destroy)(raw);
                return Err(err);
            }
        }
        Ok(RocFftExecutionInfo {
            raw,
            inner: Arc::clone(&self.inner),
        })
    }
}

impl Drop for RocFftSessionInner {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.funcs.cleanup)();
        }
    }
}

impl RocFftPlan {
    pub fn execute_in_place_complex_f32(
        &self,
        input: &DeviceBuffer<[f32; 2]>,
        info: Option<&RocFftExecutionInfo>,
    ) -> Result<()> {
        validate_buffer_len(
            "rocFFT complex input",
            input.len(),
            self.required_complex_elements,
        )?;
        let input_buffer = input.as_mut_ptr().cast::<c_void>();
        let mut input_buffers = [input_buffer];
        let info = info.map_or(ptr::null_mut(), |info| info.raw);
        unsafe {
            check_rocfft(
                (self.inner.funcs.execute)(
                    self.raw,
                    input_buffers.as_mut_ptr(),
                    ptr::null_mut(),
                    info,
                ),
                "rocfft_execute",
            )
        }
    }
}

impl Drop for RocFftPlan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.inner.funcs.plan_destroy)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl Drop for RocFftExecutionInfo {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.inner.funcs.execution_info_destroy)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl RocBlasFunctions {
    unsafe fn load(lib: Arc<DynamicLibrary>) -> Result<Self> {
        Ok(Self {
            create_handle: unsafe { lib.symbol(c"rocblas_create_handle")? },
            destroy_handle: unsafe { lib.symbol(c"rocblas_destroy_handle")? },
            set_stream: unsafe { lib.symbol(c"rocblas_set_stream")? },
            sgemm: unsafe { lib.symbol(c"rocblas_sgemm")? },
            _lib: lib,
        })
    }
}

impl RocFftFunctions {
    unsafe fn load(lib: Arc<DynamicLibrary>) -> Result<Self> {
        Ok(Self {
            setup: unsafe { lib.symbol(c"rocfft_setup")? },
            cleanup: unsafe { lib.symbol(c"rocfft_cleanup")? },
            plan_create: unsafe { lib.symbol(c"rocfft_plan_create")? },
            plan_destroy: unsafe { lib.symbol(c"rocfft_plan_destroy")? },
            execute: unsafe { lib.symbol(c"rocfft_execute")? },
            execution_info_create: unsafe { lib.symbol(c"rocfft_execution_info_create")? },
            execution_info_destroy: unsafe { lib.symbol(c"rocfft_execution_info_destroy")? },
            execution_info_set_stream: unsafe { lib.symbol(c"rocfft_execution_info_set_stream")? },
            _lib: lib,
        })
    }
}

impl DynamicLibrary {
    fn open(candidates: &[String]) -> Result<Self> {
        let mut failures = Vec::new();
        for candidate in candidates {
            let name = CString::new(candidate.as_str()).map_err(|_| {
                Error::Library(format!(
                    "library candidate `{candidate}` contains a NUL byte"
                ))
            })?;
            let handle = unsafe {
                let _ = dlerror();
                dlopen(name.as_ptr(), RTLD_NOW)
            };
            if !handle.is_null() {
                return Ok(Self {
                    handle,
                    name: candidate.clone(),
                });
            }
            failures.push(format!("{candidate}: {}", dl_error_string()));
        }
        Err(Error::Library(format!(
            "could not load any candidate [{}]",
            failures.join("; ")
        )))
    }

    unsafe fn symbol<T: Copy>(&self, name: &CStr) -> Result<T> {
        if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
            return Err(Error::Library(format!(
                "symbol `{}` has unsupported function-pointer size",
                name.to_string_lossy()
            )));
        }
        let ptr = unsafe {
            let _ = dlerror();
            dlsym(self.handle, name.as_ptr())
        };
        if ptr.is_null() {
            return Err(Error::Library(format!(
                "missing symbol `{}` in {}: {}",
                name.to_string_lossy(),
                self.name,
                dl_error_string()
            )));
        }
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = dlclose(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

fn library_candidates(names: &[&str]) -> Vec<String> {
    let mut candidates = Vec::new();
    for name in names {
        candidates.push((*name).to_string());
    }
    let rocm_path = std::env::var_os("ROCM_PATH")
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/opt/rocm"));
    for name in names {
        let candidate = rocm_path.join("lib").join(name);
        let candidate = candidate.to_string_lossy().into_owned();
        if !candidates.iter().any(|existing| existing == &candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn check_rocblas(status: RocBlasStatus, op: &str) -> Result<()> {
    if status == ROCBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Library(format!(
            "{op} returned rocBLAS status {status}"
        )))
    }
}

fn check_rocfft(status: RocFftStatus, op: &str) -> Result<()> {
    if status == ROCFFT_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Library(format!(
            "{op} returned rocFFT status {status}"
        )))
    }
}

fn matrix_elements(leading_dim: u32, columns: u32) -> Result<usize> {
    (leading_dim as usize)
        .checked_mul(columns as usize)
        .ok_or_else(|| Error::Library("matrix element count overflows usize".to_string()))
}

fn c_int_from_u32(label: &str, value: u32) -> Result<c_int> {
    c_int::try_from(value)
        .map_err(|_| Error::Library(format!("{label} value {value} exceeds rocBLAS int range")))
}

fn dl_error_string() -> String {
    let ptr = unsafe { dlerror() };
    if ptr.is_null() {
        "unknown dynamic linker error".to_string()
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_layout_rejects_zero_dimensions() {
        assert!(SgemmLayout::column_major(0, 1, 1).is_err());
    }

    #[test]
    fn sgemm_layout_validates_buffer_lengths() {
        let layout = SgemmLayout::column_major(2, 3, 4).unwrap();
        assert!(layout.validate(8, 12, 6).is_ok());
        assert!(layout.validate(7, 12, 6).is_err());
        assert!(layout.validate(8, 11, 6).is_err());
        assert!(layout.validate(8, 12, 5).is_err());
    }

    #[test]
    fn library_report_can_be_queried() {
        let report = RocmLibraryReport::query();
        assert!(!report.rocblas.detail.is_empty());
        assert!(!report.rocfft.detail.is_empty());
    }

    #[test]
    fn rocblas_handle_smoke_if_library_is_available() {
        let Ok(blas) = RocBlas::open() else {
            return;
        };
        let _handle = blas
            .create_handle()
            .expect("available rocBLAS library should create a handle");
    }

    #[test]
    fn rocfft_setup_smoke_if_library_is_available() {
        let Ok(fft) = RocFft::open() else {
            return;
        };
        let _session = fft
            .setup()
            .expect("available rocFFT library should initialize");
    }
}
