use crate::libraries::LibraryAvailability;
use crate::{Error, Result};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

const RTLD_NOW: c_int = 2;

type RocTxVersionFn = unsafe extern "C" fn() -> c_uint;
type RocTxMarkA = unsafe extern "C" fn(*const c_char);
type RocTxRangePushA = unsafe extern "C" fn(*const c_char) -> c_int;
type RocTxRangePop = unsafe extern "C" fn() -> c_int;
type RocTxRangeStartA = unsafe extern "C" fn(*const c_char) -> u64;
type RocTxRangeStop = unsafe extern "C" fn(u64);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocTxVersion {
    pub major: u32,
    pub minor: u32,
}

pub struct RocTx {
    funcs: Arc<RocTxFunctions>,
}

pub struct RocTxScopedRange<'a> {
    roctx: &'a RocTx,
    active: bool,
}

struct RocTxFunctions {
    _lib: Arc<DynamicLibrary>,
    version_major: RocTxVersionFn,
    version_minor: RocTxVersionFn,
    mark: RocTxMarkA,
    range_push: RocTxRangePushA,
    range_pop: RocTxRangePop,
    range_start: RocTxRangeStartA,
    range_stop: RocTxRangeStop,
}

struct DynamicLibrary {
    handle: *mut c_void,
    name: String,
}

unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}
unsafe impl Send for RocTxFunctions {}
unsafe impl Sync for RocTxFunctions {}

impl RocTx {
    pub fn open() -> Result<Self> {
        let lib = Arc::new(DynamicLibrary::open(&library_candidates(&[
            "libroctx64.so",
            "libroctx64.so.4",
            "libroctx64.so.4.1.0",
        ]))?);
        let funcs = unsafe { RocTxFunctions::load(lib)? };
        Ok(Self {
            funcs: Arc::new(funcs),
        })
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    pub fn availability() -> LibraryAvailability {
        match Self::open() {
            Ok(roctx) => {
                let version = roctx.version();
                LibraryAvailability {
                    available: true,
                    detail: format!(
                        "loaded {} version {}.{}",
                        roctx.funcs._lib.name, version.major, version.minor
                    ),
                }
            }
            Err(err) => LibraryAvailability {
                available: false,
                detail: err.to_string(),
            },
        }
    }

    pub fn version(&self) -> RocTxVersion {
        let major = unsafe { (self.funcs.version_major)() };
        let minor = unsafe { (self.funcs.version_minor)() };
        RocTxVersion { major, minor }
    }

    pub fn mark(&self, message: impl AsRef<str>) -> Result<()> {
        let message = c_message(message.as_ref())?;
        unsafe {
            (self.funcs.mark)(message.as_ptr());
        }
        Ok(())
    }

    pub fn range_push(&self, message: impl AsRef<str>) -> Result<i32> {
        let message = c_message(message.as_ref())?;
        Ok(unsafe { (self.funcs.range_push)(message.as_ptr()) })
    }

    pub fn range_pop(&self) -> i32 {
        unsafe { (self.funcs.range_pop)() }
    }

    pub fn scoped_range(&self, message: impl AsRef<str>) -> Result<RocTxScopedRange<'_>> {
        self.range_push(message)?;
        Ok(RocTxScopedRange {
            roctx: self,
            active: true,
        })
    }

    pub fn range_start(&self, message: impl AsRef<str>) -> Result<u64> {
        let message = c_message(message.as_ref())?;
        Ok(unsafe { (self.funcs.range_start)(message.as_ptr()) })
    }

    pub fn range_stop(&self, id: u64) {
        unsafe {
            (self.funcs.range_stop)(id);
        }
    }
}

impl Drop for RocTxScopedRange<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.roctx.range_pop();
            self.active = false;
        }
    }
}

impl RocTxFunctions {
    unsafe fn load(lib: Arc<DynamicLibrary>) -> Result<Self> {
        Ok(Self {
            version_major: unsafe { lib.symbol(c"roctx_version_major")? },
            version_minor: unsafe { lib.symbol(c"roctx_version_minor")? },
            mark: unsafe { lib.symbol(c"roctxMarkA")? },
            range_push: unsafe { lib.symbol(c"roctxRangePushA")? },
            range_pop: unsafe { lib.symbol(c"roctxRangePop")? },
            range_start: unsafe { lib.symbol(c"roctxRangeStartA")? },
            range_stop: unsafe { lib.symbol(c"roctxRangeStop")? },
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
            "could not load rocTX library from candidates [{}]",
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

fn c_message(message: &str) -> Result<CString> {
    CString::new(message)
        .map_err(|_| Error::Library("rocTX message contains a NUL byte".to_string()))
}

fn library_candidates(names: &[&str]) -> Vec<String> {
    let mut candidates = Vec::new();
    for name in names {
        candidates.push((*name).to_string());
    }
    let rocm_path = rocm_path();
    for name in names {
        let candidate = rocm_path.join("lib").join(name);
        let candidate = candidate.to_string_lossy().into_owned();
        if !candidates.iter().any(|existing| existing == &candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn rocm_path() -> PathBuf {
    std::env::var_os("ROCM_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
}

fn dl_error_string() -> String {
    let err = unsafe { dlerror() };
    if err.is_null() {
        "unknown dlopen/dlsym error".to_string()
    } else {
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roctx_availability_is_queryable() {
        let availability = RocTx::availability();
        assert!(!availability.detail.is_empty());
    }

    #[test]
    fn roctx_marker_smoke_if_available() {
        let Ok(roctx) = RocTx::open() else {
            return;
        };
        roctx.mark("rocm-oxide test marker").unwrap();
        {
            let _range = roctx.scoped_range("rocm-oxide scoped range").unwrap();
        }
        let id = roctx.range_start("rocm-oxide process range").unwrap();
        roctx.range_stop(id);
    }

    #[test]
    fn roctx_rejects_nul_messages_if_available() {
        let Ok(roctx) = RocTx::open() else {
            return;
        };
        let err = roctx
            .mark("bad\0message")
            .expect_err("interior NUL should be rejected before rocTX call");
        assert!(err.to_string().contains("NUL"));
    }
}
