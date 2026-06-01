use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub type HiprtcResult = c_int;
pub type HiprtcProgram = *mut std::ffi::c_void;

pub const HIPRTC_SUCCESS: HiprtcResult = 0;

unsafe extern "C" {
    fn hiprtcGetErrorString(result: HiprtcResult) -> *const c_char;
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcCompileProgram(
        prog: HiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size: *mut usize) -> HiprtcResult;
    fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut c_char) -> HiprtcResult;
    fn hiprtcGetCodeSize(prog: HiprtcProgram, code_size: *mut usize) -> HiprtcResult;
    fn hiprtcGetCode(prog: HiprtcProgram, code: *mut c_char) -> HiprtcResult;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> HiprtcResult;
}

#[derive(Debug, Clone)]
pub struct Error {
    code: HiprtcResult,
    message: String,
    log: Option<String>,
}

impl Error {
    fn from_code(code: HiprtcResult, log: Option<String>) -> Self {
        let message = unsafe {
            let ptr = hiprtcGetErrorString(code);
            if ptr.is_null() {
                format!("HIPRTC error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Self { code, message, log }
    }

    fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            code: -1,
            message: message.into(),
            log: None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(log) = &self.log {
            write!(f, "{} ({})\n{}", self.message, self.code, log)
        } else {
            write!(f, "{} ({})", self.message, self.code)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecializationBackend {
    Hiprtc,
    Comgr,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecializationCacheKey {
    pub backend: SpecializationBackend,
    pub arch: String,
    pub source_hash: u64,
    pub options_hash: u64,
    pub launch_metadata_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecializationCacheStats {
    pub entries: usize,
    pub hits: usize,
    pub misses: usize,
}

#[derive(Default)]
pub struct SpecializationCache {
    entries: Mutex<HashMap<SpecializationCacheKey, Arc<Vec<u8>>>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl SpecializationCacheKey {
    pub fn new(source: &str, arch: &str, options: &[String], launch_metadata: &str) -> Self {
        Self::with_backend(
            SpecializationBackend::Hiprtc,
            source,
            arch,
            options,
            launch_metadata,
        )
    }

    pub fn with_backend(
        backend: SpecializationBackend,
        source: &str,
        arch: &str,
        options: &[String],
        launch_metadata: &str,
    ) -> Self {
        Self {
            backend,
            arch: arch.to_string(),
            source_hash: stable_hash(source.as_bytes()),
            options_hash: stable_hash_strings(options.iter().map(String::as_str)),
            launch_metadata_hash: stable_hash(launch_metadata.as_bytes()),
        }
    }

    pub fn persistent_filename(&self) -> String {
        format!(
            "{}-{}-{:016x}-{:016x}-{:016x}.hsaco",
            self.backend.as_str(),
            sanitize_cache_component(&self.arch),
            self.source_hash,
            self.options_hash,
            self.launch_metadata_hash
        )
    }
}

impl SpecializationBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hiprtc => "hiprtc",
            Self::Comgr => "comgr",
        }
    }
}

impl SpecializationCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_compile<F>(&self, key: SpecializationCacheKey, compile: F) -> Result<Arc<Vec<u8>>>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        if let Some(code_object) = self
            .entries
            .lock()
            .expect("HIPRTC specialization cache mutex poisoned")
            .get(&key)
            .cloned()
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(code_object);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let compiled = Arc::new(compile()?);
        let mut entries = self
            .entries
            .lock()
            .expect("HIPRTC specialization cache mutex poisoned");
        let entry = entries.entry(key).or_insert_with(|| Arc::clone(&compiled));
        Ok(Arc::clone(entry))
    }

    pub fn stats(&self) -> SpecializationCacheStats {
        SpecializationCacheStats {
            entries: self
                .entries
                .lock()
                .expect("HIPRTC specialization cache mutex poisoned")
                .len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    pub fn clear(&self) {
        self.entries
            .lock()
            .expect("HIPRTC specialization cache mutex poisoned")
            .clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

fn check(code: HiprtcResult) -> Result<()> {
    if code == HIPRTC_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_code(code, None))
    }
}

struct Program {
    raw: HiprtcProgram,
}

impl Program {
    fn new(src: &CStr, name: &CStr) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check(hiprtcCreateProgram(
                &mut raw,
                src.as_ptr(),
                name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            ))?;
        }
        Ok(Self { raw })
    }

    fn log(&self) -> String {
        let mut size = 0usize;
        unsafe {
            if hiprtcGetProgramLogSize(self.raw, &mut size) != HIPRTC_SUCCESS || size == 0 {
                return String::new();
            }

            let mut bytes = vec![0u8; size];
            if hiprtcGetProgramLog(self.raw, bytes.as_mut_ptr().cast::<c_char>()) != HIPRTC_SUCCESS
            {
                return String::new();
            }
            CStr::from_ptr(bytes.as_ptr().cast::<c_char>())
                .to_string_lossy()
                .into_owned()
        }
    }

    fn compile(&self, options: &[CString]) -> Result<()> {
        let option_ptrs = options.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let code = unsafe {
            hiprtcCompileProgram(self.raw, option_ptrs.len() as c_int, option_ptrs.as_ptr())
        };
        if code == HIPRTC_SUCCESS {
            Ok(())
        } else {
            Err(Error::from_code(code, Some(self.log())))
        }
    }

    fn code(&self) -> Result<Vec<u8>> {
        let mut size = 0usize;
        unsafe {
            check(hiprtcGetCodeSize(self.raw, &mut size))?;
            let mut bytes = vec![0u8; size];
            check(hiprtcGetCode(self.raw, bytes.as_mut_ptr().cast::<c_char>()))?;
            Ok(bytes)
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hiprtcDestroyProgram(&mut self.raw);
            }
        }
    }
}

pub fn compile_code_object(source: &str, arch: &str) -> Result<Vec<u8>> {
    compile_code_object_with_options(source, arch, &[])
}

pub fn compile_code_object_with_options(
    source: &str,
    arch: &str,
    extra_options: &[&str],
) -> Result<Vec<u8>> {
    let options = compile_options(arch, extra_options);
    compile_code_object_with_resolved_options(source, &options)
}

pub fn compile_code_object_cached(source: &str, arch: &str) -> Result<Arc<Vec<u8>>> {
    compile_code_object_cached_with_metadata(source, arch, &[], "")
}

pub fn compile_code_object_cached_comgr(source: &str, arch: &str) -> Result<Arc<Vec<u8>>> {
    compile_code_object_cached_comgr_with_metadata(source, arch, &[], "")
}

pub fn compile_code_object_cached_with_metadata(
    source: &str,
    arch: &str,
    extra_options: &[&str],
    launch_metadata: &str,
) -> Result<Arc<Vec<u8>>> {
    let options = compile_options(arch, extra_options);
    let key = SpecializationCacheKey::new(source, arch, &options, launch_metadata);
    global_specialization_cache().get_or_compile(key, || {
        compile_code_object_with_resolved_options(source, &options)
    })
}

pub fn compile_code_object_cached_comgr_with_metadata(
    source: &str,
    arch: &str,
    extra_options: &[&str],
    launch_metadata: &str,
) -> Result<Arc<Vec<u8>>> {
    let options = comgr_compile_options(extra_options);
    let key = SpecializationCacheKey::with_backend(
        SpecializationBackend::Comgr,
        source,
        arch,
        &options,
        launch_metadata,
    );
    global_specialization_cache().get_or_compile(key.clone(), || {
        persistent_code_object_cache_get_or_compile(&key, || {
            compile_code_object_comgr_with_resolved_options(source, arch, &options)
        })
    })
}

pub fn specialization_cache_stats() -> SpecializationCacheStats {
    global_specialization_cache().stats()
}

pub fn clear_specialization_cache() {
    global_specialization_cache().clear();
}

pub fn global_specialization_cache() -> &'static SpecializationCache {
    static CACHE: OnceLock<SpecializationCache> = OnceLock::new();
    CACHE.get_or_init(SpecializationCache::new)
}

fn compile_options(arch: &str, extra_options: &[&str]) -> Vec<String> {
    let mut options = vec![
        format!("--gpu-architecture={arch}"),
        "-std=c++17".to_string(),
        "-O3".to_string(),
    ];
    options.extend(extra_options.iter().map(|option| (*option).to_string()));
    options
}

fn comgr_compile_options(extra_options: &[&str]) -> Vec<String> {
    let mut options = vec![
        "-x".to_string(),
        "hip".to_string(),
        "-std=c++17".to_string(),
        "-O3".to_string(),
        "-Wno-macro-redefined".to_string(),
        format!("-I{}", rocm_path().join("include").display()),
    ];
    options.extend(extra_options.iter().map(|option| (*option).to_string()));
    options
}

fn rocm_path() -> PathBuf {
    std::env::var_os("ROCM_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
}

fn compile_code_object_with_resolved_options(source: &str, options: &[String]) -> Result<Vec<u8>> {
    let source = CString::new(source)
        .map_err(|_| Error::invalid_input("kernel source contained a NUL byte"))?;
    let name = c"kernel.hip";
    let program = Program::new(&source, name)?;
    let options = options
        .iter()
        .map(|option| {
            CString::new(option.as_str()).map_err(|_| {
                Error::invalid_input(format!("HIPRTC option `{option}` contained a NUL byte"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    program.compile(&options)?;
    program.code()
}

fn compile_code_object_comgr_with_resolved_options(
    source: &str,
    arch: &str,
    options: &[String],
) -> Result<Vec<u8>> {
    let comgr = crate::libraries::Comgr::open()
        .map_err(|err| Error::invalid_input(format!("COMGR backend unavailable: {err}")))?;
    comgr
        .compile_hip_source_to_code_object_with_options(source, arch, options)
        .map_err(|err| Error::invalid_input(format!("COMGR backend failed: {err}")))
}

fn persistent_code_object_cache_get_or_compile<F>(
    key: &SpecializationCacheKey,
    compile: F,
) -> Result<Vec<u8>>
where
    F: FnOnce() -> Result<Vec<u8>>,
{
    let path = persistent_code_object_cache_path(key);
    if let Ok(bytes) = fs::read(&path) {
        if bytes.starts_with(b"\x7fELF") {
            return Ok(bytes);
        }
    }

    let bytes = compile()?;
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_ok() {
            let temp_path = path.with_extension("tmp");
            if fs::write(&temp_path, &bytes).is_ok() {
                let _ = fs::rename(temp_path, path);
            }
        }
    }
    Ok(bytes)
}

pub fn persistent_code_object_cache_path(key: &SpecializationCacheKey) -> PathBuf {
    persistent_code_object_cache_dir().join(key.persistent_filename())
}

pub fn persistent_code_object_cache_dir() -> PathBuf {
    if let Some(path) =
        std::env::var_os("ROCM_OXIDE_CODE_OBJECT_CACHE_DIR").filter(|value| !value.is_empty())
    {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(path).join("rocm-oxide").join("code-objects");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home)
            .join(".cache")
            .join("rocm-oxide")
            .join("code-objects");
    }
    PathBuf::from("target")
        .join("rocm-oxide-cache")
        .join("code-objects")
}

fn stable_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn stable_hash_strings<'a>(strings: impl Iterator<Item = &'a str>) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for string in strings {
        for byte in string.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn sanitize_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        SpecializationBackend, SpecializationCache, SpecializationCacheKey,
        persistent_code_object_cache_path,
    };
    use std::sync::Arc;

    #[test]
    fn specialization_cache_key_changes_with_metadata() {
        let options = vec!["--gpu-architecture=gfx1100".to_string(), "-O3".to_string()];
        let a = SpecializationCacheKey::new(
            "extern \"C\" __global__ void k() {}",
            "gfx1100",
            &options,
            "block=128",
        );
        let b = SpecializationCacheKey::new(
            "extern \"C\" __global__ void k() {}",
            "gfx1100",
            &options,
            "block=256",
        );
        assert_ne!(a, b);
    }

    #[test]
    fn specialization_cache_key_changes_with_backend() {
        let options = vec!["--gpu-architecture=gfx1100".to_string(), "-O3".to_string()];
        let hiprtc = SpecializationCacheKey::new("source", "gfx1100", &options, "meta");
        let comgr = SpecializationCacheKey::with_backend(
            SpecializationBackend::Comgr,
            "source",
            "gfx1100",
            &options,
            "meta",
        );
        assert_ne!(hiprtc, comgr);
    }

    #[test]
    fn persistent_cache_path_separates_backend_and_arch() {
        let options = vec!["-O3".to_string()];
        let key = SpecializationCacheKey::with_backend(
            SpecializationBackend::Comgr,
            "source",
            "amdgcn-amd-amdhsa--gfx1201",
            &options,
            "meta",
        );
        let path = persistent_code_object_cache_path(&key);
        let filename = path
            .file_name()
            .expect("cache path should include a filename")
            .to_string_lossy();
        assert!(filename.starts_with("comgr-amdgcn-amd-amdhsa--gfx1201-"));
        assert!(filename.ends_with(".hsaco"));
    }

    #[test]
    fn specialization_cache_reuses_successful_compile() {
        let cache = SpecializationCache::new();
        let key = SpecializationCacheKey::new("source", "gfx1100", &["-O3".to_string()], "meta");
        let first = cache
            .get_or_compile(key.clone(), || Ok(vec![1, 2, 3]))
            .expect("first compile should work");
        let second = cache
            .get_or_compile(key, || Ok(vec![9, 9, 9]))
            .expect("second compile should hit cache");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(second.as_slice(), [1, 2, 3]);
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
}
