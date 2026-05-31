use std::ffi::{CStr, CString, c_char, c_int};
use std::fmt;
use std::ptr;

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
    let source = CString::new(source).expect("kernel source contained NUL");
    let name = c"kernel.hip";
    let program = Program::new(&source, name)?;
    let options = [
        CString::new(format!("--gpu-architecture={arch}")).unwrap(),
        CString::new("-std=c++17").unwrap(),
        CString::new("-O3").unwrap(),
    ];
    program.compile(&options)?;
    program.code()
}
