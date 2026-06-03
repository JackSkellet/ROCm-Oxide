use crate::hip::{DeviceBuffer, Stream};
use crate::{Error, Result, validate_buffer_len};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

const RTLD_NOW: c_int = 2;
const ROCBLAS_STATUS_SUCCESS: c_int = 0;
const ROCBLAS_OPERATION_NONE: c_int = 111;
const ROCFFT_STATUS_SUCCESS: c_int = 0;
const HIPBLAS_STATUS_SUCCESS: c_int = 0;
const HIPBLAS_OP_N: c_int = 111;
const HIPBLAS_COMPUTE_32F: c_int = 2;
const HIP_R_32F: c_int = 0;
const HIPBLASLT_MATMUL_DESC_TRANSA: c_int = 0;
const HIPBLASLT_MATMUL_DESC_TRANSB: c_int = 1;
const HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: c_int = 1;
const HIPBLASLT_MAX_REQUESTED_ALGOS: i32 = 64;
const HIPBLASLT_MAX_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
const AMD_COMGR_STATUS_SUCCESS: c_int = 0;
const AMD_COMGR_LANGUAGE_HIP: c_int = 0x3;
const AMD_COMGR_DATA_KIND_SOURCE: c_int = 0x1;
const AMD_COMGR_DATA_KIND_DIAGNOSTIC: c_int = 0x4;
const AMD_COMGR_DATA_KIND_LOG: c_int = 0x5;
const AMD_COMGR_DATA_KIND_RELOCATABLE: c_int = 0x7;
const AMD_COMGR_DATA_KIND_EXECUTABLE: c_int = 0x8;
const AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE: c_int = 0x7;
const AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE: c_int = 0xD;
const ROCPRIM_SHIM_STATUS_UNAVAILABLE: c_int = 1_000_001;

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

type HipBlasLtStatus = c_int;
type HipBlasLtHandleRaw = *mut c_void;
type HipBlasLtMatmulDescRaw = *mut c_void;
type HipBlasLtMatrixLayoutRaw = *mut c_void;
type HipBlasLtMatmulPreferenceRaw = *mut c_void;
type HipBlasLtCreate = unsafe extern "C" fn(*mut HipBlasLtHandleRaw) -> HipBlasLtStatus;
type HipBlasLtDestroy = unsafe extern "C" fn(HipBlasLtHandleRaw) -> HipBlasLtStatus;
type HipBlasLtGetVersion = unsafe extern "C" fn(HipBlasLtHandleRaw, *mut c_int) -> HipBlasLtStatus;
type HipBlasLtMatmulDescCreate =
    unsafe extern "C" fn(*mut HipBlasLtMatmulDescRaw, c_int, c_int) -> HipBlasLtStatus;
type HipBlasLtMatmulDescDestroy = unsafe extern "C" fn(HipBlasLtMatmulDescRaw) -> HipBlasLtStatus;
type HipBlasLtMatmulDescSetAttribute =
    unsafe extern "C" fn(HipBlasLtMatmulDescRaw, c_int, *const c_void, usize) -> HipBlasLtStatus;
type HipBlasLtMatrixLayoutCreate =
    unsafe extern "C" fn(*mut HipBlasLtMatrixLayoutRaw, c_int, u64, u64, i64) -> HipBlasLtStatus;
type HipBlasLtMatrixLayoutDestroy =
    unsafe extern "C" fn(HipBlasLtMatrixLayoutRaw) -> HipBlasLtStatus;
type HipBlasLtMatmulPreferenceCreate =
    unsafe extern "C" fn(*mut HipBlasLtMatmulPreferenceRaw) -> HipBlasLtStatus;
type HipBlasLtMatmulPreferenceDestroy =
    unsafe extern "C" fn(HipBlasLtMatmulPreferenceRaw) -> HipBlasLtStatus;
type HipBlasLtMatmulPreferenceSetAttribute = unsafe extern "C" fn(
    HipBlasLtMatmulPreferenceRaw,
    c_int,
    *const c_void,
    usize,
) -> HipBlasLtStatus;
type HipBlasLtMatmulAlgoGetHeuristic = unsafe extern "C" fn(
    HipBlasLtHandleRaw,
    HipBlasLtMatmulDescRaw,
    HipBlasLtMatrixLayoutRaw,
    HipBlasLtMatrixLayoutRaw,
    HipBlasLtMatrixLayoutRaw,
    HipBlasLtMatrixLayoutRaw,
    HipBlasLtMatmulPreferenceRaw,
    c_int,
    *mut HipBlasLtMatmulHeuristicResultRaw,
    *mut c_int,
) -> HipBlasLtStatus;
type HipBlasLtMatmul = unsafe extern "C" fn(
    HipBlasLtHandleRaw,
    HipBlasLtMatmulDescRaw,
    *const c_void,
    *const c_void,
    HipBlasLtMatrixLayoutRaw,
    *const c_void,
    HipBlasLtMatrixLayoutRaw,
    *const c_void,
    *const c_void,
    HipBlasLtMatrixLayoutRaw,
    *mut c_void,
    HipBlasLtMatrixLayoutRaw,
    *const HipBlasLtMatmulAlgoRaw,
    *mut c_void,
    usize,
    *mut c_void,
) -> HipBlasLtStatus;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct HipBlasLtMatmulAlgoRaw {
    data: [u8; 16],
    max_workspace_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct HipBlasLtMatmulHeuristicResultRaw {
    algo: HipBlasLtMatmulAlgoRaw,
    workspace_size: usize,
    state: c_int,
    waves_count: f32,
    reserved: [c_int; 4],
}

type AmdComgrStatus = c_int;
type AmdComgrLanguage = c_int;
type AmdComgrDataKind = c_int;
type AmdComgrActionKind = c_int;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AmdComgrData {
    handle: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AmdComgrDataSet {
    handle: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AmdComgrActionInfo {
    handle: u64,
}
type AmdComgrStatusString =
    unsafe extern "C" fn(AmdComgrStatus, *mut *const c_char) -> AmdComgrStatus;
type AmdComgrGetVersion = unsafe extern "C" fn(*mut usize, *mut usize);
type AmdComgrCreateData =
    unsafe extern "C" fn(AmdComgrDataKind, *mut AmdComgrData) -> AmdComgrStatus;
type AmdComgrReleaseData = unsafe extern "C" fn(AmdComgrData) -> AmdComgrStatus;
type AmdComgrSetData = unsafe extern "C" fn(AmdComgrData, usize, *const c_char) -> AmdComgrStatus;
type AmdComgrSetDataName = unsafe extern "C" fn(AmdComgrData, *const c_char) -> AmdComgrStatus;
type AmdComgrGetData =
    unsafe extern "C" fn(AmdComgrData, *mut usize, *mut c_char) -> AmdComgrStatus;
type AmdComgrCreateDataSet = unsafe extern "C" fn(*mut AmdComgrDataSet) -> AmdComgrStatus;
type AmdComgrDestroyDataSet = unsafe extern "C" fn(AmdComgrDataSet) -> AmdComgrStatus;
type AmdComgrDataSetAdd = unsafe extern "C" fn(AmdComgrDataSet, AmdComgrData) -> AmdComgrStatus;
type AmdComgrActionDataCount =
    unsafe extern "C" fn(AmdComgrDataSet, AmdComgrDataKind, *mut usize) -> AmdComgrStatus;
type AmdComgrActionDataGetData = unsafe extern "C" fn(
    AmdComgrDataSet,
    AmdComgrDataKind,
    usize,
    *mut AmdComgrData,
) -> AmdComgrStatus;
type AmdComgrCreateActionInfo = unsafe extern "C" fn(*mut AmdComgrActionInfo) -> AmdComgrStatus;
type AmdComgrDestroyActionInfo = unsafe extern "C" fn(AmdComgrActionInfo) -> AmdComgrStatus;
type AmdComgrActionInfoSetIsaName =
    unsafe extern "C" fn(AmdComgrActionInfo, *const c_char) -> AmdComgrStatus;
type AmdComgrActionInfoSetLanguage =
    unsafe extern "C" fn(AmdComgrActionInfo, AmdComgrLanguage) -> AmdComgrStatus;
type AmdComgrActionInfoSetOptionList =
    unsafe extern "C" fn(AmdComgrActionInfo, *const *const c_char, usize) -> AmdComgrStatus;
type AmdComgrActionInfoSetLogging =
    unsafe extern "C" fn(AmdComgrActionInfo, bool) -> AmdComgrStatus;
type AmdComgrDoAction = unsafe extern "C" fn(
    AmdComgrActionKind,
    AmdComgrActionInfo,
    AmdComgrDataSet,
    AmdComgrDataSet,
) -> AmdComgrStatus;

type RocPrimStatus = c_int;
type RocPrimUnary<T> = unsafe extern "C" fn(
    *mut c_void,
    *mut usize,
    *const T,
    *mut T,
    usize,
    *mut c_void,
) -> RocPrimStatus;
type RocPrimExclusive<T> = unsafe extern "C" fn(
    *mut c_void,
    *mut usize,
    *const T,
    *mut T,
    T,
    usize,
    *mut c_void,
) -> RocPrimStatus;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

unsafe extern "C" {
    fn rocm_oxide_rocprim_available() -> c_int;
    fn rocm_oxide_rocprim_reduce_sum_u32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const u32,
        output: *mut u32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_reduce_sum_i32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const i32,
        output: *mut i32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_reduce_sum_f32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const f32,
        output: *mut f32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_inclusive_sum_u32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const u32,
        output: *mut u32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_inclusive_sum_i32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const i32,
        output: *mut i32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_inclusive_sum_f32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const f32,
        output: *mut f32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_exclusive_sum_u32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const u32,
        output: *mut u32,
        initial_value: u32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_exclusive_sum_i32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const i32,
        output: *mut i32,
        initial_value: i32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_exclusive_sum_f32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const f32,
        output: *mut f32,
        initial_value: f32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_sort_keys_u32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const u32,
        output: *mut u32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_select_flagged_u32(
        temporary_storage: *mut c_void,
        storage_size: *mut usize,
        input: *const u32,
        flags: *const u8,
        output: *mut u32,
        selected_count: *mut u32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_rocprim_transform_add_u32(
        input: *const u32,
        output: *mut u32,
        addend: u32,
        size: usize,
        stream: *mut c_void,
    ) -> RocPrimStatus;
    fn rocm_oxide_thrust_available() -> c_int;
    fn rocm_oxide_thrust_sort_u32(data: *mut u32, size: usize, stream: *mut c_void) -> c_int;
    fn rocm_oxide_thrust_sort_by_key_u32(
        keys: *mut u32,
        values: *mut u32,
        size: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn rocm_oxide_thrust_unique_u32(
        data: *mut u32,
        size: usize,
        new_size_out: *mut usize,
        stream: *mut c_void,
    ) -> c_int;
    fn rocm_oxide_thrust_count_u32(
        data: *const u32,
        size: usize,
        value: u32,
        count_out: *mut usize,
        stream: *mut c_void,
    ) -> c_int;
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
    pub hipblaslt: LibraryAvailability,
    pub comgr: LibraryAvailability,
    pub rocprim: LibraryAvailability,
    pub rocthrust: LibraryAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixIntegrationReport {
    pub hipblaslt: LibraryAvailability,
    pub composable_kernel: LibraryAvailability,
    pub rocwmma: LibraryAvailability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComgrVersion {
    pub major: usize,
    pub minor: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipBlasLtMatrixLayout {
    pub rows: u64,
    pub cols: u64,
    pub leading_dim: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipBlasLtMatmulProblem {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub a: HipBlasLtMatrixLayout,
    pub b: HipBlasLtMatrixLayout,
    pub c: HipBlasLtMatrixLayout,
    pub d: HipBlasLtMatrixLayout,
    pub max_workspace_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipBlasLtHeuristicSummary {
    pub requested_algo_count: i32,
    pub returned_algo_count: i32,
    pub best_workspace_bytes: Option<usize>,
    pub best_state: Option<i32>,
    pub best_waves_count: Option<f32>,
    pub workspace_limit_bytes: u64,
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

pub struct HipBlasLt {
    funcs: Arc<HipBlasLtFunctions>,
}

pub struct HipBlasLtHandle {
    raw: HipBlasLtHandleRaw,
    funcs: Arc<HipBlasLtFunctions>,
}

struct HipBlasLtMatmulDesc {
    raw: HipBlasLtMatmulDescRaw,
    funcs: Arc<HipBlasLtFunctions>,
}

struct HipBlasLtMatrixLayoutDesc {
    raw: HipBlasLtMatrixLayoutRaw,
    funcs: Arc<HipBlasLtFunctions>,
}

struct HipBlasLtMatmulPreference {
    raw: HipBlasLtMatmulPreferenceRaw,
    funcs: Arc<HipBlasLtFunctions>,
}

pub struct Comgr {
    funcs: Arc<ComgrFunctions>,
}

pub struct RocPrim;

/// Compile-time rocThrust wrapper (header-only, no dynamic loading required).
#[derive(Clone, Copy, Debug)]
pub struct RocThrust;

pub struct DeviceAlgorithmTemporaryStorage {
    buffer: DeviceBuffer<u8>,
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

struct HipBlasLtFunctions {
    _lib: Arc<DynamicLibrary>,
    create: HipBlasLtCreate,
    destroy: HipBlasLtDestroy,
    get_version: HipBlasLtGetVersion,
    matmul_desc_create: HipBlasLtMatmulDescCreate,
    matmul_desc_destroy: HipBlasLtMatmulDescDestroy,
    matmul_desc_set_attribute: HipBlasLtMatmulDescSetAttribute,
    matrix_layout_create: HipBlasLtMatrixLayoutCreate,
    matrix_layout_destroy: HipBlasLtMatrixLayoutDestroy,
    matmul_preference_create: HipBlasLtMatmulPreferenceCreate,
    matmul_preference_destroy: HipBlasLtMatmulPreferenceDestroy,
    matmul_preference_set_attribute: HipBlasLtMatmulPreferenceSetAttribute,
    matmul_algo_get_heuristic: HipBlasLtMatmulAlgoGetHeuristic,
    matmul: HipBlasLtMatmul,
}

struct ComgrFunctions {
    _lib: Arc<DynamicLibrary>,
    status_string: AmdComgrStatusString,
    get_version: AmdComgrGetVersion,
    create_data: AmdComgrCreateData,
    release_data: AmdComgrReleaseData,
    set_data: AmdComgrSetData,
    set_data_name: AmdComgrSetDataName,
    get_data: AmdComgrGetData,
    create_data_set: AmdComgrCreateDataSet,
    destroy_data_set: AmdComgrDestroyDataSet,
    data_set_add: AmdComgrDataSetAdd,
    action_data_count: AmdComgrActionDataCount,
    action_data_get_data: AmdComgrActionDataGetData,
    create_action_info: AmdComgrCreateActionInfo,
    destroy_action_info: AmdComgrDestroyActionInfo,
    action_info_set_isa_name: AmdComgrActionInfoSetIsaName,
    action_info_set_language: AmdComgrActionInfoSetLanguage,
    action_info_set_option_list: AmdComgrActionInfoSetOptionList,
    action_info_set_logging: AmdComgrActionInfoSetLogging,
    do_action: AmdComgrDoAction,
}

struct ComgrData {
    raw: AmdComgrData,
    funcs: Arc<ComgrFunctions>,
}

struct ComgrDataSet {
    raw: AmdComgrDataSet,
    funcs: Arc<ComgrFunctions>,
}

struct ComgrActionInfo {
    raw: AmdComgrActionInfo,
    funcs: Arc<ComgrFunctions>,
}

struct RocFftSessionInner {
    funcs: Arc<RocFftFunctions>,
}

struct DynamicLibrary {
    handle: *mut c_void,
    name: String,
}

// DynamicLibrary owns a process-local dlopen handle and resolves immutable
// function pointers. Optional-library handles are opaque ROCm handles whose
// creation/destruction remains paired in their wrapper Drop implementations.
unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}
unsafe impl Send for RocBlasFunctions {}
unsafe impl Sync for RocBlasFunctions {}
unsafe impl Send for RocFftFunctions {}
unsafe impl Sync for RocFftFunctions {}
unsafe impl Send for HipBlasLtFunctions {}
unsafe impl Sync for HipBlasLtFunctions {}
unsafe impl Send for ComgrFunctions {}
unsafe impl Sync for ComgrFunctions {}
unsafe impl Send for RocBlasHandle {}
unsafe impl Sync for RocBlasHandle {}
unsafe impl Send for HipBlasLtHandle {}
unsafe impl Sync for HipBlasLtHandle {}
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
            hipblaslt: match HipBlasLt::open() {
                Ok(lt) => LibraryAvailability::available(format!("loaded {}", lt.funcs._lib.name)),
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
            comgr: match Comgr::open() {
                Ok(comgr) => {
                    let version = comgr.version();
                    LibraryAvailability::available(format!(
                        "loaded {} version {}.{}",
                        comgr.funcs._lib.name, version.major, version.minor
                    ))
                }
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
            rocprim: match RocPrim::open() {
                Ok(_) => LibraryAvailability::available("compiled rocPRIM/hipCUB shim"),
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
            rocthrust: match RocThrust::open() {
                Ok(_) => LibraryAvailability::available("compiled rocThrust shim"),
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
        }
    }
}

impl MatrixIntegrationReport {
    pub fn query() -> Self {
        Self {
            hipblaslt: match HipBlasLt::open() {
                Ok(lt) => LibraryAvailability::available(format!("loaded {}", lt.funcs._lib.name)),
                Err(err) => LibraryAvailability::unavailable(err.to_string()),
            },
            composable_kernel: composable_kernel_availability(),
            rocwmma: rocwmma_availability(),
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

impl HipBlasLtMatrixLayout {
    fn fp32_column_major(label: &str, rows: u64, cols: u64, leading_dim: u64) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::Library(format!(
                "hipBLASLt {label} matrix dimensions must be nonzero, got rows={rows}, cols={cols}"
            )));
        }
        if leading_dim < rows {
            return Err(Error::Library(format!(
                "hipBLASLt {label} leading dimension {leading_dim} is smaller than rows {rows}"
            )));
        }
        Ok(Self {
            rows,
            cols,
            leading_dim: positive_i64_from_u64(
                &format!("hipBLASLt {label} leading dimension"),
                leading_dim,
            )?,
        })
    }
}

impl HipBlasLtMatmulProblem {
    pub fn sgemm_nn(m: u64, n: u64, k: u64, max_workspace_bytes: u64) -> Result<Self> {
        Self::sgemm_nn_with_leading_dimensions(m, n, k, m, k, m, m, max_workspace_bytes)
    }

    pub fn sgemm_nn_with_leading_dimensions(
        m: u64,
        n: u64,
        k: u64,
        lda: u64,
        ldb: u64,
        ldc: u64,
        ldd: u64,
        max_workspace_bytes: u64,
    ) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(Error::Library(format!(
                "hipBLASLt SGEMM dimensions m, n, and k must be nonzero, got m={m}, n={n}, k={k}"
            )));
        }
        if max_workspace_bytes > HIPBLASLT_MAX_WORKSPACE_BYTES {
            return Err(Error::Library(format!(
                "hipBLASLt SGEMM workspace cap is {max_workspace_bytes} bytes, but ROCm-Oxide limits automatic workspace to {HIPBLASLT_MAX_WORKSPACE_BYTES} bytes"
            )));
        }
        let a = HipBlasLtMatrixLayout::fp32_column_major("A", m, k, lda)?;
        let b = HipBlasLtMatrixLayout::fp32_column_major("B", k, n, ldb)?;
        let c = HipBlasLtMatrixLayout::fp32_column_major("C", m, n, ldc)?;
        let d = HipBlasLtMatrixLayout::fp32_column_major("D", m, n, ldd)?;
        Ok(Self {
            m,
            n,
            k,
            a,
            b,
            c,
            d,
            max_workspace_bytes,
        })
    }

    fn validate_buffers(
        self,
        a_len: usize,
        b_len: usize,
        c_len: usize,
        d_len: usize,
    ) -> Result<()> {
        validate_buffer_len(
            "hipBLASLt SGEMM A",
            a_len,
            hipblaslt_matrix_elements(self.a)?,
        )?;
        validate_buffer_len(
            "hipBLASLt SGEMM B",
            b_len,
            hipblaslt_matrix_elements(self.b)?,
        )?;
        validate_buffer_len(
            "hipBLASLt SGEMM C",
            c_len,
            hipblaslt_matrix_elements(self.c)?,
        )?;
        validate_buffer_len(
            "hipBLASLt SGEMM D",
            d_len,
            hipblaslt_matrix_elements(self.d)?,
        )?;
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

impl HipBlasLt {
    pub fn open() -> Result<Self> {
        let lib = Arc::new(DynamicLibrary::open(&library_candidates(&[
            "libhipblaslt.so",
            "libhipblaslt.so.1",
        ]))?);
        let funcs = unsafe { HipBlasLtFunctions::load(lib)? };
        Ok(Self {
            funcs: Arc::new(funcs),
        })
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    pub fn create_handle(&self) -> Result<HipBlasLtHandle> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_hipblaslt((self.funcs.create)(&mut raw), "hipblasLtCreate")?;
        }
        Ok(HipBlasLtHandle {
            raw,
            funcs: Arc::clone(&self.funcs),
        })
    }
}

impl HipBlasLtHandle {
    pub fn version(&self) -> Result<i32> {
        let mut version = 0;
        unsafe {
            check_hipblaslt(
                (self.funcs.get_version)(self.raw, &mut version),
                "hipblasLtGetVersion",
            )?;
        }
        Ok(version)
    }

    pub fn sgemm_nn_heuristics(
        &self,
        problem: HipBlasLtMatmulProblem,
        requested_algo_count: i32,
    ) -> Result<HipBlasLtHeuristicSummary> {
        let (results, returned_algo_count) =
            self.sgemm_nn_heuristic_results(problem, requested_algo_count)?;
        Ok(summarize_hipblaslt_heuristics(
            requested_algo_count,
            returned_algo_count,
            &results,
            problem.max_workspace_bytes,
        ))
    }

    pub fn sgemm_nn(
        &self,
        problem: HipBlasLtMatmulProblem,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        c: &DeviceBuffer<f32>,
        d: &DeviceBuffer<f32>,
        requested_algo_count: i32,
    ) -> Result<HipBlasLtHeuristicSummary> {
        let (results, returned_algo_count) =
            self.sgemm_nn_heuristic_results(problem, requested_algo_count)?;
        let best = best_hipblaslt_heuristic(&results, returned_algo_count)?;
        validate_hipblaslt_workspace(best.workspace_size, problem.max_workspace_bytes)?;
        let workspace = (best.workspace_size > 0)
            .then(|| DeviceAlgorithmTemporaryStorage::new(best.workspace_size))
            .transpose()?;
        let stream = Stream::new()?;
        unsafe {
            self.launch_sgemm_nn_on_stream(
                &stream,
                problem,
                alpha,
                a,
                b,
                beta,
                c,
                d,
                &best.algo,
                best.workspace_size,
                workspace.as_ref(),
            )?;
        }
        stream.synchronize()?;
        Ok(summarize_hipblaslt_heuristics(
            requested_algo_count,
            returned_algo_count,
            &results,
            problem.max_workspace_bytes,
        ))
    }

    /// Enqueues FP32 column-major `D = alpha * A * B + beta * C` through hipBLASLt.
    ///
    /// # Safety
    ///
    /// The caller must keep `a`, `b`, `c`, `d`, `temporary_storage`, and `stream`
    /// alive until the stream reaches the enqueued matmul. When a heuristic
    /// requires workspace, `temporary_storage` must refer to device memory that
    /// is not concurrently mutated by other work on the same interval.
    pub unsafe fn sgemm_nn_on_stream(
        &self,
        stream: &Stream,
        problem: HipBlasLtMatmulProblem,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        c: &DeviceBuffer<f32>,
        d: &DeviceBuffer<f32>,
        temporary_storage: Option<&DeviceAlgorithmTemporaryStorage>,
        requested_algo_count: i32,
    ) -> Result<HipBlasLtHeuristicSummary> {
        let (results, returned_algo_count) =
            self.sgemm_nn_heuristic_results(problem, requested_algo_count)?;
        let best = best_hipblaslt_heuristic(&results, returned_algo_count)?;
        validate_hipblaslt_workspace(best.workspace_size, problem.max_workspace_bytes)?;
        unsafe {
            self.launch_sgemm_nn_on_stream(
                stream,
                problem,
                alpha,
                a,
                b,
                beta,
                c,
                d,
                &best.algo,
                best.workspace_size,
                temporary_storage,
            )?;
        }
        Ok(summarize_hipblaslt_heuristics(
            requested_algo_count,
            returned_algo_count,
            &results,
            problem.max_workspace_bytes,
        ))
    }

    fn sgemm_nn_heuristic_results(
        &self,
        problem: HipBlasLtMatmulProblem,
        requested_algo_count: i32,
    ) -> Result<(Vec<HipBlasLtMatmulHeuristicResultRaw>, i32)> {
        if requested_algo_count <= 0 {
            return Err(Error::Library(format!(
                "hipBLASLt requested algorithm count must be positive, got {requested_algo_count}"
            )));
        }
        if requested_algo_count > HIPBLASLT_MAX_REQUESTED_ALGOS {
            return Err(Error::Library(format!(
                "hipBLASLt requested algorithm count {requested_algo_count} exceeds ROCm-Oxide cap {HIPBLASLT_MAX_REQUESTED_ALGOS}"
            )));
        }

        let matmul_desc = HipBlasLtMatmulDesc::sgemm_nn(Arc::clone(&self.funcs))?;
        let a = HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.a)?;
        let b = HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.b)?;
        let c = HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.c)?;
        let d = HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.d)?;
        let preference = HipBlasLtMatmulPreference::with_max_workspace(
            Arc::clone(&self.funcs),
            problem.max_workspace_bytes,
        )?;
        let mut results =
            vec![HipBlasLtMatmulHeuristicResultRaw::default(); requested_algo_count as usize];
        let mut returned_algo_count = 0;
        unsafe {
            check_hipblaslt(
                (self.funcs.matmul_algo_get_heuristic)(
                    self.raw,
                    matmul_desc.raw,
                    a.raw,
                    b.raw,
                    c.raw,
                    d.raw,
                    preference.raw,
                    requested_algo_count,
                    results.as_mut_ptr(),
                    &mut returned_algo_count,
                ),
                "hipblasLtMatmulAlgoGetHeuristic",
            )?;
        }

        Ok((results, returned_algo_count))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_sgemm_nn_on_stream(
        &self,
        stream: &Stream,
        problem: HipBlasLtMatmulProblem,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        c: &DeviceBuffer<f32>,
        d: &DeviceBuffer<f32>,
        algo: &HipBlasLtMatmulAlgoRaw,
        workspace_bytes: usize,
        temporary_storage: Option<&DeviceAlgorithmTemporaryStorage>,
    ) -> Result<()> {
        problem.validate_buffers(a.len(), b.len(), c.len(), d.len())?;
        let workspace = if workspace_bytes == 0 {
            ptr::null_mut()
        } else {
            let storage = temporary_storage.ok_or_else(|| {
                Error::Library(format!(
                    "hipBLASLt SGEMM requires {workspace_bytes} bytes of temporary storage"
                ))
            })?;
            validate_buffer_len(
                "hipBLASLt SGEMM workspace",
                storage.bytes(),
                workspace_bytes,
            )?;
            storage.as_mut_ptr()
        };

        let matmul_desc = HipBlasLtMatmulDesc::sgemm_nn(Arc::clone(&self.funcs))?;
        let a_desc =
            HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.a)?;
        let b_desc =
            HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.b)?;
        let c_desc =
            HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.c)?;
        let d_desc =
            HipBlasLtMatrixLayoutDesc::fp32_column_major(Arc::clone(&self.funcs), problem.d)?;

        unsafe {
            check_hipblaslt(
                (self.funcs.matmul)(
                    self.raw,
                    matmul_desc.raw,
                    (&alpha as *const f32).cast::<c_void>(),
                    a.as_ptr().cast::<c_void>(),
                    a_desc.raw,
                    b.as_ptr().cast::<c_void>(),
                    b_desc.raw,
                    (&beta as *const f32).cast::<c_void>(),
                    c.as_ptr().cast::<c_void>(),
                    c_desc.raw,
                    d.as_mut_ptr().cast::<c_void>(),
                    d_desc.raw,
                    algo,
                    workspace,
                    workspace_bytes,
                    stream.as_raw(),
                ),
                "hipblasLtMatmul",
            )
        }
    }
}

impl Drop for HipBlasLtHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.funcs.destroy)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl HipBlasLtMatmulDesc {
    fn sgemm_nn(funcs: Arc<HipBlasLtFunctions>) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_hipblaslt(
                (funcs.matmul_desc_create)(&mut raw, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                "hipblasLtMatmulDescCreate",
            )?;
        }
        let desc = Self { raw, funcs };
        desc.set_operation(HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLAS_OP_N)?;
        desc.set_operation(HIPBLASLT_MATMUL_DESC_TRANSB, HIPBLAS_OP_N)?;
        Ok(desc)
    }

    fn set_operation(&self, attribute: c_int, value: c_int) -> Result<()> {
        unsafe {
            check_hipblaslt(
                (self.funcs.matmul_desc_set_attribute)(
                    self.raw,
                    attribute,
                    (&value as *const c_int).cast(),
                    std::mem::size_of_val(&value),
                ),
                "hipblasLtMatmulDescSetAttribute",
            )
        }
    }
}

impl Drop for HipBlasLtMatmulDesc {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.funcs.matmul_desc_destroy)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl HipBlasLtMatrixLayoutDesc {
    fn fp32_column_major(
        funcs: Arc<HipBlasLtFunctions>,
        layout: HipBlasLtMatrixLayout,
    ) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_hipblaslt(
                (funcs.matrix_layout_create)(
                    &mut raw,
                    HIP_R_32F,
                    layout.rows,
                    layout.cols,
                    layout.leading_dim,
                ),
                "hipblasLtMatrixLayoutCreate",
            )?;
        }
        Ok(Self { raw, funcs })
    }
}

impl Drop for HipBlasLtMatrixLayoutDesc {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.funcs.matrix_layout_destroy)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl HipBlasLtMatmulPreference {
    fn with_max_workspace(
        funcs: Arc<HipBlasLtFunctions>,
        max_workspace_bytes: u64,
    ) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_hipblaslt(
                (funcs.matmul_preference_create)(&mut raw),
                "hipblasLtMatmulPreferenceCreate",
            )?;
        }
        let preference = Self { raw, funcs };
        preference.set_max_workspace(max_workspace_bytes)?;
        Ok(preference)
    }

    fn set_max_workspace(&self, max_workspace_bytes: u64) -> Result<()> {
        unsafe {
            check_hipblaslt(
                (self.funcs.matmul_preference_set_attribute)(
                    self.raw,
                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&max_workspace_bytes as *const u64).cast(),
                    std::mem::size_of_val(&max_workspace_bytes),
                ),
                "hipblasLtMatmulPreferenceSetAttribute",
            )
        }
    }
}

impl Drop for HipBlasLtMatmulPreference {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.funcs.matmul_preference_destroy)(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl Comgr {
    pub fn open() -> Result<Self> {
        let lib = Arc::new(DynamicLibrary::open(&library_candidates(&[
            "libamd_comgr.so",
            "libamd_comgr.so.3",
        ]))?);
        let funcs = unsafe { ComgrFunctions::load(lib)? };
        Ok(Self {
            funcs: Arc::new(funcs),
        })
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    pub fn version(&self) -> ComgrVersion {
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            (self.funcs.get_version)(&mut major, &mut minor);
        }
        ComgrVersion { major, minor }
    }

    pub fn compile_hip_source_to_code_object(
        &self,
        source: &str,
        arch: &str,
        extra_options: &[&str],
    ) -> Result<Vec<u8>> {
        let options = comgr_compile_options(extra_options);
        self.compile_hip_source_to_code_object_with_options(source, arch, &options)
    }

    pub fn compile_hip_source_to_code_object_with_options(
        &self,
        source: &str,
        arch: &str,
        options: &[String],
    ) -> Result<Vec<u8>> {
        let source_data = ComgrData::new(Arc::clone(&self.funcs), AMD_COMGR_DATA_KIND_SOURCE)?;
        source_data.set_name("kernel.hip")?;
        source_data.set_data(source.as_bytes())?;

        let source_input = ComgrDataSet::new(Arc::clone(&self.funcs))?;
        source_input.add(&source_data)?;
        let relocatable_output = ComgrDataSet::new(Arc::clone(&self.funcs))?;

        let isa_name = comgr_isa_name(arch);
        let compile_info = ComgrActionInfo::new(Arc::clone(&self.funcs))?;
        compile_info.set_isa_name(&isa_name)?;
        compile_info.set_language(AMD_COMGR_LANGUAGE_HIP)?;
        compile_info.set_option_list(options)?;
        compile_info.set_logging(true)?;

        let compile_status = unsafe {
            (self.funcs.do_action)(
                AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE,
                compile_info.raw,
                source_input.raw,
                relocatable_output.raw,
            )
        };
        if compile_status != AMD_COMGR_STATUS_SUCCESS {
            let logs = relocatable_output.diagnostics_and_logs();
            let mut message = format!(
                "COMGR compile HIP source to relocatable failed: {}",
                self.funcs.status_message(compile_status)
            );
            if !logs.trim().is_empty() {
                message.push('\n');
                message.push_str(logs.trim());
            }
            return Err(Error::Library(message));
        }
        if relocatable_output.count(AMD_COMGR_DATA_KIND_RELOCATABLE)? == 0 {
            return Err(Error::Library(
                "COMGR compile produced no relocatable code object".to_string(),
            ));
        }

        let link_info = ComgrActionInfo::new(Arc::clone(&self.funcs))?;
        link_info.set_isa_name(&isa_name)?;
        link_info.set_logging(true)?;
        let executable_output = ComgrDataSet::new(Arc::clone(&self.funcs))?;
        let link_status = unsafe {
            (self.funcs.do_action)(
                AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                link_info.raw,
                relocatable_output.raw,
                executable_output.raw,
            )
        };
        if link_status != AMD_COMGR_STATUS_SUCCESS {
            let logs = executable_output.diagnostics_and_logs();
            let mut message = format!(
                "COMGR link relocatable to executable failed: {}",
                self.funcs.status_message(link_status)
            );
            if !logs.trim().is_empty() {
                message.push('\n');
                message.push_str(logs.trim());
            }
            return Err(Error::Library(message));
        }

        executable_output.first_data_bytes(AMD_COMGR_DATA_KIND_EXECUTABLE)
    }
}

impl ComgrData {
    fn new(funcs: Arc<ComgrFunctions>, kind: AmdComgrDataKind) -> Result<Self> {
        let mut raw = AmdComgrData::default();
        unsafe {
            check_comgr(
                &funcs,
                (funcs.create_data)(kind, &mut raw),
                "amd_comgr_create_data",
            )?;
        }
        Ok(Self { raw, funcs })
    }

    fn set_name(&self, name: &str) -> Result<()> {
        let name = CString::new(name)
            .map_err(|_| Error::Library("COMGR data name contains a NUL byte".to_string()))?;
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.set_data_name)(self.raw, name.as_ptr()),
                "amd_comgr_set_data_name",
            )
        }
    }

    fn set_data(&self, bytes: &[u8]) -> Result<()> {
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.set_data)(self.raw, bytes.len(), bytes.as_ptr().cast::<c_char>()),
                "amd_comgr_set_data",
            )
        }
    }

    fn bytes(&self) -> Result<Vec<u8>> {
        let mut size = 0usize;
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.get_data)(self.raw, &mut size, ptr::null_mut()),
                "amd_comgr_get_data size",
            )?;
            let mut bytes = vec![0u8; size];
            check_comgr(
                &self.funcs,
                (self.funcs.get_data)(self.raw, &mut size, bytes.as_mut_ptr().cast::<c_char>()),
                "amd_comgr_get_data",
            )?;
            bytes.truncate(size);
            Ok(bytes)
        }
    }

    fn text_lossy(&self) -> String {
        match self.bytes() {
            Ok(bytes) => String::from_utf8_lossy(&bytes)
                .trim_end_matches('\0')
                .to_string(),
            Err(err) => format!("<failed to read COMGR output: {err}>"),
        }
    }
}

impl Drop for ComgrData {
    fn drop(&mut self) {
        if self.raw.handle != 0 {
            unsafe {
                let _ = (self.funcs.release_data)(self.raw);
            }
            self.raw.handle = 0;
        }
    }
}

impl ComgrDataSet {
    fn new(funcs: Arc<ComgrFunctions>) -> Result<Self> {
        let mut raw = AmdComgrDataSet::default();
        unsafe {
            check_comgr(
                &funcs,
                (funcs.create_data_set)(&mut raw),
                "amd_comgr_create_data_set",
            )?;
        }
        Ok(Self { raw, funcs })
    }

    fn add(&self, data: &ComgrData) -> Result<()> {
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.data_set_add)(self.raw, data.raw),
                "amd_comgr_data_set_add",
            )
        }
    }

    fn count(&self, kind: AmdComgrDataKind) -> Result<usize> {
        let mut count = 0usize;
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.action_data_count)(self.raw, kind, &mut count),
                "amd_comgr_action_data_count",
            )?;
        }
        Ok(count)
    }

    fn get(&self, kind: AmdComgrDataKind, index: usize) -> Result<ComgrData> {
        let mut raw = AmdComgrData::default();
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.action_data_get_data)(self.raw, kind, index, &mut raw),
                "amd_comgr_action_data_get_data",
            )?;
        }
        Ok(ComgrData {
            raw,
            funcs: Arc::clone(&self.funcs),
        })
    }

    fn first_data_bytes(&self, kind: AmdComgrDataKind) -> Result<Vec<u8>> {
        let count = self.count(kind)?;
        if count == 0 {
            let logs = self.diagnostics_and_logs();
            let mut message = "COMGR action produced no executable code object".to_string();
            if !logs.trim().is_empty() {
                message.push('\n');
                message.push_str(logs.trim());
            }
            return Err(Error::Library(message));
        }
        self.get(kind, 0)?.bytes()
    }

    fn diagnostics_and_logs(&self) -> String {
        let mut parts = Vec::new();
        for kind in [AMD_COMGR_DATA_KIND_LOG, AMD_COMGR_DATA_KIND_DIAGNOSTIC] {
            let Ok(count) = self.count(kind) else {
                continue;
            };
            for index in 0..count {
                if let Ok(data) = self.get(kind, index) {
                    let text = data.text_lossy();
                    if !text.trim().is_empty() {
                        parts.push(text);
                    }
                }
            }
        }
        parts.join("\n")
    }
}

impl Drop for ComgrDataSet {
    fn drop(&mut self) {
        if self.raw.handle != 0 {
            unsafe {
                let _ = (self.funcs.destroy_data_set)(self.raw);
            }
            self.raw.handle = 0;
        }
    }
}

impl ComgrActionInfo {
    fn new(funcs: Arc<ComgrFunctions>) -> Result<Self> {
        let mut raw = AmdComgrActionInfo::default();
        unsafe {
            check_comgr(
                &funcs,
                (funcs.create_action_info)(&mut raw),
                "amd_comgr_create_action_info",
            )?;
        }
        Ok(Self { raw, funcs })
    }

    fn set_isa_name(&self, isa_name: &str) -> Result<()> {
        let isa_name = CString::new(isa_name)
            .map_err(|_| Error::Library("COMGR ISA name contains a NUL byte".to_string()))?;
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.action_info_set_isa_name)(self.raw, isa_name.as_ptr()),
                "amd_comgr_action_info_set_isa_name",
            )
        }
    }

    fn set_language(&self, language: AmdComgrLanguage) -> Result<()> {
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.action_info_set_language)(self.raw, language),
                "amd_comgr_action_info_set_language",
            )
        }
    }

    fn set_option_list(&self, options: &[String]) -> Result<()> {
        let options = options
            .iter()
            .map(|option| {
                CString::new(option.as_str()).map_err(|_| {
                    Error::Library(format!("COMGR option `{option}` contains a NUL byte"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let option_ptrs = options
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.action_info_set_option_list)(
                    self.raw,
                    option_ptrs.as_ptr(),
                    option_ptrs.len(),
                ),
                "amd_comgr_action_info_set_option_list",
            )
        }
    }

    fn set_logging(&self, enabled: bool) -> Result<()> {
        unsafe {
            check_comgr(
                &self.funcs,
                (self.funcs.action_info_set_logging)(self.raw, enabled),
                "amd_comgr_action_info_set_logging",
            )
        }
    }
}

impl Drop for ComgrActionInfo {
    fn drop(&mut self) {
        if self.raw.handle != 0 {
            unsafe {
                let _ = (self.funcs.destroy_action_info)(self.raw);
            }
            self.raw.handle = 0;
        }
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

impl RocPrim {
    pub fn open() -> Result<Self> {
        if unsafe { rocm_oxide_rocprim_available() } != 0 {
            Ok(Self)
        } else {
            Err(Error::Library(
                "rocPRIM/hipCUB headers were unavailable when ROCm-Oxide was built".to_string(),
            ))
        }
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    pub fn reduce_sum_u32_storage_bytes(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<usize> {
        validate_reduce_u32(input, output)?;
        query_rocprim_u32_storage(
            rocm_oxide_rocprim_reduce_sum_u32,
            "rocPRIM reduce_sum_u32 storage query",
            input,
            output,
        )
    }

    pub fn inclusive_sum_u32_storage_bytes(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<usize> {
        validate_scan_u32("rocPRIM inclusive scan output", input, output)?;
        query_rocprim_u32_storage(
            rocm_oxide_rocprim_inclusive_sum_u32,
            "rocPRIM inclusive_sum_u32 storage query",
            input,
            output,
        )
    }

    pub fn exclusive_sum_u32_storage_bytes(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<usize> {
        validate_scan_u32("rocPRIM exclusive scan output", input, output)?;
        let mut storage_bytes = 0usize;
        unsafe {
            check_rocprim(
                rocm_oxide_rocprim_exclusive_sum_u32(
                    ptr::null_mut(),
                    &mut storage_bytes,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    0,
                    input.len(),
                    ptr::null_mut(),
                ),
                "rocPRIM exclusive_sum_u32 storage query",
            )?;
        }
        Ok(storage_bytes)
    }

    pub fn temporary_storage_for_reduce_sum_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<DeviceAlgorithmTemporaryStorage> {
        DeviceAlgorithmTemporaryStorage::new(self.reduce_sum_u32_storage_bytes(input, output)?)
    }

    pub fn temporary_storage_for_inclusive_sum_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<DeviceAlgorithmTemporaryStorage> {
        DeviceAlgorithmTemporaryStorage::new(self.inclusive_sum_u32_storage_bytes(input, output)?)
    }

    pub fn temporary_storage_for_exclusive_sum_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<DeviceAlgorithmTemporaryStorage> {
        DeviceAlgorithmTemporaryStorage::new(self.exclusive_sum_u32_storage_bytes(input, output)?)
    }

    pub fn reduce_sum_u32_on_stream(
        &self,
        stream: &Stream,
        temporary_storage: &DeviceAlgorithmTemporaryStorage,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let required = self.reduce_sum_u32_storage_bytes(input, output)?;
        validate_temporary_storage(temporary_storage, required)?;
        call_rocprim_u32(
            rocm_oxide_rocprim_reduce_sum_u32,
            "rocPRIM reduce_sum_u32",
            stream,
            temporary_storage,
            input,
            output,
        )
    }

    pub fn inclusive_sum_u32_on_stream(
        &self,
        stream: &Stream,
        temporary_storage: &DeviceAlgorithmTemporaryStorage,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let required = self.inclusive_sum_u32_storage_bytes(input, output)?;
        validate_temporary_storage(temporary_storage, required)?;
        call_rocprim_u32(
            rocm_oxide_rocprim_inclusive_sum_u32,
            "rocPRIM inclusive_sum_u32",
            stream,
            temporary_storage,
            input,
            output,
        )
    }

    pub fn exclusive_sum_u32_on_stream(
        &self,
        stream: &Stream,
        temporary_storage: &DeviceAlgorithmTemporaryStorage,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
        initial_value: u32,
    ) -> Result<()> {
        let required = self.exclusive_sum_u32_storage_bytes(input, output)?;
        validate_temporary_storage(temporary_storage, required)?;
        let mut storage_bytes = temporary_storage.bytes();
        unsafe {
            check_rocprim(
                rocm_oxide_rocprim_exclusive_sum_u32(
                    temporary_storage.as_mut_ptr(),
                    &mut storage_bytes,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    initial_value,
                    input.len(),
                    stream.as_raw(),
                ),
                "rocPRIM exclusive_sum_u32",
            )
        }
    }

    pub fn reduce_sum_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = Stream::new()?;
        let storage = self.temporary_storage_for_reduce_sum_u32(input, output)?;
        self.reduce_sum_u32_on_stream(&stream, &storage, input, output)?;
        Ok(stream.synchronize()?)
    }

    pub fn inclusive_sum_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = Stream::new()?;
        let storage = self.temporary_storage_for_inclusive_sum_u32(input, output)?;
        self.inclusive_sum_u32_on_stream(&stream, &storage, input, output)?;
        Ok(stream.synchronize()?)
    }

    pub fn exclusive_sum_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
        initial_value: u32,
    ) -> Result<()> {
        let stream = Stream::new()?;
        let storage = self.temporary_storage_for_exclusive_sum_u32(input, output)?;
        self.exclusive_sum_u32_on_stream(&stream, &storage, input, output, initial_value)?;
        Ok(stream.synchronize()?)
    }

    pub fn reduce_sum_i32(
        &self,
        input: &DeviceBuffer<i32>,
        output: &DeviceBuffer<i32>,
    ) -> Result<()> {
        validate_reduce(input, output)?;
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_unary_storage(
            rocm_oxide_rocprim_reduce_sum_i32,
            "rocPRIM reduce_sum_i32 storage query",
            input,
            output,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_unary(
            rocm_oxide_rocprim_reduce_sum_i32,
            "rocPRIM reduce_sum_i32",
            &stream,
            &storage,
            input,
            output,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn reduce_sum_f32(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        validate_reduce(input, output)?;
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_unary_storage(
            rocm_oxide_rocprim_reduce_sum_f32,
            "rocPRIM reduce_sum_f32 storage query",
            input,
            output,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_unary(
            rocm_oxide_rocprim_reduce_sum_f32,
            "rocPRIM reduce_sum_f32",
            &stream,
            &storage,
            input,
            output,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn inclusive_sum_i32(
        &self,
        input: &DeviceBuffer<i32>,
        output: &DeviceBuffer<i32>,
    ) -> Result<()> {
        validate_scan("rocPRIM inclusive scan output", input, output)?;
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_unary_storage(
            rocm_oxide_rocprim_inclusive_sum_i32,
            "rocPRIM inclusive_sum_i32 storage query",
            input,
            output,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_unary(
            rocm_oxide_rocprim_inclusive_sum_i32,
            "rocPRIM inclusive_sum_i32",
            &stream,
            &storage,
            input,
            output,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn inclusive_sum_f32(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        validate_scan("rocPRIM inclusive scan output", input, output)?;
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_unary_storage(
            rocm_oxide_rocprim_inclusive_sum_f32,
            "rocPRIM inclusive_sum_f32 storage query",
            input,
            output,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_unary(
            rocm_oxide_rocprim_inclusive_sum_f32,
            "rocPRIM inclusive_sum_f32",
            &stream,
            &storage,
            input,
            output,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn exclusive_sum_i32(
        &self,
        input: &DeviceBuffer<i32>,
        output: &DeviceBuffer<i32>,
        initial_value: i32,
    ) -> Result<()> {
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_exclusive_storage(
            rocm_oxide_rocprim_exclusive_sum_i32,
            "rocPRIM exclusive_sum_i32 storage query",
            input,
            output,
            initial_value,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_exclusive(
            rocm_oxide_rocprim_exclusive_sum_i32,
            "rocPRIM exclusive_sum_i32",
            &stream,
            &storage,
            input,
            output,
            initial_value,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn exclusive_sum_f32(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        initial_value: f32,
    ) -> Result<()> {
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_exclusive_storage(
            rocm_oxide_rocprim_exclusive_sum_f32,
            "rocPRIM exclusive_sum_f32 storage query",
            input,
            output,
            initial_value,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_exclusive(
            rocm_oxide_rocprim_exclusive_sum_f32,
            "rocPRIM exclusive_sum_f32",
            &stream,
            &storage,
            input,
            output,
            initial_value,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn sort_keys_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
    ) -> Result<()> {
        validate_buffer_len("rocPRIM sort output", output.len(), input.len())?;
        let stream = Stream::new()?;
        let storage_bytes = query_rocprim_unary_storage(
            rocm_oxide_rocprim_sort_keys_u32,
            "rocPRIM sort_keys_u32 storage query",
            input,
            output,
        )?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_unary(
            rocm_oxide_rocprim_sort_keys_u32,
            "rocPRIM sort_keys_u32",
            &stream,
            &storage,
            input,
            output,
        )?;
        Ok(stream.synchronize()?)
    }

    pub fn select_flagged_u32(
        &self,
        input: &DeviceBuffer<u32>,
        flags: &DeviceBuffer<u8>,
        output: &DeviceBuffer<u32>,
        selected_count: &DeviceBuffer<u32>,
    ) -> Result<()> {
        validate_select_flagged_u32(input, flags, output, selected_count)?;
        let stream = Stream::new()?;
        let storage_bytes =
            query_rocprim_select_flagged_u32_storage(input, flags, output, selected_count)?;
        let storage = DeviceAlgorithmTemporaryStorage::new(storage_bytes)?;
        call_rocprim_select_flagged_u32(&stream, &storage, input, flags, output, selected_count)?;
        Ok(stream.synchronize()?)
    }

    pub fn transform_add_u32(
        &self,
        input: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u32>,
        addend: u32,
    ) -> Result<()> {
        validate_buffer_len("rocPRIM transform output", output.len(), input.len())?;
        let stream = Stream::new()?;
        unsafe {
            check_rocprim(
                rocm_oxide_rocprim_transform_add_u32(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    addend,
                    input.len(),
                    stream.as_raw(),
                ),
                "rocPRIM transform_add_u32",
            )?;
        }
        Ok(stream.synchronize()?)
    }
}

impl RocThrust {
    pub fn open() -> Result<Self> {
        if unsafe { rocm_oxide_thrust_available() } != 0 {
            Ok(Self)
        } else {
            Err(Error::Library(
                "rocThrust headers were unavailable when ROCm-Oxide was built".to_string(),
            ))
        }
    }

    pub fn is_available() -> bool {
        Self::open().is_ok()
    }

    /// Sorts `data` in ascending order in-place.
    pub fn sort_u32_on_stream(&self, stream: &Stream, data: &mut DeviceBuffer<u32>) -> Result<()> {
        unsafe {
            check_thrust(
                rocm_oxide_thrust_sort_u32(data.as_mut_ptr(), data.len(), stream.as_raw()),
                "rocThrust sort_u32",
            )
        }
    }

    pub fn sort_u32(&self, data: &mut DeviceBuffer<u32>) -> Result<()> {
        let stream = Stream::new()?;
        self.sort_u32_on_stream(&stream, data)?;
        Ok(stream.synchronize()?)
    }

    /// Sorts `keys` and reorders `values` to match, both in ascending key
    /// order, in-place.
    pub fn sort_by_key_u32_on_stream(
        &self,
        stream: &Stream,
        keys: &mut DeviceBuffer<u32>,
        values: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        validate_buffer_len("rocThrust sort_by_key values", values.len(), keys.len())?;
        unsafe {
            check_thrust(
                rocm_oxide_thrust_sort_by_key_u32(
                    keys.as_mut_ptr(),
                    values.as_mut_ptr(),
                    keys.len(),
                    stream.as_raw(),
                ),
                "rocThrust sort_by_key_u32",
            )
        }
    }

    pub fn sort_by_key_u32(
        &self,
        keys: &mut DeviceBuffer<u32>,
        values: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = Stream::new()?;
        self.sort_by_key_u32_on_stream(&stream, keys, values)?;
        Ok(stream.synchronize()?)
    }

    /// Removes consecutive duplicate elements in `data`. Returns the number of
    /// unique elements; the suffix of `data` beyond that count is undefined.
    pub fn unique_u32_on_stream(
        &self,
        stream: &Stream,
        data: &mut DeviceBuffer<u32>,
    ) -> Result<usize> {
        let mut new_size = 0usize;
        unsafe {
            check_thrust(
                rocm_oxide_thrust_unique_u32(
                    data.as_mut_ptr(),
                    data.len(),
                    &mut new_size,
                    stream.as_raw(),
                ),
                "rocThrust unique_u32",
            )?;
        }
        Ok(new_size)
    }

    pub fn unique_u32(&self, data: &mut DeviceBuffer<u32>) -> Result<usize> {
        let stream = Stream::new()?;
        let n = self.unique_u32_on_stream(&stream, data)?;
        stream.synchronize()?;
        Ok(n)
    }

    /// Counts elements in `data` equal to `value`.
    pub fn count_u32_on_stream(
        &self,
        stream: &Stream,
        data: &DeviceBuffer<u32>,
        value: u32,
    ) -> Result<usize> {
        let mut count = 0usize;
        unsafe {
            check_thrust(
                rocm_oxide_thrust_count_u32(
                    data.as_ptr(),
                    data.len(),
                    value,
                    &mut count,
                    stream.as_raw(),
                ),
                "rocThrust count_u32",
            )?;
        }
        Ok(count)
    }

    pub fn count_u32(&self, data: &DeviceBuffer<u32>, value: u32) -> Result<usize> {
        let stream = Stream::new()?;
        let n = self.count_u32_on_stream(&stream, data, value)?;
        stream.synchronize()?;
        Ok(n)
    }
}

impl DeviceAlgorithmTemporaryStorage {
    pub fn new(bytes: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::new(bytes)?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.buffer.len()
    }

    pub fn as_device_buffer(&self) -> &DeviceBuffer<u8> {
        &self.buffer
    }

    fn as_mut_ptr(&self) -> *mut c_void {
        self.buffer.as_mut_ptr().cast::<c_void>()
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

impl HipBlasLtFunctions {
    unsafe fn load(lib: Arc<DynamicLibrary>) -> Result<Self> {
        Ok(Self {
            create: unsafe { lib.symbol(c"hipblasLtCreate")? },
            destroy: unsafe { lib.symbol(c"hipblasLtDestroy")? },
            get_version: unsafe { lib.symbol(c"hipblasLtGetVersion")? },
            matmul_desc_create: unsafe { lib.symbol(c"hipblasLtMatmulDescCreate")? },
            matmul_desc_destroy: unsafe { lib.symbol(c"hipblasLtMatmulDescDestroy")? },
            matmul_desc_set_attribute: unsafe { lib.symbol(c"hipblasLtMatmulDescSetAttribute")? },
            matrix_layout_create: unsafe { lib.symbol(c"hipblasLtMatrixLayoutCreate")? },
            matrix_layout_destroy: unsafe { lib.symbol(c"hipblasLtMatrixLayoutDestroy")? },
            matmul_preference_create: unsafe { lib.symbol(c"hipblasLtMatmulPreferenceCreate")? },
            matmul_preference_destroy: unsafe { lib.symbol(c"hipblasLtMatmulPreferenceDestroy")? },
            matmul_preference_set_attribute: unsafe {
                lib.symbol(c"hipblasLtMatmulPreferenceSetAttribute")?
            },
            matmul_algo_get_heuristic: unsafe { lib.symbol(c"hipblasLtMatmulAlgoGetHeuristic")? },
            matmul: unsafe { lib.symbol(c"hipblasLtMatmul")? },
            _lib: lib,
        })
    }
}

impl ComgrFunctions {
    unsafe fn load(lib: Arc<DynamicLibrary>) -> Result<Self> {
        Ok(Self {
            status_string: unsafe { lib.symbol(c"amd_comgr_status_string")? },
            get_version: unsafe { lib.symbol(c"amd_comgr_get_version")? },
            create_data: unsafe { lib.symbol(c"amd_comgr_create_data")? },
            release_data: unsafe { lib.symbol(c"amd_comgr_release_data")? },
            set_data: unsafe { lib.symbol(c"amd_comgr_set_data")? },
            set_data_name: unsafe { lib.symbol(c"amd_comgr_set_data_name")? },
            get_data: unsafe { lib.symbol(c"amd_comgr_get_data")? },
            create_data_set: unsafe { lib.symbol(c"amd_comgr_create_data_set")? },
            destroy_data_set: unsafe { lib.symbol(c"amd_comgr_destroy_data_set")? },
            data_set_add: unsafe { lib.symbol(c"amd_comgr_data_set_add")? },
            action_data_count: unsafe { lib.symbol(c"amd_comgr_action_data_count")? },
            action_data_get_data: unsafe { lib.symbol(c"amd_comgr_action_data_get_data")? },
            create_action_info: unsafe { lib.symbol(c"amd_comgr_create_action_info")? },
            destroy_action_info: unsafe { lib.symbol(c"amd_comgr_destroy_action_info")? },
            action_info_set_isa_name: unsafe { lib.symbol(c"amd_comgr_action_info_set_isa_name")? },
            action_info_set_language: unsafe { lib.symbol(c"amd_comgr_action_info_set_language")? },
            action_info_set_option_list: unsafe {
                lib.symbol(c"amd_comgr_action_info_set_option_list")?
            },
            action_info_set_logging: unsafe { lib.symbol(c"amd_comgr_action_info_set_logging")? },
            do_action: unsafe { lib.symbol(c"amd_comgr_do_action")? },
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

fn rocm_path() -> std::path::PathBuf {
    std::env::var_os("ROCM_PATH")
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/opt/rocm"))
}

fn composable_kernel_availability() -> LibraryAvailability {
    let root = rocm_path();
    let headers = root.join("include").join("ck").join("ck.hpp");
    let gemm_lib = root.join("lib").join("libdevice_gemm_operations.a");
    let cmake = root
        .join("lib")
        .join("cmake")
        .join("composable_kernel")
        .join("composable_kernelConfig.cmake");
    if headers.is_file() && gemm_lib.is_file() {
        LibraryAvailability::available(format!(
            "headers={}, device_gemm={}, cmake={}",
            headers.display(),
            gemm_lib.display(),
            cmake.is_file()
        ))
    } else {
        LibraryAvailability::unavailable(format!(
            "missing Composable Kernel headers or device GEMM archive under {}",
            root.display()
        ))
    }
}

fn rocwmma_availability() -> LibraryAvailability {
    let root = rocm_path();
    let header = root.join("include").join("rocwmma").join("rocwmma.hpp");
    if header.is_file() {
        LibraryAvailability::available(format!("headers={}", header.display()))
    } else {
        LibraryAvailability::unavailable(format!(
            "missing rocWMMA headers under {}",
            root.join("include").display()
        ))
    }
}

impl ComgrFunctions {
    fn status_message(&self, status: AmdComgrStatus) -> String {
        let mut message = ptr::null();
        let result = unsafe { (self.status_string)(status, &mut message) };
        if result != AMD_COMGR_STATUS_SUCCESS || message.is_null() {
            format!("COMGR status {status}")
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

fn check_comgr(funcs: &ComgrFunctions, status: AmdComgrStatus, op: &str) -> Result<()> {
    if status == AMD_COMGR_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Library(format!(
            "{op} failed: {}",
            funcs.status_message(status)
        )))
    }
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

fn comgr_isa_name(arch: &str) -> String {
    if arch.starts_with("amdgcn-amd-amdhsa") {
        arch.to_string()
    } else {
        format!("amdgcn-amd-amdhsa--{arch}")
    }
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

fn check_hipblaslt(status: HipBlasLtStatus, op: &str) -> Result<()> {
    if status == HIPBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Library(format!(
            "{op} returned hipBLASLt status {status}"
        )))
    }
}

fn check_rocprim(status: RocPrimStatus, op: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else if status == ROCPRIM_SHIM_STATUS_UNAVAILABLE {
        Err(Error::Library(format!(
            "{op} is unavailable because the rocPRIM/hipCUB shim was built without ROCm algorithm headers"
        )))
    } else {
        Err(Error::Library(format!(
            "{op} returned HIP/rocPRIM status {status}"
        )))
    }
}

fn check_thrust(status: c_int, op: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else if status == ROCPRIM_SHIM_STATUS_UNAVAILABLE {
        Err(Error::Library(format!(
            "{op} is unavailable because the rocThrust shim was built without Thrust headers"
        )))
    } else {
        Err(Error::Library(format!(
            "{op} returned HIP/Thrust status {status}"
        )))
    }
}

fn validate_reduce_u32(input: &DeviceBuffer<u32>, output: &DeviceBuffer<u32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::Library(
            "rocPRIM reductions require at least one input element".to_string(),
        ));
    }
    validate_buffer_len("rocPRIM reduce output", output.len(), 1)
}

fn validate_scan_u32(
    output_name: &str,
    input: &DeviceBuffer<u32>,
    output: &DeviceBuffer<u32>,
) -> Result<()> {
    if input.is_empty() {
        return Err(Error::Library(
            "rocPRIM scans require at least one input element".to_string(),
        ));
    }
    validate_buffer_len(output_name, output.len(), input.len())
}

fn validate_temporary_storage(
    storage: &DeviceAlgorithmTemporaryStorage,
    required: usize,
) -> Result<()> {
    if storage.bytes() < required {
        Err(Error::Library(format!(
            "rocPRIM temporary storage has {} bytes, but {required} bytes are required",
            storage.bytes()
        )))
    } else {
        Ok(())
    }
}

fn query_rocprim_u32_storage(
    function: RocPrimUnary<u32>,
    op: &str,
    input: &DeviceBuffer<u32>,
    output: &DeviceBuffer<u32>,
) -> Result<usize> {
    let mut storage_bytes = 0usize;
    unsafe {
        check_rocprim(
            function(
                ptr::null_mut(),
                &mut storage_bytes,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
                ptr::null_mut(),
            ),
            op,
        )?;
    }
    Ok(storage_bytes)
}

fn call_rocprim_u32(
    function: RocPrimUnary<u32>,
    op: &str,
    stream: &Stream,
    temporary_storage: &DeviceAlgorithmTemporaryStorage,
    input: &DeviceBuffer<u32>,
    output: &DeviceBuffer<u32>,
) -> Result<()> {
    let mut storage_bytes = temporary_storage.bytes();
    unsafe {
        check_rocprim(
            function(
                temporary_storage.as_mut_ptr(),
                &mut storage_bytes,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
                stream.as_raw(),
            ),
            op,
        )
    }
}

fn validate_reduce<T>(input: &DeviceBuffer<T>, output: &DeviceBuffer<T>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::Library(
            "rocPRIM reductions require at least one input element".to_string(),
        ));
    }
    validate_buffer_len("rocPRIM reduce output", output.len(), 1)
}

fn validate_scan<T>(
    output_name: &str,
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
) -> Result<()> {
    if input.is_empty() {
        return Err(Error::Library(
            "rocPRIM scans require at least one input element".to_string(),
        ));
    }
    validate_buffer_len(output_name, output.len(), input.len())
}

fn query_rocprim_unary_storage<T>(
    function: RocPrimUnary<T>,
    op: &str,
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
) -> Result<usize> {
    let mut storage_bytes = 0usize;
    unsafe {
        check_rocprim(
            function(
                ptr::null_mut(),
                &mut storage_bytes,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
                ptr::null_mut(),
            ),
            op,
        )?;
    }
    Ok(storage_bytes)
}

fn query_rocprim_exclusive_storage<T: Copy>(
    function: RocPrimExclusive<T>,
    op: &str,
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
    initial_value: T,
) -> Result<usize> {
    validate_scan("rocPRIM exclusive scan output", input, output)?;
    let mut storage_bytes = 0usize;
    unsafe {
        check_rocprim(
            function(
                ptr::null_mut(),
                &mut storage_bytes,
                input.as_ptr(),
                output.as_mut_ptr(),
                initial_value,
                input.len(),
                ptr::null_mut(),
            ),
            op,
        )?;
    }
    Ok(storage_bytes)
}

fn call_rocprim_unary<T>(
    function: RocPrimUnary<T>,
    op: &str,
    stream: &Stream,
    temporary_storage: &DeviceAlgorithmTemporaryStorage,
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
) -> Result<()> {
    let mut storage_bytes = temporary_storage.bytes();
    unsafe {
        check_rocprim(
            function(
                temporary_storage.as_mut_ptr(),
                &mut storage_bytes,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
                stream.as_raw(),
            ),
            op,
        )
    }
}

fn call_rocprim_exclusive<T: Copy>(
    function: RocPrimExclusive<T>,
    op: &str,
    stream: &Stream,
    temporary_storage: &DeviceAlgorithmTemporaryStorage,
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
    initial_value: T,
) -> Result<()> {
    let mut storage_bytes = temporary_storage.bytes();
    unsafe {
        check_rocprim(
            function(
                temporary_storage.as_mut_ptr(),
                &mut storage_bytes,
                input.as_ptr(),
                output.as_mut_ptr(),
                initial_value,
                input.len(),
                stream.as_raw(),
            ),
            op,
        )
    }
}

fn validate_select_flagged_u32(
    input: &DeviceBuffer<u32>,
    flags: &DeviceBuffer<u8>,
    output: &DeviceBuffer<u32>,
    selected_count: &DeviceBuffer<u32>,
) -> Result<()> {
    validate_buffer_len("rocPRIM select flags", flags.len(), input.len())?;
    validate_buffer_len("rocPRIM select output", output.len(), input.len())?;
    validate_buffer_len("rocPRIM select count", selected_count.len(), 1)
}

fn query_rocprim_select_flagged_u32_storage(
    input: &DeviceBuffer<u32>,
    flags: &DeviceBuffer<u8>,
    output: &DeviceBuffer<u32>,
    selected_count: &DeviceBuffer<u32>,
) -> Result<usize> {
    validate_select_flagged_u32(input, flags, output, selected_count)?;
    let mut storage_bytes = 0usize;
    unsafe {
        check_rocprim(
            rocm_oxide_rocprim_select_flagged_u32(
                ptr::null_mut(),
                &mut storage_bytes,
                input.as_ptr(),
                flags.as_ptr(),
                output.as_mut_ptr(),
                selected_count.as_mut_ptr(),
                input.len(),
                ptr::null_mut(),
            ),
            "rocPRIM select_flagged_u32 storage query",
        )?;
    }
    Ok(storage_bytes)
}

fn call_rocprim_select_flagged_u32(
    stream: &Stream,
    temporary_storage: &DeviceAlgorithmTemporaryStorage,
    input: &DeviceBuffer<u32>,
    flags: &DeviceBuffer<u8>,
    output: &DeviceBuffer<u32>,
    selected_count: &DeviceBuffer<u32>,
) -> Result<()> {
    let mut storage_bytes = temporary_storage.bytes();
    unsafe {
        check_rocprim(
            rocm_oxide_rocprim_select_flagged_u32(
                temporary_storage.as_mut_ptr(),
                &mut storage_bytes,
                input.as_ptr(),
                flags.as_ptr(),
                output.as_mut_ptr(),
                selected_count.as_mut_ptr(),
                input.len(),
                stream.as_raw(),
            ),
            "rocPRIM select_flagged_u32",
        )
    }
}

fn matrix_elements(leading_dim: u32, columns: u32) -> Result<usize> {
    (leading_dim as usize)
        .checked_mul(columns as usize)
        .ok_or_else(|| Error::Library("matrix element count overflows usize".to_string()))
}

fn hipblaslt_matrix_elements(layout: HipBlasLtMatrixLayout) -> Result<usize> {
    let leading_dim = usize::try_from(layout.leading_dim).map_err(|_| {
        Error::Library(format!(
            "hipBLASLt leading dimension {} cannot be represented as usize",
            layout.leading_dim
        ))
    })?;
    let columns = usize::try_from(layout.cols).map_err(|_| {
        Error::Library(format!(
            "hipBLASLt column count {} cannot be represented as usize",
            layout.cols
        ))
    })?;
    leading_dim
        .checked_mul(columns)
        .ok_or_else(|| Error::Library("hipBLASLt matrix element count overflows usize".to_string()))
}

fn best_hipblaslt_heuristic(
    results: &[HipBlasLtMatmulHeuristicResultRaw],
    returned_algo_count: i32,
) -> Result<HipBlasLtMatmulHeuristicResultRaw> {
    if returned_algo_count <= 0 {
        return Err(Error::Library(
            "hipBLASLt returned no SGEMM algorithms for the requested problem".to_string(),
        ));
    }
    let best = results[0];
    if best.state != HIPBLAS_STATUS_SUCCESS {
        return Err(Error::Library(format!(
            "hipBLASLt best SGEMM algorithm returned state {}",
            best.state
        )));
    }
    Ok(best)
}

fn validate_hipblaslt_workspace(workspace_size: usize, workspace_limit_bytes: u64) -> Result<()> {
    let workspace_size_u64 = u64::try_from(workspace_size).map_err(|_| {
        Error::Library(format!(
            "hipBLASLt returned workspace size {workspace_size} that cannot fit u64"
        ))
    })?;
    if workspace_size_u64 > workspace_limit_bytes {
        return Err(Error::Library(format!(
            "hipBLASLt returned workspace size {workspace_size_u64} bytes, exceeding requested limit {workspace_limit_bytes} bytes"
        )));
    }
    if workspace_size_u64 > HIPBLASLT_MAX_WORKSPACE_BYTES {
        return Err(Error::Library(format!(
            "hipBLASLt returned workspace size {workspace_size_u64} bytes, exceeding ROCm-Oxide automatic workspace cap {HIPBLASLT_MAX_WORKSPACE_BYTES} bytes"
        )));
    }
    Ok(())
}

fn summarize_hipblaslt_heuristics(
    requested_algo_count: i32,
    returned_algo_count: i32,
    results: &[HipBlasLtMatmulHeuristicResultRaw],
    workspace_limit_bytes: u64,
) -> HipBlasLtHeuristicSummary {
    let best = (returned_algo_count > 0).then_some(results[0]);
    HipBlasLtHeuristicSummary {
        requested_algo_count,
        returned_algo_count,
        best_workspace_bytes: best.map(|result| result.workspace_size),
        best_state: best.map(|result| result.state),
        best_waves_count: best.map(|result| result.waves_count),
        workspace_limit_bytes,
    }
}

fn c_int_from_u32(label: &str, value: u32) -> Result<c_int> {
    c_int::try_from(value)
        .map_err(|_| Error::Library(format!("{label} value {value} exceeds rocBLAS int range")))
}

fn positive_i64_from_u64(label: &str, value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::Library(format!("{label} value {value} exceeds i64 range")))
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
        assert!(!report.hipblaslt.detail.is_empty());
        assert!(!report.comgr.detail.is_empty());
        assert!(!report.rocprim.detail.is_empty());
    }

    #[test]
    fn matrix_integration_report_can_be_queried() {
        let report = MatrixIntegrationReport::query();
        assert!(!report.hipblaslt.detail.is_empty());
        assert!(!report.composable_kernel.detail.is_empty());
        assert!(!report.rocwmma.detail.is_empty());
    }

    #[test]
    fn hipblaslt_sgemm_problem_validates_dimensions() {
        assert!(HipBlasLtMatmulProblem::sgemm_nn(0, 16, 16, 0).is_err());
        assert!(
            HipBlasLtMatmulProblem::sgemm_nn_with_leading_dimensions(
                16, 16, 16, 15, 16, 16, 16, 0,
            )
            .is_err()
        );

        let problem = HipBlasLtMatmulProblem::sgemm_nn(16, 32, 64, 1024)
            .expect("valid SGEMM descriptor should be accepted");
        assert_eq!(problem.a.rows, 16);
        assert_eq!(problem.a.cols, 64);
        assert_eq!(problem.b.rows, 64);
        assert_eq!(problem.b.cols, 32);
        assert_eq!(problem.max_workspace_bytes, 1024);
    }

    #[test]
    fn hipblaslt_sgemm_problem_rejects_excess_workspace_cap() {
        let err = HipBlasLtMatmulProblem::sgemm_nn(16, 16, 16, HIPBLASLT_MAX_WORKSPACE_BYTES + 1)
            .expect_err("oversized automatic workspace cap should fail");
        assert!(err.to_string().contains("automatic workspace"));
    }

    #[test]
    fn hipblaslt_sgemm_problem_rejects_each_invalid_leading_dimension() {
        assert!(
            HipBlasLtMatmulProblem::sgemm_nn_with_leading_dimensions(
                16, 16, 16, 15, 16, 16, 16, 0,
            )
            .expect_err("A leading dimension smaller than m should fail")
            .to_string()
            .contains("A leading dimension")
        );
        assert!(
            HipBlasLtMatmulProblem::sgemm_nn_with_leading_dimensions(
                16, 16, 16, 16, 15, 16, 16, 0,
            )
            .expect_err("B leading dimension smaller than k should fail")
            .to_string()
            .contains("B leading dimension")
        );
        assert!(
            HipBlasLtMatmulProblem::sgemm_nn_with_leading_dimensions(
                16, 16, 16, 16, 16, 15, 16, 0,
            )
            .expect_err("C leading dimension smaller than m should fail")
            .to_string()
            .contains("C leading dimension")
        );
        assert!(
            HipBlasLtMatmulProblem::sgemm_nn_with_leading_dimensions(
                16, 16, 16, 16, 16, 16, 15, 0,
            )
            .expect_err("D leading dimension smaller than m should fail")
            .to_string()
            .contains("D leading dimension")
        );
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

    #[test]
    fn hipblaslt_handle_smoke_if_library_is_available() {
        let Ok(lt) = HipBlasLt::open() else {
            return;
        };
        let handle = lt
            .create_handle()
            .expect("available hipBLASLt library should create a handle");
        assert!(handle.version().expect("hipBLASLt version should query") > 0);
    }

    #[test]
    fn hipblaslt_sgemm_heuristic_smoke_if_library_is_available() {
        let Ok(lt) = HipBlasLt::open() else {
            return;
        };
        let Ok(handle) = lt.create_handle() else {
            return;
        };
        let problem = HipBlasLtMatmulProblem::sgemm_nn(32, 32, 32, 4 * 1024 * 1024)
            .expect("valid SGEMM descriptor should be accepted");
        let summary = handle
            .sgemm_nn_heuristics(problem, 8)
            .expect("available hipBLASLt library should query SGEMM heuristics");
        assert_eq!(summary.requested_algo_count, 8);
        assert!(summary.returned_algo_count >= 0);
        if let Some(workspace) = summary.best_workspace_bytes {
            assert!(workspace <= summary.workspace_limit_bytes as usize);
        }
    }

    #[test]
    fn hipblaslt_sgemm_executes_if_library_is_available() {
        let Ok(lt) = HipBlasLt::open() else {
            return;
        };
        let Ok(handle) = lt.create_handle() else {
            return;
        };

        let m = 64usize;
        let n = 64usize;
        let k = 64usize;
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            a[i + i * m] = 1.0;
        }
        let b = (0..k * n)
            .map(|index| (index as f32) * 0.25 + 1.0)
            .collect::<Vec<_>>();
        let c = vec![0.0f32; m * n];
        let d = DeviceBuffer::<f32>::new(m * n).expect("output allocation should succeed");
        let d_a = DeviceBuffer::from_slice(&a).expect("A allocation should succeed");
        let d_b = DeviceBuffer::from_slice(&b).expect("B allocation should succeed");
        let d_c = DeviceBuffer::from_slice(&c).expect("C allocation should succeed");
        let problem =
            HipBlasLtMatmulProblem::sgemm_nn(m as u64, n as u64, k as u64, 4 * 1024 * 1024)
                .expect("valid SGEMM descriptor should be accepted");

        let summary = handle
            .sgemm_nn(problem, 1.0, &d_a, &d_b, 0.0, &d_c, &d, 8)
            .expect("available hipBLASLt library should execute SGEMM");
        assert!(summary.returned_algo_count > 0);

        let out = d.copy_to_vec().expect("D copy should succeed");
        for (actual, expected) in out.iter().zip(&b) {
            assert!((actual - expected).abs() < 0.001);
        }
    }

    #[test]
    fn hipblaslt_heuristics_reject_nonpositive_algorithm_count_if_library_is_available() {
        let Ok(lt) = HipBlasLt::open() else {
            return;
        };
        let Ok(handle) = lt.create_handle() else {
            return;
        };
        let problem = HipBlasLtMatmulProblem::sgemm_nn(16, 16, 16, 0)
            .expect("valid SGEMM descriptor should be accepted");
        let err = handle
            .sgemm_nn_heuristics(problem, 0)
            .expect_err("nonpositive algorithm count should fail before FFI");
        assert!(err.to_string().contains("requested algorithm count"));
    }

    #[test]
    fn hipblaslt_heuristics_reject_excess_algorithm_count_if_library_is_available() {
        let Ok(lt) = HipBlasLt::open() else {
            return;
        };
        let Ok(handle) = lt.create_handle() else {
            return;
        };
        let problem = HipBlasLtMatmulProblem::sgemm_nn(16, 16, 16, 0)
            .expect("valid SGEMM descriptor should be accepted");
        let err = handle
            .sgemm_nn_heuristics(problem, HIPBLASLT_MAX_REQUESTED_ALGOS + 1)
            .expect_err("excess algorithm count should fail before FFI");
        assert!(err.to_string().contains("exceeds ROCm-Oxide cap"));
    }

    #[test]
    fn comgr_version_smoke_if_library_is_available() {
        let Ok(comgr) = Comgr::open() else {
            return;
        };
        let version = comgr.version();
        assert!(version.major > 0 || version.minor > 0);
    }

    #[test]
    fn comgr_compile_hip_source_smoke_if_available() {
        let Ok(comgr) = Comgr::open() else {
            return;
        };
        let Ok(device) = crate::Device::first() else {
            return;
        };
        let source = r#"
#include <hip/hip_runtime.h>
extern "C" __global__
void comgr_smoke(float* out) {
    out[0] = 7.0f;
}
"#;
        let code_object = comgr
            .compile_hip_source_to_code_object(source, device.arch(), &[])
            .expect("COMGR should compile and link a simple HIP kernel");
        assert!(code_object.starts_with(b"\x7fELF"));
    }

    #[test]
    fn rocprim_reduce_and_scan_smoke_if_available() {
        let Ok(rocprim) = RocPrim::open() else {
            return;
        };
        let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4]).expect("upload should work");

        let reduced = DeviceBuffer::<u32>::new(1).expect("reduce output allocation should work");
        rocprim
            .reduce_sum_u32(&input, &reduced)
            .expect("rocPRIM reduce should work");
        assert_eq!(reduced.copy_to_vec().expect("reduce download"), [10]);

        let inclusive =
            DeviceBuffer::<u32>::new(input.len()).expect("inclusive output allocation should work");
        rocprim
            .inclusive_sum_u32(&input, &inclusive)
            .expect("rocPRIM inclusive scan should work");
        assert_eq!(
            inclusive.copy_to_vec().expect("inclusive download"),
            [1, 3, 6, 10]
        );

        let exclusive =
            DeviceBuffer::<u32>::new(input.len()).expect("exclusive output allocation should work");
        rocprim
            .exclusive_sum_u32(&input, &exclusive, 0)
            .expect("rocPRIM exclusive scan should work");
        assert_eq!(
            exclusive.copy_to_vec().expect("exclusive download"),
            [0, 1, 3, 6]
        );

        let signed_input = DeviceBuffer::from_slice(&[-3i32, 4, 7, -2]).expect("upload i32");
        let signed_reduced = DeviceBuffer::<i32>::new(1).expect("i32 reduce output");
        rocprim
            .reduce_sum_i32(&signed_input, &signed_reduced)
            .expect("rocPRIM i32 reduce should work");
        assert_eq!(signed_reduced.copy_to_vec().expect("i32 reduce"), [6]);
        let signed_scan = DeviceBuffer::<i32>::new(signed_input.len()).expect("i32 scan output");
        rocprim
            .inclusive_sum_i32(&signed_input, &signed_scan)
            .expect("rocPRIM i32 scan should work");
        assert_eq!(signed_scan.copy_to_vec().expect("i32 scan"), [-3, 1, 8, 6]);

        let float_input = DeviceBuffer::from_slice(&[1.0f32, 2.5, -0.5]).expect("upload f32");
        let float_reduced = DeviceBuffer::<f32>::new(1).expect("f32 reduce output");
        rocprim
            .reduce_sum_f32(&float_input, &float_reduced)
            .expect("rocPRIM f32 reduce should work");
        assert_eq!(float_reduced.copy_to_vec().expect("f32 reduce"), [3.0]);
        let float_exclusive = DeviceBuffer::<f32>::new(float_input.len()).expect("f32 scan output");
        rocprim
            .exclusive_sum_f32(&float_input, &float_exclusive, 0.0)
            .expect("rocPRIM f32 exclusive scan should work");
        assert_eq!(
            float_exclusive.copy_to_vec().expect("f32 exclusive"),
            [0.0, 1.0, 3.5]
        );
    }

    #[test]
    fn rocprim_sort_select_and_transform_smoke_if_available() {
        let Ok(rocprim) = RocPrim::open() else {
            return;
        };
        let sort_input = DeviceBuffer::from_slice(&[9u32, 2, 7, 2, 1]).expect("sort upload");
        let sort_output = DeviceBuffer::<u32>::new(sort_input.len()).expect("sort output");
        rocprim
            .sort_keys_u32(&sort_input, &sort_output)
            .expect("rocPRIM sort should work");
        assert_eq!(
            sort_output.copy_to_vec().expect("sort download"),
            [1, 2, 2, 7, 9]
        );

        let select_input =
            DeviceBuffer::from_slice(&[10u32, 20, 30, 40, 50]).expect("select upload");
        let flags = DeviceBuffer::from_slice(&[1u8, 0, 1, 0, 1]).expect("flags upload");
        let selected = DeviceBuffer::<u32>::new(select_input.len()).expect("select output");
        let selected_count = DeviceBuffer::<u32>::new(1).expect("select count");
        rocprim
            .select_flagged_u32(&select_input, &flags, &selected, &selected_count)
            .expect("rocPRIM flagged select should work");
        assert_eq!(
            selected_count.copy_to_vec().expect("select count download"),
            [3]
        );
        assert_eq!(
            &selected.copy_to_vec().expect("select download")[..3],
            [10, 30, 50]
        );

        let transform_output =
            DeviceBuffer::<u32>::new(select_input.len()).expect("transform output");
        rocprim
            .transform_add_u32(&select_input, &transform_output, 7)
            .expect("rocPRIM transform should work");
        assert_eq!(
            transform_output.copy_to_vec().expect("transform download"),
            [17, 27, 37, 47, 57]
        );
    }

    #[test]
    fn rocprim_rejects_short_temporary_storage_if_available() {
        let Ok(rocprim) = RocPrim::open() else {
            return;
        };
        let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4]).expect("upload should work");
        let output = DeviceBuffer::<u32>::new(1).expect("reduce output allocation should work");
        let required = rocprim
            .reduce_sum_u32_storage_bytes(&input, &output)
            .expect("storage query should work");
        if required == 0 {
            return;
        }
        let storage = DeviceAlgorithmTemporaryStorage::new(required - 1)
            .expect("temp allocation should work");
        let err = rocprim
            .reduce_sum_u32_on_stream(
                &Stream::new().expect("stream should work"),
                &storage,
                &input,
                &output,
            )
            .expect_err("short temp storage should fail before launch");
        assert!(err.to_string().contains("temporary storage"));
    }
}
