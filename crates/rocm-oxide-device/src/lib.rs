#![no_std]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(gpu_intrinsics)]
#![feature(gpu_launch_sized_workgroup_mem)]
#![feature(stdarch_amdgpu)]

use core::arch::amdgpu;
use core::intrinsics::gpu::{amdgpu_dispatch_ptr, gpu_launch_sized_workgroup_mem};
use core::sync::atomic::Ordering;
use core::{marker::PhantomData, ptr};

pub mod math {
    const FRAC_PI_2_F32: f32 = core::f32::consts::FRAC_PI_2;
    const FRAC_PI_4_F32: f32 = core::f32::consts::FRAC_PI_4;
    const PI_F64: f64 = core::f64::consts::PI;
    const TAU_F64: f64 = core::f64::consts::TAU;
    const FRAC_PI_2_F64: f64 = core::f64::consts::FRAC_PI_2;
    const FRAC_PI_4_F64: f64 = core::f64::consts::FRAC_PI_4;

    #[inline(always)]
    pub fn sqrt_f32(value: f32) -> f32 {
        core::intrinsics::sqrtf32(value)
    }

    #[inline(always)]
    pub fn sqrt_f64(value: f64) -> f64 {
        core::intrinsics::sqrtf64(value)
    }

    #[inline(always)]
    pub fn rsqrt_f32(value: f32) -> f32 {
        1.0 / sqrt_f32(value)
    }

    #[inline(always)]
    pub fn rsqrt_f64(value: f64) -> f64 {
        1.0 / sqrt_f64(value)
    }

    #[inline(always)]
    pub fn sin_f32(value: f32) -> f32 {
        core::intrinsics::sinf32(value)
    }

    #[inline(always)]
    pub fn sin_f64(value: f64) -> f64 {
        let x = reduce_angle_f64(value);
        let x2 = x * x;
        x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))))
    }

    #[inline(always)]
    pub fn cos_f32(value: f32) -> f32 {
        core::intrinsics::cosf32(value)
    }

    #[inline(always)]
    pub fn cos_f64(value: f64) -> f64 {
        let x = reduce_angle_f64(value);
        let x2 = x * x;
        1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0)))
    }

    #[inline(always)]
    pub fn atan_f32(value: f32) -> f32 {
        if value != value {
            return value;
        }
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let abs = if value < 0.0 { -value } else { value };
        if abs == f32::INFINITY {
            return sign * FRAC_PI_2_F32;
        }
        let reduced = if abs > 1.0 { 1.0 / abs } else { abs };
        let core = atan_unit_f32(reduced);
        sign * if abs > 1.0 {
            FRAC_PI_2_F32 - core
        } else {
            core
        }
    }

    #[inline(always)]
    pub fn atan_f64(value: f64) -> f64 {
        if value != value {
            return value;
        }
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let abs = if value < 0.0 { -value } else { value };
        if abs == f64::INFINITY {
            return sign * FRAC_PI_2_F64;
        }
        let reduced = if abs > 1.0 { 1.0 / abs } else { abs };
        let core = atan_unit_f64(reduced);
        sign * if abs > 1.0 {
            FRAC_PI_2_F64 - core
        } else {
            core
        }
    }

    #[inline(always)]
    pub fn min_f32(a: f32, b: f32) -> f32 {
        core::intrinsics::minimumf32(a, b)
    }

    #[inline(always)]
    pub fn max_f32(a: f32, b: f32) -> f32 {
        core::intrinsics::maximumf32(a, b)
    }

    #[inline(always)]
    pub fn min_f64(a: f64, b: f64) -> f64 {
        core::intrinsics::minimumf64(a, b)
    }

    #[inline(always)]
    pub fn max_f64(a: f64, b: f64) -> f64 {
        core::intrinsics::maximumf64(a, b)
    }

    #[inline(always)]
    fn atan_unit_f32(value: f32) -> f32 {
        let abs = if value < 0.0 { -value } else { value };
        FRAC_PI_4_F32 * value - value * (abs - 1.0) * (0.2447 + 0.0663 * abs)
    }

    #[inline(always)]
    fn atan_unit_f64(value: f64) -> f64 {
        let abs = if value < 0.0 { -value } else { value };
        FRAC_PI_4_F64 * value - value * (abs - 1.0) * (0.2447 + 0.0663 * abs)
    }

    #[inline(always)]
    fn reduce_angle_f64(value: f64) -> f64 {
        if value != value || value == f64::INFINITY || value == f64::NEG_INFINITY {
            return value;
        }
        let mut x = value;
        while x > PI_F64 {
            x -= TAU_F64;
        }
        while x < -PI_F64 {
            x += TAU_F64;
        }
        x
    }
}

pub mod atomic {
    use core::marker::PhantomData;
    use core::sync::atomic::{AtomicU32, Ordering};

    #[repr(u8)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum AtomicScope {
        Workgroup = 0,
        Device = 1,
        System = 2,
    }

    #[repr(u8)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum AtomicOrdering {
        Relaxed = 0,
        Acquire = 1,
        Release = 2,
        AcqRel = 3,
        SeqCst = 4,
    }

    impl AtomicOrdering {
        #[inline(always)]
        pub fn as_core(self) -> Ordering {
            match self {
                Self::Relaxed => Ordering::Relaxed,
                Self::Acquire => Ordering::Acquire,
                Self::Release => Ordering::Release,
                Self::AcqRel => Ordering::AcqRel,
                Self::SeqCst => Ordering::SeqCst,
            }
        }
    }

    impl From<Ordering> for AtomicOrdering {
        #[inline(always)]
        fn from(ordering: Ordering) -> Self {
            match ordering {
                Ordering::Relaxed => Self::Relaxed,
                Ordering::Acquire => Self::Acquire,
                Ordering::Release => Self::Release,
                Ordering::AcqRel => Self::AcqRel,
                Ordering::SeqCst => Self::SeqCst,
                _ => Self::SeqCst,
            }
        }
    }

    pub trait Scope {
        const SCOPE: AtomicScope;
    }

    pub enum Workgroup {}
    pub enum Device {}
    pub enum System {}

    impl Scope for Workgroup {
        const SCOPE: AtomicScope = AtomicScope::Workgroup;
    }

    impl Scope for Device {
        const SCOPE: AtomicScope = AtomicScope::Device;
    }

    impl Scope for System {
        const SCOPE: AtomicScope = AtomicScope::System;
    }

    #[repr(transparent)]
    pub struct AtomicU32Ref<S: Scope> {
        inner: AtomicU32,
        _scope: PhantomData<S>,
    }

    impl<S: Scope> AtomicU32Ref<S> {
        #[inline(always)]
        pub unsafe fn from_ptr<'a>(ptr: *mut u32) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub unsafe fn from_const_ptr<'a>(ptr: *const u32) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub const fn scope() -> AtomicScope {
            S::SCOPE
        }

        #[inline(always)]
        pub fn load(&self, ordering: AtomicOrdering) -> u32 {
            let _ = S::SCOPE;
            self.inner.load(ordering.as_core())
        }

        #[inline(always)]
        pub fn store(&self, value: u32, ordering: AtomicOrdering) {
            let _ = S::SCOPE;
            self.inner.store(value, ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: u32, ordering: AtomicOrdering) -> u32 {
            let _ = S::SCOPE;
            self.inner.fetch_add(value, ordering.as_core())
        }
    }

    pub type WorkgroupAtomicU32 = AtomicU32Ref<Workgroup>;
    pub type DeviceAtomicU32 = AtomicU32Ref<Device>;
    pub type SystemAtomicU32 = AtomicU32Ref<System>;

    #[inline(always)]
    pub unsafe fn atomic_add_u32_scoped(
        ptr: *mut u32,
        value: u32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> u32 {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicU32::from_ptr(ptr) }.fetch_add(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicU32::from_ptr(ptr) }
                .fetch_add(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicU32::from_ptr(ptr) }
                .fetch_add(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_store_u32_scoped(
        ptr: *mut u32,
        value: u32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicU32::from_ptr(ptr) }.store(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicU32::from_ptr(ptr) }
                .store(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicU32::from_ptr(ptr) }
                .store(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_load_u32_scoped(
        ptr: *const u32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> u32 {
        match scope {
            AtomicScope::Workgroup => unsafe { WorkgroupAtomicU32::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::Device => unsafe { DeviceAtomicU32::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::System => unsafe { SystemAtomicU32::from_const_ptr(ptr) }
                .load(ordering),
        }
    }
}

pub use atomic::{
    AtomicOrdering, AtomicScope, DeviceAtomicU32, SystemAtomicU32, WorkgroupAtomicU32,
};

#[inline(always)]
pub fn thread_idx_x() -> u32 {
    amdgpu::workitem_id_x()
}

#[inline(always)]
pub fn thread_idx_y() -> u32 {
    amdgpu::workitem_id_y()
}

#[inline(always)]
pub fn thread_idx_z() -> u32 {
    amdgpu::workitem_id_z()
}

#[inline(always)]
pub fn block_idx_x() -> u32 {
    amdgpu::workgroup_id_x()
}

#[inline(always)]
pub fn block_idx_y() -> u32 {
    amdgpu::workgroup_id_y()
}

#[inline(always)]
pub fn block_idx_z() -> u32 {
    amdgpu::workgroup_id_z()
}

#[inline(always)]
pub fn block_dim_x() -> u32 {
    dispatch_packet().workgroup_size_x as u32
}

#[inline(always)]
pub fn block_dim_y() -> u32 {
    dispatch_packet().workgroup_size_y as u32
}

#[inline(always)]
pub fn block_dim_z() -> u32 {
    dispatch_packet().workgroup_size_z as u32
}

#[inline(always)]
pub fn grid_dim_x() -> u32 {
    dispatch_packet().grid_size_x.div_ceil(block_dim_x())
}

#[inline(always)]
pub fn grid_dim_y() -> u32 {
    dispatch_packet().grid_size_y.div_ceil(block_dim_y())
}

#[inline(always)]
pub fn grid_dim_z() -> u32 {
    dispatch_packet().grid_size_z.div_ceil(block_dim_z())
}

#[inline(always)]
pub fn global_id_x() -> usize {
    (block_idx_x() as usize * block_dim_x() as usize) + thread_idx_x() as usize
}

#[inline(always)]
pub fn global_id_y() -> usize {
    (block_idx_y() as usize * block_dim_y() as usize) + thread_idx_y() as usize
}

#[inline(always)]
pub fn global_id_z() -> usize {
    (block_idx_z() as usize * block_dim_z() as usize) + thread_idx_z() as usize
}

#[inline(always)]
pub fn dispatch_id() -> u64 {
    amdgpu::dispatch_id()
}

#[inline(always)]
pub fn wavefront_size() -> u32 {
    amdgpu::wavefrontsize()
}

#[inline(always)]
pub fn lane_id() -> u32 {
    amdgpu::mbcnt_hi(u32::MAX, amdgpu::mbcnt_lo(u32::MAX, 0))
}

#[inline(always)]
pub fn wave_id_in_workgroup() -> u32 {
    amdgpu::s_get_waveid_in_workgroup()
}

#[inline(always)]
pub fn is_first_lane() -> bool {
    lane_id() == 0
}

#[inline(always)]
pub fn workgroup_barrier() {
    amdgpu::s_barrier()
}

#[inline(always)]
pub fn wave_barrier() {
    amdgpu::wave_barrier()
}

#[inline(always)]
pub fn ballot(predicate: bool) -> u64 {
    amdgpu::ballot(predicate)
}

#[inline(always)]
pub fn read_first_lane_u32(value: u32) -> u32 {
    amdgpu::readfirstlane_u32(value)
}

#[inline(always)]
pub fn wave_reduce_add_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_add::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_min_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_umin::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_max_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_umax::<0>(value)
}

#[inline(always)]
pub unsafe fn atomic_add_u32(ptr: *mut u32, value: u32) -> u32 {
    unsafe {
        atomic::atomic_add_u32_scoped(
            ptr,
            value,
            AtomicScope::Device,
            AtomicOrdering::Relaxed,
        )
    }
}

#[inline(always)]
pub unsafe fn atomic_store_u32(ptr: *mut u32, value: u32, ordering: Ordering) {
    unsafe { DeviceAtomicU32::from_ptr(ptr) }.store(value, ordering.into());
}

#[inline(always)]
pub unsafe fn atomic_load_u32(ptr: *const u32, ordering: Ordering) -> u32 {
    unsafe { DeviceAtomicU32::from_const_ptr(ptr) }.load(ordering.into())
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceSlice<T> {
    ptr: *const T,
    len: usize,
}

impl<T> DeviceSlice<T> {
    #[inline(always)]
    pub const unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self { ptr, len }
    }

    #[inline(always)]
    pub const fn as_ptr(self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub const fn len(self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(self, index: usize) -> *const T {
        unsafe { self.ptr.add(index) }
    }

    #[inline(always)]
    pub fn get(self, index: usize) -> Option<*const T> {
        if index < self.len {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub unsafe fn read_unchecked(self, index: usize) -> T
    where
        T: Copy,
    {
        unsafe { ptr::read(self.get_unchecked(index)) }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceSliceMut<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceSliceMut<T> {
    #[inline(always)]
    pub const unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }

    #[inline(always)]
    pub const fn as_ptr(self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub const fn as_mut_ptr(self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub const fn len(self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub const fn as_const(self) -> DeviceSlice<T> {
        DeviceSlice {
            ptr: self.ptr,
            len: self.len,
        }
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(self, index: usize) -> *mut T {
        unsafe { self.ptr.add(index) }
    }

    #[inline(always)]
    pub fn get(self, index: usize) -> Option<*mut T> {
        if index < self.len {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub unsafe fn read_unchecked(self, index: usize) -> T
    where
        T: Copy,
    {
        unsafe { ptr::read(self.get_unchecked(index)) }
    }

    #[inline(always)]
    pub unsafe fn write_unchecked(self, index: usize, value: T) {
        unsafe {
            ptr::write(self.get_unchecked(index), value);
        }
    }
}

pub struct DynamicSharedMem<T> {
    _marker: PhantomData<T>,
}

impl<T> DynamicSharedMem<T> {
    #[inline(always)]
    pub unsafe fn get() -> *mut T {
        gpu_launch_sized_workgroup_mem::<T>()
    }

    #[inline(always)]
    pub unsafe fn offset(byte_offset: usize) -> *mut T {
        unsafe { Self::get().cast::<u8>().add(byte_offset).cast::<T>() }
    }
}

#[repr(C)]
struct HsaKernelDispatchPacket {
    full_header: u32,
    workgroup_size_x: u16,
    workgroup_size_y: u16,
    workgroup_size_z: u16,
    reserved0: u16,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: *const u8,
    reserved1: u32,
    reserved2: u64,
    completion_signal: u64,
}

#[inline(always)]
fn dispatch_packet() -> &'static HsaKernelDispatchPacket {
    unsafe { &*(amdgpu_dispatch_ptr().cast::<HsaKernelDispatchPacket>()) }
}
