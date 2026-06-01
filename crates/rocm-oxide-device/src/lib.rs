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

pub mod debug {
    use super::amdgpu;

    #[inline(always)]
    pub fn dispatch_id() -> u64 {
        super::dispatch_id()
    }

    #[inline(always)]
    pub fn program_counter() -> u64 {
        amdgpu::s_getpc() as u64
    }

    #[inline(always)]
    pub fn sleep<const COUNT: u32>() {
        amdgpu::s_sleep::<COUNT>()
    }

    #[inline(always)]
    pub fn trap<const CODE: u32>() -> ! {
        amdgpu::s_sethalt::<CODE>()
    }

    #[inline(always)]
    pub fn breakpoint() -> ! {
        trap::<2>()
    }

    #[inline(always)]
    pub fn assert_or_trap(condition: bool) {
        if !condition {
            trap::<1>();
        }
    }
}

#[macro_export]
macro_rules! gpu_assert {
    ($condition:expr $(,)?) => {
        $crate::debug::assert_or_trap($condition)
    };
}

pub mod atomic {
    use core::marker::PhantomData;
    use core::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};

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

    #[inline(always)]
    fn compare_exchange_failure_ordering(ordering: AtomicOrdering) -> Ordering {
        match ordering.as_core() {
            Ordering::Relaxed | Ordering::Release => Ordering::Relaxed,
            Ordering::Acquire | Ordering::AcqRel => Ordering::Acquire,
            Ordering::SeqCst => Ordering::SeqCst,
            _ => Ordering::SeqCst,
        }
    }

    unsafe extern "C" {
        #[link_name = "__rocm_oxide_atomic_scope_workgroup"]
        fn mark_atomic_scope_workgroup(ptr: *const u32);
        #[link_name = "__rocm_oxide_atomic_scope_device"]
        fn mark_atomic_scope_device(ptr: *const u32);
        #[link_name = "__rocm_oxide_atomic_scope_system"]
        fn mark_atomic_scope_system(ptr: *const u32);
    }

    pub trait Scope {
        const SCOPE: AtomicScope;

        unsafe fn mark_atomic_ptr<T>(ptr: *const T);
    }

    pub enum Workgroup {}
    pub enum Device {}
    pub enum System {}

    impl Scope for Workgroup {
        const SCOPE: AtomicScope = AtomicScope::Workgroup;

        #[inline(always)]
        unsafe fn mark_atomic_ptr<T>(ptr: *const T) {
            unsafe { mark_atomic_scope_workgroup(ptr.cast::<u32>()) };
        }
    }

    impl Scope for Device {
        const SCOPE: AtomicScope = AtomicScope::Device;

        #[inline(always)]
        unsafe fn mark_atomic_ptr<T>(ptr: *const T) {
            unsafe { mark_atomic_scope_device(ptr.cast::<u32>()) };
        }
    }

    impl Scope for System {
        const SCOPE: AtomicScope = AtomicScope::System;

        #[inline(always)]
        unsafe fn mark_atomic_ptr<T>(ptr: *const T) {
            unsafe { mark_atomic_scope_system(ptr.cast::<u32>()) };
        }
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
            unsafe { S::mark_atomic_ptr(self as *const Self as *const u32) };
            self.inner.load(ordering.as_core())
        }

        #[inline(always)]
        pub fn store(&self, value: u32, ordering: AtomicOrdering) {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const u32) };
            self.inner.store(value, ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: u32, ordering: AtomicOrdering) -> u32 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const u32) };
            self.inner.fetch_add(value, ordering.as_core())
        }
    }

    pub type WorkgroupAtomicU32 = AtomicU32Ref<Workgroup>;
    pub type DeviceAtomicU32 = AtomicU32Ref<Device>;
    pub type SystemAtomicU32 = AtomicU32Ref<System>;

    #[repr(transparent)]
    pub struct AtomicI32Ref<S: Scope> {
        inner: AtomicI32,
        _scope: PhantomData<S>,
    }

    impl<S: Scope> AtomicI32Ref<S> {
        #[inline(always)]
        pub unsafe fn from_ptr<'a>(ptr: *mut i32) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub unsafe fn from_const_ptr<'a>(ptr: *const i32) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub const fn scope() -> AtomicScope {
            S::SCOPE
        }

        #[inline(always)]
        pub fn load(&self, ordering: AtomicOrdering) -> i32 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const i32) };
            self.inner.load(ordering.as_core())
        }

        #[inline(always)]
        pub fn store(&self, value: i32, ordering: AtomicOrdering) {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const i32) };
            self.inner.store(value, ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: i32, ordering: AtomicOrdering) -> i32 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const i32) };
            self.inner.fetch_add(value, ordering.as_core())
        }
    }

    pub type WorkgroupAtomicI32 = AtomicI32Ref<Workgroup>;
    pub type DeviceAtomicI32 = AtomicI32Ref<Device>;
    pub type SystemAtomicI32 = AtomicI32Ref<System>;

    #[repr(transparent)]
    pub struct AtomicU64Ref<S: Scope> {
        inner: AtomicU64,
        _scope: PhantomData<S>,
    }

    impl<S: Scope> AtomicU64Ref<S> {
        #[inline(always)]
        pub unsafe fn from_ptr<'a>(ptr: *mut u64) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub unsafe fn from_const_ptr<'a>(ptr: *const u64) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub const fn scope() -> AtomicScope {
            S::SCOPE
        }

        #[inline(always)]
        pub fn load(&self, ordering: AtomicOrdering) -> u64 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const u64) };
            self.inner.load(ordering.as_core())
        }

        #[inline(always)]
        pub fn store(&self, value: u64, ordering: AtomicOrdering) {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const u64) };
            self.inner.store(value, ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: u64, ordering: AtomicOrdering) -> u64 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const u64) };
            self.inner.fetch_add(value, ordering.as_core())
        }
    }

    pub type WorkgroupAtomicU64 = AtomicU64Ref<Workgroup>;
    pub type DeviceAtomicU64 = AtomicU64Ref<Device>;
    pub type SystemAtomicU64 = AtomicU64Ref<System>;

    #[repr(transparent)]
    pub struct AtomicI64Ref<S: Scope> {
        inner: AtomicI64,
        _scope: PhantomData<S>,
    }

    impl<S: Scope> AtomicI64Ref<S> {
        #[inline(always)]
        pub unsafe fn from_ptr<'a>(ptr: *mut i64) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub unsafe fn from_const_ptr<'a>(ptr: *const i64) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub const fn scope() -> AtomicScope {
            S::SCOPE
        }

        #[inline(always)]
        pub fn load(&self, ordering: AtomicOrdering) -> i64 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const i64) };
            self.inner.load(ordering.as_core())
        }

        #[inline(always)]
        pub fn store(&self, value: i64, ordering: AtomicOrdering) {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const i64) };
            self.inner.store(value, ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: i64, ordering: AtomicOrdering) -> i64 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const i64) };
            self.inner.fetch_add(value, ordering.as_core())
        }
    }

    pub type WorkgroupAtomicI64 = AtomicI64Ref<Workgroup>;
    pub type DeviceAtomicI64 = AtomicI64Ref<Device>;
    pub type SystemAtomicI64 = AtomicI64Ref<System>;

    #[repr(transparent)]
    pub struct AtomicF32Ref<S: Scope> {
        inner: AtomicU32,
        _scope: PhantomData<S>,
    }

    impl<S: Scope> AtomicF32Ref<S> {
        #[inline(always)]
        pub unsafe fn from_ptr<'a>(ptr: *mut f32) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub unsafe fn from_const_ptr<'a>(ptr: *const f32) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub const fn scope() -> AtomicScope {
            S::SCOPE
        }

        #[inline(always)]
        pub fn load(&self, ordering: AtomicOrdering) -> f32 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const f32) };
            f32::from_bits(self.inner.load(ordering.as_core()))
        }

        #[inline(always)]
        pub fn store(&self, value: f32, ordering: AtomicOrdering) {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const f32) };
            self.inner.store(value.to_bits(), ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: f32, ordering: AtomicOrdering) -> f32 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const f32) };
            let mut current = self.inner.load(ordering.as_core());
            let failure_ordering = compare_exchange_failure_ordering(ordering);
            loop {
                let next = (f32::from_bits(current) + value).to_bits();
                unsafe { S::mark_atomic_ptr(self as *const Self as *const f32) };
                match self.inner.compare_exchange_weak(
                    current,
                    next,
                    ordering.as_core(),
                    failure_ordering,
                ) {
                    Ok(previous) => return f32::from_bits(previous),
                    Err(observed) => current = observed,
                }
            }
        }
    }

    pub type WorkgroupAtomicF32 = AtomicF32Ref<Workgroup>;
    pub type DeviceAtomicF32 = AtomicF32Ref<Device>;
    pub type SystemAtomicF32 = AtomicF32Ref<System>;

    #[repr(transparent)]
    pub struct AtomicF64Ref<S: Scope> {
        inner: AtomicU64,
        _scope: PhantomData<S>,
    }

    impl<S: Scope> AtomicF64Ref<S> {
        #[inline(always)]
        pub unsafe fn from_ptr<'a>(ptr: *mut f64) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub unsafe fn from_const_ptr<'a>(ptr: *const f64) -> &'a Self {
            unsafe { &*ptr.cast::<Self>() }
        }

        #[inline(always)]
        pub const fn scope() -> AtomicScope {
            S::SCOPE
        }

        #[inline(always)]
        pub fn load(&self, ordering: AtomicOrdering) -> f64 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const f64) };
            f64::from_bits(self.inner.load(ordering.as_core()))
        }

        #[inline(always)]
        pub fn store(&self, value: f64, ordering: AtomicOrdering) {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const f64) };
            self.inner.store(value.to_bits(), ordering.as_core());
        }

        #[inline(always)]
        pub fn fetch_add(&self, value: f64, ordering: AtomicOrdering) -> f64 {
            unsafe { S::mark_atomic_ptr(self as *const Self as *const f64) };
            let mut current = self.inner.load(ordering.as_core());
            let failure_ordering = compare_exchange_failure_ordering(ordering);
            loop {
                let next = (f64::from_bits(current) + value).to_bits();
                unsafe { S::mark_atomic_ptr(self as *const Self as *const f64) };
                match self.inner.compare_exchange_weak(
                    current,
                    next,
                    ordering.as_core(),
                    failure_ordering,
                ) {
                    Ok(previous) => return f64::from_bits(previous),
                    Err(observed) => current = observed,
                }
            }
        }
    }

    pub type WorkgroupAtomicF64 = AtomicF64Ref<Workgroup>;
    pub type DeviceAtomicF64 = AtomicF64Ref<Device>;
    pub type SystemAtomicF64 = AtomicF64Ref<System>;

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

    #[inline(always)]
    pub unsafe fn atomic_add_i32_scoped(
        ptr: *mut i32,
        value: i32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> i32 {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicI32::from_ptr(ptr) }.fetch_add(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicI32::from_ptr(ptr) }
                .fetch_add(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicI32::from_ptr(ptr) }
                .fetch_add(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_store_i32_scoped(
        ptr: *mut i32,
        value: i32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicI32::from_ptr(ptr) }.store(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicI32::from_ptr(ptr) }
                .store(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicI32::from_ptr(ptr) }
                .store(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_load_i32_scoped(
        ptr: *const i32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> i32 {
        match scope {
            AtomicScope::Workgroup => unsafe { WorkgroupAtomicI32::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::Device => unsafe { DeviceAtomicI32::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::System => unsafe { SystemAtomicI32::from_const_ptr(ptr) }
                .load(ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_add_u64_scoped(
        ptr: *mut u64,
        value: u64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> u64 {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicU64::from_ptr(ptr) }.fetch_add(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicU64::from_ptr(ptr) }
                .fetch_add(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicU64::from_ptr(ptr) }
                .fetch_add(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_store_u64_scoped(
        ptr: *mut u64,
        value: u64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicU64::from_ptr(ptr) }.store(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicU64::from_ptr(ptr) }
                .store(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicU64::from_ptr(ptr) }
                .store(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_load_u64_scoped(
        ptr: *const u64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> u64 {
        match scope {
            AtomicScope::Workgroup => unsafe { WorkgroupAtomicU64::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::Device => unsafe { DeviceAtomicU64::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::System => unsafe { SystemAtomicU64::from_const_ptr(ptr) }
                .load(ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_add_i64_scoped(
        ptr: *mut i64,
        value: i64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> i64 {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicI64::from_ptr(ptr) }.fetch_add(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicI64::from_ptr(ptr) }
                .fetch_add(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicI64::from_ptr(ptr) }
                .fetch_add(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_store_i64_scoped(
        ptr: *mut i64,
        value: i64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicI64::from_ptr(ptr) }.store(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicI64::from_ptr(ptr) }
                .store(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicI64::from_ptr(ptr) }
                .store(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_load_i64_scoped(
        ptr: *const i64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> i64 {
        match scope {
            AtomicScope::Workgroup => unsafe { WorkgroupAtomicI64::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::Device => unsafe { DeviceAtomicI64::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::System => unsafe { SystemAtomicI64::from_const_ptr(ptr) }
                .load(ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_add_f32_scoped(
        ptr: *mut f32,
        value: f32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> f32 {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicF32::from_ptr(ptr) }.fetch_add(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicF32::from_ptr(ptr) }
                .fetch_add(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicF32::from_ptr(ptr) }
                .fetch_add(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_store_f32_scoped(
        ptr: *mut f32,
        value: f32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicF32::from_ptr(ptr) }.store(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicF32::from_ptr(ptr) }
                .store(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicF32::from_ptr(ptr) }
                .store(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_load_f32_scoped(
        ptr: *const f32,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> f32 {
        match scope {
            AtomicScope::Workgroup => unsafe { WorkgroupAtomicF32::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::Device => unsafe { DeviceAtomicF32::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::System => unsafe { SystemAtomicF32::from_const_ptr(ptr) }
                .load(ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_add_f64_scoped(
        ptr: *mut f64,
        value: f64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> f64 {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicF64::from_ptr(ptr) }.fetch_add(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicF64::from_ptr(ptr) }
                .fetch_add(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicF64::from_ptr(ptr) }
                .fetch_add(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_store_f64_scoped(
        ptr: *mut f64,
        value: f64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) {
        match scope {
            AtomicScope::Workgroup => {
                unsafe { WorkgroupAtomicF64::from_ptr(ptr) }.store(value, ordering)
            }
            AtomicScope::Device => unsafe { DeviceAtomicF64::from_ptr(ptr) }
                .store(value, ordering),
            AtomicScope::System => unsafe { SystemAtomicF64::from_ptr(ptr) }
                .store(value, ordering),
        }
    }

    #[inline(always)]
    pub unsafe fn atomic_load_f64_scoped(
        ptr: *const f64,
        scope: AtomicScope,
        ordering: AtomicOrdering,
    ) -> f64 {
        match scope {
            AtomicScope::Workgroup => unsafe { WorkgroupAtomicF64::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::Device => unsafe { DeviceAtomicF64::from_const_ptr(ptr) }
                .load(ordering),
            AtomicScope::System => unsafe { SystemAtomicF64::from_const_ptr(ptr) }
                .load(ordering),
        }
    }
}

pub use atomic::{
    AtomicOrdering, AtomicScope, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicI32,
    DeviceAtomicI64, DeviceAtomicU32, DeviceAtomicU64, SystemAtomicF32, SystemAtomicF64,
    SystemAtomicI32, SystemAtomicI64, SystemAtomicU32, SystemAtomicU64, WorkgroupAtomicF32,
    WorkgroupAtomicF64, WorkgroupAtomicI32, WorkgroupAtomicI64, WorkgroupAtomicU32,
    WorkgroupAtomicU64,
};

pub mod cooperative {
    use super::{
        ballot, block_dim_x, block_dim_y, block_dim_z, block_idx_x, block_idx_y, block_idx_z,
        block_reduce_add_f32, block_reduce_add_i32, block_reduce_add_u32, block_reduce_and_u32,
        block_reduce_max_f32, block_reduce_max_i32, block_reduce_max_u32, block_reduce_min_f32,
        block_reduce_min_i32, block_reduce_min_u32, block_reduce_or_u32, block_reduce_xor_u32,
        block_scan_exclusive_add_f32, block_scan_exclusive_add_i32,
        block_scan_exclusive_add_u32, block_scan_inclusive_add_f32,
        block_scan_inclusive_add_i32, block_scan_inclusive_add_u32, inverse_ballot, lane_id,
        read_first_lane_u32, thread_idx_x, thread_idx_y, thread_idx_z, wave_barrier,
        wave_id_in_workgroup, wave_match_any_u32, wave_reduce_add_i32, wave_reduce_add_u32,
        wave_reduce_and_u32, wave_reduce_max_i32, wave_reduce_max_u32, wave_reduce_min_i32,
        wave_reduce_min_u32, wave_reduce_or_u32, wave_reduce_xor_u32, wave_shuffle_down_u32,
        wave_shuffle_f32, wave_shuffle_i32, wave_shuffle_u32, wave_shuffle_up_u32,
        wave_shuffle_xor_u32, wavefront_size, workgroup_barrier,
    };

    #[derive(Clone, Copy)]
    pub struct ThreadBlock;

    #[derive(Clone, Copy)]
    pub struct Wavefront;

    #[derive(Clone, Copy)]
    pub struct StaticTile<const N: u32>;

    #[inline(always)]
    pub const fn this_thread_block() -> ThreadBlock {
        ThreadBlock
    }

    #[inline(always)]
    pub const fn this_wavefront() -> Wavefront {
        Wavefront
    }

    #[inline(always)]
    pub const fn tiled_partition<const N: u32>(_group: ThreadBlock) -> StaticTile<N> {
        StaticTile
    }

    macro_rules! block_reduce_method {
        ($method:ident, $root:ident, $ty:ty) => {
            #[inline(always)]
            pub unsafe fn $method(self, scratch: *mut $ty, value: $ty) -> $ty {
                unsafe { super::$root(scratch, value) }
            }
        };
    }

    macro_rules! block_scan_method {
        ($method:ident, $root:ident, $ty:ty) => {
            #[inline(always)]
            pub unsafe fn $method(self, scratch: *mut $ty, value: $ty) -> $ty {
                unsafe { super::$root(scratch, value) }
            }
        };
    }

    impl ThreadBlock {
        #[inline(always)]
        pub fn size(self) -> u32 {
            block_dim_x() * block_dim_y() * block_dim_z()
        }

        #[inline(always)]
        pub fn thread_rank(self) -> u32 {
            thread_idx_x()
                + thread_idx_y() * block_dim_x()
                + thread_idx_z() * block_dim_x() * block_dim_y()
        }

        #[inline(always)]
        pub fn group_index_x(self) -> u32 {
            block_idx_x()
        }

        #[inline(always)]
        pub fn group_index_y(self) -> u32 {
            block_idx_y()
        }

        #[inline(always)]
        pub fn group_index_z(self) -> u32 {
            block_idx_z()
        }

        #[inline(always)]
        pub fn thread_index_x(self) -> u32 {
            thread_idx_x()
        }

        #[inline(always)]
        pub fn thread_index_y(self) -> u32 {
            thread_idx_y()
        }

        #[inline(always)]
        pub fn thread_index_z(self) -> u32 {
            thread_idx_z()
        }

        #[inline(always)]
        pub fn sync(self) {
            workgroup_barrier()
        }

        #[inline(always)]
        pub unsafe fn reduce_add_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_reduce_add_u32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn reduce_add_i32(self, scratch: *mut i32, value: i32) -> i32 {
            unsafe { block_reduce_add_i32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn reduce_add_f32(self, scratch: *mut f32, value: f32) -> f32 {
            unsafe { block_reduce_add_f32(scratch, value) }
        }

        block_reduce_method!(reduce_add_u64, block_reduce_add_u64, u64);
        block_reduce_method!(reduce_add_i64, block_reduce_add_i64, i64);
        block_reduce_method!(reduce_add_f64, block_reduce_add_f64, f64);

        #[inline(always)]
        pub unsafe fn reduce_min_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_reduce_min_u32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn reduce_min_i32(self, scratch: *mut i32, value: i32) -> i32 {
            unsafe { block_reduce_min_i32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn reduce_min_f32(self, scratch: *mut f32, value: f32) -> f32 {
            unsafe { block_reduce_min_f32(scratch, value) }
        }

        block_reduce_method!(reduce_min_u64, block_reduce_min_u64, u64);
        block_reduce_method!(reduce_min_i64, block_reduce_min_i64, i64);
        block_reduce_method!(reduce_min_f64, block_reduce_min_f64, f64);

        #[inline(always)]
        pub unsafe fn reduce_max_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_reduce_max_u32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn reduce_max_i32(self, scratch: *mut i32, value: i32) -> i32 {
            unsafe { block_reduce_max_i32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn reduce_max_f32(self, scratch: *mut f32, value: f32) -> f32 {
            unsafe { block_reduce_max_f32(scratch, value) }
        }

        block_reduce_method!(reduce_max_u64, block_reduce_max_u64, u64);
        block_reduce_method!(reduce_max_i64, block_reduce_max_i64, i64);
        block_reduce_method!(reduce_max_f64, block_reduce_max_f64, f64);

        #[inline(always)]
        pub unsafe fn reduce_and_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_reduce_and_u32(scratch, value) }
        }

        block_reduce_method!(reduce_and_i32, block_reduce_and_i32, i32);
        block_reduce_method!(reduce_and_u64, block_reduce_and_u64, u64);
        block_reduce_method!(reduce_and_i64, block_reduce_and_i64, i64);

        #[inline(always)]
        pub unsafe fn reduce_or_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_reduce_or_u32(scratch, value) }
        }

        block_reduce_method!(reduce_or_i32, block_reduce_or_i32, i32);
        block_reduce_method!(reduce_or_u64, block_reduce_or_u64, u64);
        block_reduce_method!(reduce_or_i64, block_reduce_or_i64, i64);

        #[inline(always)]
        pub unsafe fn reduce_xor_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_reduce_xor_u32(scratch, value) }
        }

        block_reduce_method!(reduce_xor_i32, block_reduce_xor_i32, i32);
        block_reduce_method!(reduce_xor_u64, block_reduce_xor_u64, u64);
        block_reduce_method!(reduce_xor_i64, block_reduce_xor_i64, i64);

        #[inline(always)]
        pub unsafe fn scan_inclusive_add_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_scan_inclusive_add_u32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn scan_inclusive_add_i32(self, scratch: *mut i32, value: i32) -> i32 {
            unsafe { block_scan_inclusive_add_i32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn scan_inclusive_add_f32(self, scratch: *mut f32, value: f32) -> f32 {
            unsafe { block_scan_inclusive_add_f32(scratch, value) }
        }

        block_scan_method!(scan_inclusive_add_u64, block_scan_inclusive_add_u64, u64);
        block_scan_method!(scan_inclusive_add_i64, block_scan_inclusive_add_i64, i64);
        block_scan_method!(scan_inclusive_add_f64, block_scan_inclusive_add_f64, f64);
        block_scan_method!(scan_inclusive_min_u32, block_scan_inclusive_min_u32, u32);
        block_scan_method!(scan_inclusive_min_i32, block_scan_inclusive_min_i32, i32);
        block_scan_method!(scan_inclusive_min_f32, block_scan_inclusive_min_f32, f32);
        block_scan_method!(scan_inclusive_min_u64, block_scan_inclusive_min_u64, u64);
        block_scan_method!(scan_inclusive_min_i64, block_scan_inclusive_min_i64, i64);
        block_scan_method!(scan_inclusive_min_f64, block_scan_inclusive_min_f64, f64);
        block_scan_method!(scan_inclusive_max_u32, block_scan_inclusive_max_u32, u32);
        block_scan_method!(scan_inclusive_max_i32, block_scan_inclusive_max_i32, i32);
        block_scan_method!(scan_inclusive_max_f32, block_scan_inclusive_max_f32, f32);
        block_scan_method!(scan_inclusive_max_u64, block_scan_inclusive_max_u64, u64);
        block_scan_method!(scan_inclusive_max_i64, block_scan_inclusive_max_i64, i64);
        block_scan_method!(scan_inclusive_max_f64, block_scan_inclusive_max_f64, f64);
        block_scan_method!(scan_inclusive_and_u32, block_scan_inclusive_and_u32, u32);
        block_scan_method!(scan_inclusive_and_i32, block_scan_inclusive_and_i32, i32);
        block_scan_method!(scan_inclusive_and_u64, block_scan_inclusive_and_u64, u64);
        block_scan_method!(scan_inclusive_and_i64, block_scan_inclusive_and_i64, i64);
        block_scan_method!(scan_inclusive_or_u32, block_scan_inclusive_or_u32, u32);
        block_scan_method!(scan_inclusive_or_i32, block_scan_inclusive_or_i32, i32);
        block_scan_method!(scan_inclusive_or_u64, block_scan_inclusive_or_u64, u64);
        block_scan_method!(scan_inclusive_or_i64, block_scan_inclusive_or_i64, i64);
        block_scan_method!(scan_inclusive_xor_u32, block_scan_inclusive_xor_u32, u32);
        block_scan_method!(scan_inclusive_xor_i32, block_scan_inclusive_xor_i32, i32);
        block_scan_method!(scan_inclusive_xor_u64, block_scan_inclusive_xor_u64, u64);
        block_scan_method!(scan_inclusive_xor_i64, block_scan_inclusive_xor_i64, i64);

        #[inline(always)]
        pub unsafe fn scan_exclusive_add_u32(self, scratch: *mut u32, value: u32) -> u32 {
            unsafe { block_scan_exclusive_add_u32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn scan_exclusive_add_i32(self, scratch: *mut i32, value: i32) -> i32 {
            unsafe { block_scan_exclusive_add_i32(scratch, value) }
        }

        #[inline(always)]
        pub unsafe fn scan_exclusive_add_f32(self, scratch: *mut f32, value: f32) -> f32 {
            unsafe { block_scan_exclusive_add_f32(scratch, value) }
        }

        block_scan_method!(scan_exclusive_add_u64, block_scan_exclusive_add_u64, u64);
        block_scan_method!(scan_exclusive_add_i64, block_scan_exclusive_add_i64, i64);
        block_scan_method!(scan_exclusive_add_f64, block_scan_exclusive_add_f64, f64);
        block_scan_method!(scan_exclusive_min_u32, block_scan_exclusive_min_u32, u32);
        block_scan_method!(scan_exclusive_min_i32, block_scan_exclusive_min_i32, i32);
        block_scan_method!(scan_exclusive_min_f32, block_scan_exclusive_min_f32, f32);
        block_scan_method!(scan_exclusive_min_u64, block_scan_exclusive_min_u64, u64);
        block_scan_method!(scan_exclusive_min_i64, block_scan_exclusive_min_i64, i64);
        block_scan_method!(scan_exclusive_min_f64, block_scan_exclusive_min_f64, f64);
        block_scan_method!(scan_exclusive_max_u32, block_scan_exclusive_max_u32, u32);
        block_scan_method!(scan_exclusive_max_i32, block_scan_exclusive_max_i32, i32);
        block_scan_method!(scan_exclusive_max_f32, block_scan_exclusive_max_f32, f32);
        block_scan_method!(scan_exclusive_max_u64, block_scan_exclusive_max_u64, u64);
        block_scan_method!(scan_exclusive_max_i64, block_scan_exclusive_max_i64, i64);
        block_scan_method!(scan_exclusive_max_f64, block_scan_exclusive_max_f64, f64);
        block_scan_method!(scan_exclusive_and_u32, block_scan_exclusive_and_u32, u32);
        block_scan_method!(scan_exclusive_and_i32, block_scan_exclusive_and_i32, i32);
        block_scan_method!(scan_exclusive_and_u64, block_scan_exclusive_and_u64, u64);
        block_scan_method!(scan_exclusive_and_i64, block_scan_exclusive_and_i64, i64);
        block_scan_method!(scan_exclusive_or_u32, block_scan_exclusive_or_u32, u32);
        block_scan_method!(scan_exclusive_or_i32, block_scan_exclusive_or_i32, i32);
        block_scan_method!(scan_exclusive_or_u64, block_scan_exclusive_or_u64, u64);
        block_scan_method!(scan_exclusive_or_i64, block_scan_exclusive_or_i64, i64);
        block_scan_method!(scan_exclusive_xor_u32, block_scan_exclusive_xor_u32, u32);
        block_scan_method!(scan_exclusive_xor_i32, block_scan_exclusive_xor_i32, i32);
        block_scan_method!(scan_exclusive_xor_u64, block_scan_exclusive_xor_u64, u64);
        block_scan_method!(scan_exclusive_xor_i64, block_scan_exclusive_xor_i64, i64);
    }

    impl Wavefront {
        #[inline(always)]
        pub fn size(self) -> u32 {
            wavefront_size()
        }

        #[inline(always)]
        pub fn thread_rank(self) -> u32 {
            lane_id()
        }

        #[inline(always)]
        pub fn meta_group_rank(self) -> u32 {
            wave_id_in_workgroup()
        }

        #[inline(always)]
        pub fn sync(self) {
            wave_barrier()
        }

        #[inline(always)]
        pub fn active_mask(self) -> u64 {
            ballot(true)
        }

        #[inline(always)]
        pub fn ballot(self, predicate: bool) -> u64 {
            ballot(predicate)
        }

        #[inline(always)]
        pub fn any(self, predicate: bool) -> bool {
            self.ballot(predicate) != 0
        }

        #[inline(always)]
        pub fn all(self, predicate: bool) -> bool {
            self.ballot(predicate) == self.active_mask()
        }

        #[inline(always)]
        pub fn none(self, predicate: bool) -> bool {
            self.ballot(predicate) == 0
        }

        #[inline(always)]
        pub fn elected(self) -> bool {
            inverse_ballot(self.active_mask())
        }

        #[inline(always)]
        pub fn read_first_u32(self, value: u32) -> u32 {
            read_first_lane_u32(value)
        }

        #[inline(always)]
        pub fn shuffle_u32(self, value: u32, lane: u32) -> u32 {
            wave_shuffle_u32(value, lane)
        }

        #[inline(always)]
        pub fn shuffle_i32(self, value: i32, lane: u32) -> i32 {
            wave_shuffle_i32(value, lane)
        }

        #[inline(always)]
        pub fn shuffle_f32(self, value: f32, lane: u32) -> f32 {
            wave_shuffle_f32(value, lane)
        }

        #[inline(always)]
        pub fn shuffle_up_u32(self, value: u32, delta: u32) -> u32 {
            wave_shuffle_up_u32(value, delta)
        }

        #[inline(always)]
        pub fn shuffle_down_u32(self, value: u32, delta: u32) -> u32 {
            wave_shuffle_down_u32(value, delta)
        }

        #[inline(always)]
        pub fn shuffle_xor_u32(self, value: u32, mask: u32) -> u32 {
            wave_shuffle_xor_u32(value, mask)
        }

        #[inline(always)]
        pub fn reduce_add_u32(self, value: u32) -> u32 {
            wave_reduce_add_u32(value)
        }

        #[inline(always)]
        pub fn reduce_add_i32(self, value: i32) -> i32 {
            wave_reduce_add_i32(value)
        }

        #[inline(always)]
        pub fn reduce_min_u32(self, value: u32) -> u32 {
            wave_reduce_min_u32(value)
        }

        #[inline(always)]
        pub fn reduce_min_i32(self, value: i32) -> i32 {
            wave_reduce_min_i32(value)
        }

        #[inline(always)]
        pub fn reduce_max_u32(self, value: u32) -> u32 {
            wave_reduce_max_u32(value)
        }

        #[inline(always)]
        pub fn reduce_max_i32(self, value: i32) -> i32 {
            wave_reduce_max_i32(value)
        }

        #[inline(always)]
        pub fn reduce_and_u32(self, value: u32) -> u32 {
            wave_reduce_and_u32(value)
        }

        #[inline(always)]
        pub fn reduce_or_u32(self, value: u32) -> u32 {
            wave_reduce_or_u32(value)
        }

        #[inline(always)]
        pub fn reduce_xor_u32(self, value: u32) -> u32 {
            wave_reduce_xor_u32(value)
        }

        #[inline(always)]
        pub fn match_any_u32(self, value: u32) -> u64 {
            wave_match_any_u32(value)
        }
    }

    impl<const N: u32> StaticTile<N> {
        #[inline(always)]
        pub const fn size(self) -> u32 {
            N
        }

        #[inline(always)]
        pub fn thread_rank(self) -> u32 {
            let size = self.size();
            if size == 0 {
                0
            } else {
                this_thread_block().thread_rank() % size
            }
        }

        #[inline(always)]
        pub fn meta_group_rank(self) -> u32 {
            let size = self.size();
            if size == 0 {
                0
            } else {
                this_thread_block().thread_rank() / size
            }
        }

        #[inline(always)]
        pub fn meta_group_size(self) -> u32 {
            let size = self.size();
            if size == 0 {
                0
            } else {
                this_thread_block().size().div_ceil(size)
            }
        }

        #[inline(always)]
        pub fn sync(self) {
            if self.size() <= wavefront_size() {
                wave_barrier();
            } else {
                workgroup_barrier();
            }
        }
    }
}

pub use cooperative::{
    StaticTile, ThreadBlock, Wavefront, this_thread_block, this_wavefront, tiled_partition,
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
    let xy_thread = thread_idx_x() + thread_idx_y() * block_dim_x();
    let z_offset = thread_idx_z() * block_dim_x() * block_dim_y();
    let linear_thread = xy_thread + z_offset;
    linear_thread / wavefront_size()
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
pub fn inverse_ballot(mask: u64) -> bool {
    amdgpu::inverse_ballot(mask)
}

#[inline(always)]
pub fn read_first_lane_u32(value: u32) -> u32 {
    amdgpu::readfirstlane_u32(value)
}

#[inline(always)]
pub fn read_first_lane_u64(value: u64) -> u64 {
    amdgpu::readfirstlane_u64(value)
}

#[inline(always)]
pub fn wave_shuffle_u32(value: u32, lane: u32) -> u32 {
    unsafe { amdgpu::ds_bpermute(lane.wrapping_mul(4), value) }
}

#[inline(always)]
pub fn wave_shuffle_i32(value: i32, lane: u32) -> i32 {
    wave_shuffle_u32(value as u32, lane) as i32
}

#[inline(always)]
pub fn wave_shuffle_f32(value: f32, lane: u32) -> f32 {
    f32::from_bits(wave_shuffle_u32(value.to_bits(), lane))
}

#[inline(always)]
pub fn wave_shuffle_up_u32(value: u32, delta: u32) -> u32 {
    let lane = lane_id();
    let target = if lane >= delta {
        lane - delta
    } else {
        lane
    };
    wave_shuffle_u32(value, target)
}

#[inline(always)]
pub fn wave_shuffle_down_u32(value: u32, delta: u32) -> u32 {
    let lane = lane_id();
    let next = lane.wrapping_add(delta);
    let target = if next < wavefront_size() {
        next
    } else {
        lane
    };
    wave_shuffle_u32(value, target)
}

#[inline(always)]
pub fn wave_shuffle_xor_u32(value: u32, mask: u32) -> u32 {
    wave_shuffle_u32(value, lane_id() ^ mask)
}

#[inline(always)]
pub fn wave_reduce_add_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_add::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_add_i32(value: i32) -> i32 {
    amdgpu::wave_reduce_add::<0>(value as u32) as i32
}

#[inline(always)]
pub fn wave_reduce_min_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_umin::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_min_i32(value: i32) -> i32 {
    amdgpu::wave_reduce_min::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_max_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_umax::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_max_i32(value: i32) -> i32 {
    amdgpu::wave_reduce_max::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_and_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_and::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_or_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_or::<0>(value)
}

#[inline(always)]
pub fn wave_reduce_xor_u32(value: u32) -> u32 {
    amdgpu::wave_reduce_xor::<0>(value)
}

#[inline(always)]
pub fn wave_match_any_u32(value: u32) -> u64 {
    let mut mask = 0u64;
    let mut lane = 0u32;
    let size = wavefront_size();
    while lane < size {
        if wave_shuffle_u32(value, lane) == value {
            mask |= 1u64 << lane;
        }
        lane = lane.wrapping_add(1);
    }
    mask
}

#[inline(always)]
pub unsafe fn block_reduce_add_u32(scratch: *mut u32, value: u32) -> u32 {
    let rank = this_thread_block().thread_rank();
    let size = this_thread_block().size();
    unsafe { scratch.add(rank as usize).write(value) };
    workgroup_barrier();

    let mut stride = 1u32;
    while stride < size {
        stride <<= 1;
    }
    stride >>= 1;

    while stride > 0 {
        if rank < stride {
            let other = rank + stride;
            if other < size {
                let slot = unsafe { scratch.add(rank as usize) };
                let rhs = unsafe { scratch.add(other as usize).read() };
                unsafe { slot.write(slot.read().wrapping_add(rhs)) };
            }
        }
        workgroup_barrier();
        stride >>= 1;
    }

    let result = unsafe { scratch.read() };
    workgroup_barrier();
    result
}

#[inline(always)]
pub unsafe fn block_reduce_add_i32(scratch: *mut i32, value: i32) -> i32 {
    let rank = this_thread_block().thread_rank();
    let size = this_thread_block().size();
    unsafe { scratch.add(rank as usize).write(value) };
    workgroup_barrier();

    let mut stride = 1u32;
    while stride < size {
        stride <<= 1;
    }
    stride >>= 1;

    while stride > 0 {
        if rank < stride {
            let other = rank + stride;
            if other < size {
                let slot = unsafe { scratch.add(rank as usize) };
                let rhs = unsafe { scratch.add(other as usize).read() };
                unsafe { slot.write(slot.read().wrapping_add(rhs)) };
            }
        }
        workgroup_barrier();
        stride >>= 1;
    }

    let result = unsafe { scratch.read() };
    workgroup_barrier();
    result
}

#[inline(always)]
pub unsafe fn block_reduce_add_f32(scratch: *mut f32, value: f32) -> f32 {
    let rank = this_thread_block().thread_rank();
    let size = this_thread_block().size();
    unsafe { scratch.add(rank as usize).write(value) };
    workgroup_barrier();

    let mut stride = 1u32;
    while stride < size {
        stride <<= 1;
    }
    stride >>= 1;

    while stride > 0 {
        if rank < stride {
            let other = rank + stride;
            if other < size {
                let slot = unsafe { scratch.add(rank as usize) };
                let rhs = unsafe { scratch.add(other as usize).read() };
                unsafe { slot.write(slot.read() + rhs) };
            }
        }
        workgroup_barrier();
        stride >>= 1;
    }

    let result = unsafe { scratch.read() };
    workgroup_barrier();
    result
}

macro_rules! define_block_reduce {
    ($name:ident, $ty:ty, $lhs:ident, $rhs:ident, $combine:expr) => {
        #[inline(always)]
        pub unsafe fn $name(scratch: *mut $ty, value: $ty) -> $ty {
            let rank = this_thread_block().thread_rank();
            let size = this_thread_block().size();
            unsafe { scratch.add(rank as usize).write(value) };
            workgroup_barrier();

            let mut stride = 1u32;
            while stride < size {
                stride <<= 1;
            }
            stride >>= 1;

            while stride > 0 {
                if rank < stride {
                    let other = rank + stride;
                    if other < size {
                        let slot = unsafe { scratch.add(rank as usize) };
                        let $lhs = unsafe { slot.read() };
                        let $rhs = unsafe { scratch.add(other as usize).read() };
                        unsafe { slot.write($combine) };
                    }
                }
                workgroup_barrier();
                stride >>= 1;
            }

            let result = unsafe { scratch.read() };
            workgroup_barrier();
            result
        }
    };
}

define_block_reduce!(
    block_reduce_min_u32,
    u32,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_min_i32,
    i32,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_min_f32,
    f32,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_reduce!(block_reduce_add_u64, u64, lhs, rhs, lhs.wrapping_add(rhs));
define_block_reduce!(block_reduce_add_i64, i64, lhs, rhs, lhs.wrapping_add(rhs));
define_block_reduce!(block_reduce_add_f64, f64, lhs, rhs, lhs + rhs);
define_block_reduce!(
    block_reduce_min_u64,
    u64,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_min_i64,
    i64,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_min_f64,
    f64,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_max_u32,
    u32,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_max_i32,
    i32,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_max_f32,
    f32,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_max_u64,
    u64,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_max_i64,
    i64,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_reduce!(
    block_reduce_max_f64,
    f64,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_reduce!(block_reduce_and_u32, u32, lhs, rhs, lhs & rhs);
define_block_reduce!(block_reduce_and_i32, i32, lhs, rhs, lhs & rhs);
define_block_reduce!(block_reduce_and_u64, u64, lhs, rhs, lhs & rhs);
define_block_reduce!(block_reduce_and_i64, i64, lhs, rhs, lhs & rhs);
define_block_reduce!(block_reduce_or_u32, u32, lhs, rhs, lhs | rhs);
define_block_reduce!(block_reduce_or_i32, i32, lhs, rhs, lhs | rhs);
define_block_reduce!(block_reduce_or_u64, u64, lhs, rhs, lhs | rhs);
define_block_reduce!(block_reduce_or_i64, i64, lhs, rhs, lhs | rhs);
define_block_reduce!(block_reduce_xor_u32, u32, lhs, rhs, lhs ^ rhs);
define_block_reduce!(block_reduce_xor_i32, i32, lhs, rhs, lhs ^ rhs);
define_block_reduce!(block_reduce_xor_u64, u64, lhs, rhs, lhs ^ rhs);
define_block_reduce!(block_reduce_xor_i64, i64, lhs, rhs, lhs ^ rhs);

#[inline(always)]
pub unsafe fn block_scan_inclusive_add_u32(scratch: *mut u32, value: u32) -> u32 {
    let rank = this_thread_block().thread_rank();
    let size = this_thread_block().size();
    unsafe { scratch.add(rank as usize).write(value) };
    workgroup_barrier();

    let mut offset = 1u32;
    while offset < size {
        let addend = if rank >= offset {
            unsafe { scratch.add((rank - offset) as usize).read() }
        } else {
            0
        };
        workgroup_barrier();
        if rank >= offset {
            let slot = unsafe { scratch.add(rank as usize) };
            unsafe { slot.write(slot.read().wrapping_add(addend)) };
        }
        workgroup_barrier();
        offset <<= 1;
    }

    let result = unsafe { scratch.add(rank as usize).read() };
    workgroup_barrier();
    result
}

#[inline(always)]
pub unsafe fn block_scan_inclusive_add_i32(scratch: *mut i32, value: i32) -> i32 {
    let rank = this_thread_block().thread_rank();
    let size = this_thread_block().size();
    unsafe { scratch.add(rank as usize).write(value) };
    workgroup_barrier();

    let mut offset = 1u32;
    while offset < size {
        let addend = if rank >= offset {
            unsafe { scratch.add((rank - offset) as usize).read() }
        } else {
            0
        };
        workgroup_barrier();
        if rank >= offset {
            let slot = unsafe { scratch.add(rank as usize) };
            unsafe { slot.write(slot.read().wrapping_add(addend)) };
        }
        workgroup_barrier();
        offset <<= 1;
    }

    let result = unsafe { scratch.add(rank as usize).read() };
    workgroup_barrier();
    result
}

#[inline(always)]
pub unsafe fn block_scan_inclusive_add_f32(scratch: *mut f32, value: f32) -> f32 {
    let rank = this_thread_block().thread_rank();
    let size = this_thread_block().size();
    unsafe { scratch.add(rank as usize).write(value) };
    workgroup_barrier();

    let mut offset = 1u32;
    while offset < size {
        let addend = if rank >= offset {
            unsafe { scratch.add((rank - offset) as usize).read() }
        } else {
            0.0
        };
        workgroup_barrier();
        if rank >= offset {
            let slot = unsafe { scratch.add(rank as usize) };
            unsafe { slot.write(slot.read() + addend) };
        }
        workgroup_barrier();
        offset <<= 1;
    }

    let result = unsafe { scratch.add(rank as usize).read() };
    workgroup_barrier();
    result
}

#[inline(always)]
pub unsafe fn block_scan_exclusive_add_u32(scratch: *mut u32, value: u32) -> u32 {
    unsafe { block_scan_inclusive_add_u32(scratch, value).wrapping_sub(value) }
}

#[inline(always)]
pub unsafe fn block_scan_exclusive_add_i32(scratch: *mut i32, value: i32) -> i32 {
    unsafe { block_scan_inclusive_add_i32(scratch, value).wrapping_sub(value) }
}

#[inline(always)]
pub unsafe fn block_scan_exclusive_add_f32(scratch: *mut f32, value: f32) -> f32 {
    unsafe { block_scan_inclusive_add_f32(scratch, value) - value }
}

macro_rules! define_block_scan {
    (
        $inclusive:ident,
        $exclusive:ident,
        $ty:ty,
        $identity:expr,
        $lhs:ident,
        $rhs:ident,
        $combine:expr
    ) => {
        #[inline(always)]
        pub unsafe fn $inclusive(scratch: *mut $ty, value: $ty) -> $ty {
            let rank = this_thread_block().thread_rank();
            let size = this_thread_block().size();
            unsafe { scratch.add(rank as usize).write(value) };
            workgroup_barrier();

            let mut offset = 1u32;
            while offset < size {
                let $rhs = if rank >= offset {
                    unsafe { scratch.add((rank - offset) as usize).read() }
                } else {
                    $identity
                };
                workgroup_barrier();
                if rank >= offset {
                    let slot = unsafe { scratch.add(rank as usize) };
                    let $lhs = unsafe { slot.read() };
                    unsafe { slot.write($combine) };
                }
                workgroup_barrier();
                offset <<= 1;
            }

            let result = unsafe { scratch.add(rank as usize).read() };
            workgroup_barrier();
            result
        }

        #[inline(always)]
        pub unsafe fn $exclusive(scratch: *mut $ty, value: $ty) -> $ty {
            let _ = unsafe { $inclusive(scratch, value) };
            let rank = this_thread_block().thread_rank();
            let result = if rank == 0 {
                $identity
            } else {
                unsafe { scratch.add((rank - 1) as usize).read() }
            };
            workgroup_barrier();
            result
        }
    };
}

define_block_scan!(
    block_scan_inclusive_add_u64,
    block_scan_exclusive_add_u64,
    u64,
    0,
    lhs,
    rhs,
    lhs.wrapping_add(rhs)
);
define_block_scan!(
    block_scan_inclusive_add_i64,
    block_scan_exclusive_add_i64,
    i64,
    0,
    lhs,
    rhs,
    lhs.wrapping_add(rhs)
);
define_block_scan!(
    block_scan_inclusive_add_f64,
    block_scan_exclusive_add_f64,
    f64,
    0.0,
    lhs,
    rhs,
    lhs + rhs
);
define_block_scan!(
    block_scan_inclusive_min_u32,
    block_scan_exclusive_min_u32,
    u32,
    u32::MAX,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_min_i32,
    block_scan_exclusive_min_i32,
    i32,
    i32::MAX,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_min_f32,
    block_scan_exclusive_min_f32,
    f32,
    f32::INFINITY,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_min_u64,
    block_scan_exclusive_min_u64,
    u64,
    u64::MAX,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_min_i64,
    block_scan_exclusive_min_i64,
    i64,
    i64::MAX,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_min_f64,
    block_scan_exclusive_min_f64,
    f64,
    f64::INFINITY,
    lhs,
    rhs,
    if lhs < rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_max_u32,
    block_scan_exclusive_max_u32,
    u32,
    u32::MIN,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_max_i32,
    block_scan_exclusive_max_i32,
    i32,
    i32::MIN,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_max_f32,
    block_scan_exclusive_max_f32,
    f32,
    f32::NEG_INFINITY,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_max_u64,
    block_scan_exclusive_max_u64,
    u64,
    u64::MIN,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_max_i64,
    block_scan_exclusive_max_i64,
    i64,
    i64::MIN,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_max_f64,
    block_scan_exclusive_max_f64,
    f64,
    f64::NEG_INFINITY,
    lhs,
    rhs,
    if lhs > rhs { lhs } else { rhs }
);
define_block_scan!(
    block_scan_inclusive_and_u32,
    block_scan_exclusive_and_u32,
    u32,
    u32::MAX,
    lhs,
    rhs,
    lhs & rhs
);
define_block_scan!(
    block_scan_inclusive_and_i32,
    block_scan_exclusive_and_i32,
    i32,
    -1,
    lhs,
    rhs,
    lhs & rhs
);
define_block_scan!(
    block_scan_inclusive_and_u64,
    block_scan_exclusive_and_u64,
    u64,
    u64::MAX,
    lhs,
    rhs,
    lhs & rhs
);
define_block_scan!(
    block_scan_inclusive_and_i64,
    block_scan_exclusive_and_i64,
    i64,
    -1,
    lhs,
    rhs,
    lhs & rhs
);
define_block_scan!(
    block_scan_inclusive_or_u32,
    block_scan_exclusive_or_u32,
    u32,
    0,
    lhs,
    rhs,
    lhs | rhs
);
define_block_scan!(
    block_scan_inclusive_or_i32,
    block_scan_exclusive_or_i32,
    i32,
    0,
    lhs,
    rhs,
    lhs | rhs
);
define_block_scan!(
    block_scan_inclusive_or_u64,
    block_scan_exclusive_or_u64,
    u64,
    0,
    lhs,
    rhs,
    lhs | rhs
);
define_block_scan!(
    block_scan_inclusive_or_i64,
    block_scan_exclusive_or_i64,
    i64,
    0,
    lhs,
    rhs,
    lhs | rhs
);
define_block_scan!(
    block_scan_inclusive_xor_u32,
    block_scan_exclusive_xor_u32,
    u32,
    0,
    lhs,
    rhs,
    lhs ^ rhs
);
define_block_scan!(
    block_scan_inclusive_xor_i32,
    block_scan_exclusive_xor_i32,
    i32,
    0,
    lhs,
    rhs,
    lhs ^ rhs
);
define_block_scan!(
    block_scan_inclusive_xor_u64,
    block_scan_exclusive_xor_u64,
    u64,
    0,
    lhs,
    rhs,
    lhs ^ rhs
);
define_block_scan!(
    block_scan_inclusive_xor_i64,
    block_scan_exclusive_xor_i64,
    i64,
    0,
    lhs,
    rhs,
    lhs ^ rhs
);

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

#[derive(Clone, Copy)]
pub struct ThreadIndex {
    index: usize,
}

impl ThreadIndex {
    #[inline(always)]
    pub fn global_x() -> Self {
        Self {
            index: global_id_x(),
        }
    }

    #[inline(always)]
    pub const fn get(self) -> usize {
        self.index
    }
}

#[inline(always)]
pub fn thread_index_x_witness() -> ThreadIndex {
    ThreadIndex::global_x()
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

    #[inline(always)]
    pub fn write_for_thread(self, index: ThreadIndex, value: T) -> bool {
        let index = index.get();
        if index < self.len {
            unsafe { self.write_unchecked(index, value) };
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Copy)]
pub struct DisjointSliceMut<T> {
    slice: DeviceSliceMut<T>,
}

impl<T> DisjointSliceMut<T> {
    #[inline(always)]
    pub const unsafe fn new_unchecked(slice: DeviceSliceMut<T>) -> Self {
        Self { slice }
    }

    #[inline(always)]
    pub const fn len(self) -> usize {
        self.slice.len()
    }

    #[inline(always)]
    pub fn write_for_thread(self, index: ThreadIndex, value: T) -> bool {
        self.slice.write_for_thread(index, value)
    }

    #[inline(always)]
    pub unsafe fn write_unchecked(self, index: usize, value: T) {
        unsafe { self.slice.write_unchecked(index, value) };
    }

    #[inline(always)]
    pub const fn into_slice(self) -> DeviceSliceMut<T> {
        self.slice
    }
}

#[derive(Clone, Copy)]
pub struct WorkgroupBarrierToken {
    _private: (),
}

impl WorkgroupBarrierToken {
    #[inline(always)]
    pub fn arrive_and_wait(self) -> Self {
        workgroup_barrier();
        self
    }
}

#[inline(always)]
pub fn workgroup_barrier_token() -> WorkgroupBarrierToken {
    WorkgroupBarrierToken { _private: () }
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
