#![no_std]
#![feature(core_intrinsics)]
#![feature(gpu_intrinsics)]
#![feature(gpu_launch_sized_workgroup_mem)]
#![feature(stdarch_amdgpu)]

use core::arch::amdgpu;
use core::intrinsics::gpu::{amdgpu_dispatch_ptr, gpu_launch_sized_workgroup_mem};
use core::{marker::PhantomData, ptr};
use core::sync::atomic::{AtomicU32, Ordering};

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
    unsafe { (*(ptr.cast::<AtomicU32>())).fetch_add(value, Ordering::Relaxed) }
}

#[inline(always)]
pub unsafe fn atomic_store_u32(ptr: *mut u32, value: u32, ordering: Ordering) {
    unsafe {
        (*(ptr.cast::<AtomicU32>())).store(value, ordering);
    }
}

#[inline(always)]
pub unsafe fn atomic_load_u32(ptr: *const u32, ordering: Ordering) -> u32 {
    unsafe { (*(ptr.cast::<AtomicU32>())).load(ordering) }
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
