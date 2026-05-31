#![no_std]
#![feature(stdarch_amdgpu)]

use core::arch::amdgpu;
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
pub fn global_id_x(block_dim_x: u32) -> usize {
    (block_idx_x() as usize * block_dim_x as usize) + thread_idx_x() as usize
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
