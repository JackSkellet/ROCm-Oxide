#![no_std]
#![feature(fn_traits)]
#![feature(stdarch_amdgpu)]
#![feature(unboxed_closures)]
#![allow(improper_ctypes_definitions)]

use rocm_oxide_device as gpu;
use rocm_oxide_kernel::{device_global, kernel, shared};

#[device_global]
pub static mut ADD_ONE_DELTA: f32 = 1.0;

#[shared]
pub static mut STATIC_LDS_U32: [u32; 256] = [0; 256];

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AffineParams {
    pub scale: f32,
    pub bias: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ControlParams {
    pub seed: u32,
    pub scale: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ControlPair {
    pub left: u32,
    pub right: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReturnPacket {
    pub sum: u64,
    pub folded: u32,
    pub tag: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CastPacket {
    pub wide: u64,
    pub signed_bits: u64,
    pub float_bits: u32,
    pub narrow: u32,
}

#[derive(Clone, Copy)]
pub struct RustLayoutParams {
    pub base: u32,
    pub stride: u32,
}

#[derive(Clone, Copy)]
struct ReturnRustPair {
    left: u32,
    right: u64,
}

#[derive(Clone, Copy)]
pub struct HostAffineClosure {
    pub base: u32,
    pub stride: u32,
    pub xor_mask: u32,
}

#[derive(Clone, Copy)]
pub struct HostReferenceClosure {
    pub bias: *const u32,
    pub scale: u32,
}

pub trait ClosureCaptureParams: Copy {
    fn apply(self, value: u32) -> u32;
}

impl ClosureCaptureParams for RustLayoutParams {
    fn apply(self, value: u32) -> u32 {
        value.wrapping_mul(self.stride).wrapping_add(self.base)
    }
}

impl core::ops::FnOnce<(u32,)> for HostAffineClosure {
    type Output = u32;

    extern "rust-call" fn call_once(self, args: (u32,)) -> Self::Output {
        args.0
            .wrapping_mul(self.stride)
            .wrapping_add(self.base)
            ^ self.xor_mask
    }
}

impl core::ops::FnOnce<(u32,)> for HostReferenceClosure {
    type Output = u32;

    extern "rust-call" fn call_once(self, args: (u32,)) -> Self::Output {
        let bias = unsafe { core::ptr::read_volatile(self.bias) };
        args.0.wrapping_mul(self.scale).wrapping_add(bias)
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum ControlKind {
    Zero = 2,
    One = 5,
    Many = 9,
    Custom = 13,
}

#[kernel]
pub unsafe extern "C" fn add_one(
    out: gpu::DeviceSliceMut<f32>,
    input: gpu::DeviceSlice<f32>,
) {
    let i = gpu::global_id_x();
    if i < out.len() {
        let delta = unsafe { ADD_ONE_DELTA };
        let value = unsafe { input.read_unchecked(i) };
        unsafe { out.write_unchecked(i, value + delta) };
    }
}

#[kernel]
pub unsafe extern "C" fn vector_add(
    out: gpu::DeviceSliceMut<f32>,
    a: gpu::DeviceSlice<f32>,
    b: gpu::DeviceSlice<f32>,
) {
    let i = gpu::global_id_x();
    if i < out.len() {
        let lhs = unsafe { a.read_unchecked(i) };
        let rhs = unsafe { b.read_unchecked(i) };
        unsafe { out.write_unchecked(i, lhs + rhs) };
    }
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel(monomorphize(u32))]
pub unsafe extern "C" fn generic_copy<T: Copy>(
    out: gpu::DeviceSliceMut<T>,
    input: gpu::DeviceSlice<T>,
    n: usize,
) {
    let i = gpu::global_id_x();
    if i < n {
        let value = unsafe { input.read_unchecked(i) };
        unsafe { out.write_unchecked(i, value) };
    }
}

// rocm-oxide: len(partials)=partial_count
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn lds_block_sum(
    partials: gpu::DeviceSliceMut<f32>,
    input: gpu::DeviceSlice<f32>,
    n: usize,
    partial_count: usize,
    block_x: u32,
) {
    let block_id = gpu::block_idx_x() as usize;
    if block_id >= partial_count {
        return;
    }

    let local = gpu::thread_idx_x() as usize;
    let block_dim = gpu::block_dim_x() as usize;
    if block_dim != block_x as usize {
        return;
    }

    let scratch = unsafe { gpu::DynamicSharedMem::<f32>::get() };
    let global = block_id * block_dim + local;
    let value = if global < n {
        unsafe { input.read_unchecked(global) }
    } else {
        0.0
    };
    unsafe { scratch.add(local).write(value) };
    gpu::workgroup_barrier();

    let mut active = block_dim;
    while active > 1 {
        let half = active.div_ceil(2);
        if local < half && local + half < active {
            let left = unsafe { scratch.add(local).read() };
            let right = unsafe { scratch.add(local + half).read() };
            unsafe { scratch.add(local).write(left + right) };
        }
        gpu::workgroup_barrier();
        active = half;
    }

    if local == 0 {
        let sum = unsafe { scratch.read() };
        unsafe { partials.write_unchecked(block_id, sum) };
    }
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn static_lds_reverse(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    n: usize,
) {
    let local = gpu::thread_idx_x() as usize;
    let block_dim = gpu::block_dim_x() as usize;
    if block_dim != 256 {
        return;
    }

    let block_base = gpu::block_idx_x() as usize * block_dim;
    let global = block_base + local;
    let value = if global < n {
        unsafe { input.read_unchecked(global) }
    } else {
        0
    };
    let scratch = core::ptr::addr_of_mut!(STATIC_LDS_U32).cast::<u32>();
    unsafe { scratch.add(local).write(value) };
    gpu::workgroup_barrier();

    if global < n {
        let reversed = unsafe { scratch.add(block_dim - 1 - local).read() };
        unsafe { out.write_unchecked(global, reversed) };
    }
}

// rocm-oxide: len(params)=1
#[kernel]
pub unsafe extern "C" fn affine_transform(
    out: gpu::DeviceSliceMut<f32>,
    input: gpu::DeviceSlice<f32>,
    params: gpu::DeviceSlice<AffineParams>,
) {
    let i = gpu::global_id_x();
    if i < out.len() {
        let env = unsafe { params.read_unchecked(0) };
        let value = unsafe { input.read_unchecked(i) };
        unsafe { out.write_unchecked(i, value * env.scale + env.bias) };
    }
}

// rocm-oxide: len(out)=16
// rocm-oxide: len(input)=4
#[kernel]
pub unsafe extern "C" fn math_intrinsics(
    out: gpu::DeviceSliceMut<f32>,
    input: gpu::DeviceSlice<f32>,
) {
    if gpu::global_id_x() != 0 {
        return;
    }

    let positive = unsafe { input.read_unchecked(0) };
    let zero = unsafe { input.read_unchecked(1) };
    let one = unsafe { input.read_unchecked(2) };
    let negative = unsafe { input.read_unchecked(3) };
    let nan = gpu::math::sqrt_f32(negative);
    let min_nan = gpu::math::min_f32(nan, one);
    let max_nan = gpu::math::max_f32(one, nan);

    unsafe {
        out.write_unchecked(0, gpu::math::sqrt_f32(positive));
        out.write_unchecked(1, gpu::math::rsqrt_f32(positive));
        out.write_unchecked(2, gpu::math::sin_f32(zero));
        out.write_unchecked(3, gpu::math::cos_f32(zero));
        out.write_unchecked(4, gpu::math::atan_f32(one));
        out.write_unchecked(5, gpu::math::min_f32(-2.0, 3.0));
        out.write_unchecked(6, gpu::math::max_f32(-2.0, 3.0));
        out.write_unchecked(7, gpu::math::sqrt_f64(positive as f64) as f32);
        out.write_unchecked(8, gpu::math::rsqrt_f64(positive as f64) as f32);
        out.write_unchecked(9, gpu::math::sin_f64(zero as f64) as f32);
        out.write_unchecked(10, gpu::math::cos_f64(zero as f64) as f32);
        out.write_unchecked(11, gpu::math::atan_f64(one as f64) as f32);
        out.write_unchecked(12, if nan != nan { 1.0 } else { 0.0 });
        out.write_unchecked(13, if min_nan != min_nan { 1.0 } else { 0.0 });
        out.write_unchecked(14, if max_nan != max_nan { 1.0 } else { 0.0 });
        out.write_unchecked(15, gpu::math::min_f64(-2.0, 3.0) as f32);
    }
}

// rocm-oxide: len(out)=4
// rocm-oxide: len(counters)=3
#[kernel]
pub unsafe extern "C" fn scoped_atomics(
    out: gpu::DeviceSliceMut<u32>,
    counters: gpu::DeviceSliceMut<u32>,
) {
    let i = gpu::global_id_x();
    if i >= 256 {
        return;
    }

    let counters_ptr = counters.as_mut_ptr();
    unsafe {
        gpu::atomic::atomic_add_u32_scoped(
            counters_ptr.add(0),
            1,
            gpu::AtomicScope::Workgroup,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_u32_scoped(
            counters_ptr.add(1),
            1,
            gpu::AtomicScope::Device,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_u32_scoped(
            counters_ptr.add(2),
            1,
            gpu::AtomicScope::System,
            gpu::AtomicOrdering::Relaxed,
        );
    }

    if i == 0 {
        unsafe {
            out.write_unchecked(0, gpu::WorkgroupAtomicU32::scope() as u32);
            out.write_unchecked(1, gpu::DeviceAtomicU32::scope() as u32);
            out.write_unchecked(2, gpu::SystemAtomicU32::scope() as u32);
            out.write_unchecked(3, gpu::AtomicOrdering::Relaxed as u32);
        }
    }
}

// rocm-oxide: len(out)=6
// rocm-oxide: len(f32_counters)=3
// rocm-oxide: len(f64_counters)=3
#[kernel]
pub unsafe extern "C" fn float_scoped_atomics(
    out: gpu::DeviceSliceMut<u32>,
    f32_counters: gpu::DeviceSliceMut<f32>,
    f64_counters: gpu::DeviceSliceMut<f64>,
) {
    let i = gpu::global_id_x();
    if i >= 64 {
        return;
    }

    let f32_ptr = f32_counters.as_mut_ptr();
    let f64_ptr = f64_counters.as_mut_ptr();
    unsafe {
        gpu::atomic::atomic_add_f32_scoped(
            f32_ptr.add(0),
            0.5,
            gpu::AtomicScope::Workgroup,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_f32_scoped(
            f32_ptr.add(1),
            1.25,
            gpu::AtomicScope::Device,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_f32_scoped(
            f32_ptr.add(2),
            -0.25,
            gpu::AtomicScope::System,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_f64_scoped(
            f64_ptr.add(0),
            0.5,
            gpu::AtomicScope::Workgroup,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_f64_scoped(
            f64_ptr.add(1),
            1.25,
            gpu::AtomicScope::Device,
            gpu::AtomicOrdering::Relaxed,
        );
        gpu::atomic::atomic_add_f64_scoped(
            f64_ptr.add(2),
            -0.25,
            gpu::AtomicScope::System,
            gpu::AtomicOrdering::Relaxed,
        );
    }

    gpu::workgroup_barrier();
    if i == 0 {
        let loaded_device = unsafe {
            gpu::atomic::atomic_load_f32_scoped(
                f32_ptr.add(1),
                gpu::AtomicScope::Device,
                gpu::AtomicOrdering::Relaxed,
            )
        };
        let loaded_system = unsafe {
            gpu::atomic::atomic_load_f64_scoped(
                f64_ptr.add(2),
                gpu::AtomicScope::System,
                gpu::AtomicOrdering::Relaxed,
            )
        };
        unsafe {
            out.write_unchecked(0, gpu::WorkgroupAtomicF32::scope() as u32);
            out.write_unchecked(1, gpu::DeviceAtomicF32::scope() as u32);
            out.write_unchecked(2, gpu::SystemAtomicF32::scope() as u32);
            out.write_unchecked(3, gpu::DeviceAtomicF64::scope() as u32);
            out.write_unchecked(4, loaded_device.to_bits());
            out.write_unchecked(5, loaded_system.to_bits() as u32);
        }
    }
}

// rocm-oxide: len(out)=18
// rocm-oxide: len(scan_out)=n
#[kernel]
pub unsafe extern "C" fn block_collectives_probe(
    out: gpu::DeviceSliceMut<u32>,
    scan_out: gpu::DeviceSliceMut<u32>,
    n: usize,
    block_x: u32,
) {
    let block = gpu::this_thread_block();
    let rank = block.thread_rank();
    let i = gpu::global_id_x();
    let active = i < n;
    let value = if active { rank + 1 } else { 0 };
    let scratch_u32 = unsafe { gpu::DynamicSharedMem::<u8>::offset(0).cast::<u32>() };
    let scratch_i32 = unsafe {
        gpu::DynamicSharedMem::<u8>::offset(block_x as usize * core::mem::size_of::<u32>())
            .cast::<i32>()
    };
    let scratch_f32 = unsafe {
        gpu::DynamicSharedMem::<u8>::offset(
            block_x as usize * (core::mem::size_of::<u32>() + core::mem::size_of::<i32>()),
        )
        .cast::<f32>()
    };

    let block_sum = unsafe { block.reduce_add_u32(scratch_u32, value) };
    let block_i32_sum =
        unsafe { block.reduce_add_i32(scratch_i32, if active { value as i32 - 2 } else { 0 }) };
    let block_f32_sum = unsafe { block.reduce_add_f32(scratch_f32, value as f32 * 0.5) };
    let min_u32 = unsafe { block.reduce_min_u32(scratch_u32, if active { value } else { u32::MAX }) };
    let max_u32 = unsafe { block.reduce_max_u32(scratch_u32, value) };
    let min_i32 = unsafe {
        block.reduce_min_i32(scratch_i32, if active { value as i32 - 2 } else { i32::MAX })
    };
    let max_i32 = unsafe { block.reduce_max_i32(scratch_i32, if active { value as i32 - 2 } else { i32::MIN }) };
    let min_f32 = unsafe {
        block.reduce_min_f32(
            scratch_f32,
            if active {
                value as f32 * 0.5
            } else {
                f32::INFINITY
            },
        )
    };
    let max_f32 = unsafe { block.reduce_max_f32(scratch_f32, value as f32 * 0.5) };
    let and_bits = unsafe { block.reduce_and_u32(scratch_u32, if active { value } else { u32::MAX }) };
    let or_bits = unsafe { block.reduce_or_u32(scratch_u32, value) };
    let xor_bits = unsafe { block.reduce_xor_u32(scratch_u32, value) };
    let inclusive = unsafe { block.scan_inclusive_add_u32(scratch_u32, value) };
    let exclusive = unsafe { block.scan_exclusive_add_u32(scratch_u32, value) };
    if active {
        unsafe { scan_out.write_unchecked(i, inclusive) };
    }

    if rank == 0 {
        unsafe {
            out.write_unchecked(0, block_sum);
            out.write_unchecked(1, block_i32_sum as u32);
            out.write_unchecked(2, block_f32_sum.to_bits());
            out.write_unchecked(3, block.size());
            out.write_unchecked(8, min_u32);
            out.write_unchecked(9, max_u32);
            out.write_unchecked(10, min_i32 as u32);
            out.write_unchecked(11, max_i32 as u32);
            out.write_unchecked(12, min_f32.to_bits());
            out.write_unchecked(13, max_f32.to_bits());
            out.write_unchecked(14, and_bits);
            out.write_unchecked(15, or_bits);
            out.write_unchecked(16, xor_bits);
            out.write_unchecked(17, 1);
        }
    }
    if rank == 7 {
        unsafe {
            out.write_unchecked(4, inclusive);
            out.write_unchecked(5, exclusive);
        }
    }
    if rank == block_x - 1 {
        unsafe {
            out.write_unchecked(6, inclusive);
            out.write_unchecked(7, exclusive);
        }
    }
}

// rocm-oxide: len(out)=36
#[kernel]
pub unsafe extern "C" fn block_collectives_ext_probe(
    out: gpu::DeviceSliceMut<u64>,
    n: usize,
    block_x: u32,
) {
    let block = gpu::this_thread_block();
    let rank = block.thread_rank();
    let i = gpu::global_id_x();
    let active = i < n;
    let lane = rank as u64;
    let value = lane + 1;
    let wide = if active { (1u64 << 40) + value } else { 0 };
    let signed = if active {
        -((1i64 << 33) + value as i64)
    } else {
        0
    };
    let fvalue = if active { value as f64 * 0.25 } else { 0.0 };
    let bit = if rank < 64 { 1u64 << rank } else { 0 };
    let descending = if active {
        block_x as u64 - lane
    } else {
        u64::MAX
    };
    let scratch_u64 = unsafe { gpu::DynamicSharedMem::<u8>::offset(0).cast::<u64>() };
    let scratch_i64 = unsafe {
        gpu::DynamicSharedMem::<u8>::offset(block_x as usize * core::mem::size_of::<u64>())
            .cast::<i64>()
    };
    let scratch_f64 = unsafe {
        gpu::DynamicSharedMem::<u8>::offset(
            block_x as usize * (core::mem::size_of::<u64>() + core::mem::size_of::<i64>()),
        )
        .cast::<f64>()
    };

    let add_u64 = unsafe { block.reduce_add_u64(scratch_u64, wide) };
    let add_i64 = unsafe { block.reduce_add_i64(scratch_i64, signed) };
    let add_f64 = unsafe { block.reduce_add_f64(scratch_f64, fvalue) };
    let min_u64 = unsafe { block.reduce_min_u64(scratch_u64, if active { wide } else { u64::MAX }) };
    let max_u64 = unsafe { block.reduce_max_u64(scratch_u64, wide) };
    let min_i64 = unsafe { block.reduce_min_i64(scratch_i64, if active { signed } else { i64::MAX }) };
    let max_i64 = unsafe { block.reduce_max_i64(scratch_i64, if active { signed } else { i64::MIN }) };
    let min_f64 = unsafe {
        block.reduce_min_f64(
            scratch_f64,
            if active { fvalue } else { f64::INFINITY },
        )
    };
    let max_f64 = unsafe {
        block.reduce_max_f64(
            scratch_f64,
            if active {
                fvalue
            } else {
                f64::NEG_INFINITY
            },
        )
    };
    let and_u64 = unsafe { block.reduce_and_u64(scratch_u64, if active { !bit } else { u64::MAX }) };
    let or_u64 = unsafe { block.reduce_or_u64(scratch_u64, if active { bit } else { 0 }) };
    let xor_u64 = unsafe { block.reduce_xor_u64(scratch_u64, if active { bit } else { 0 }) };
    let and_i64 = unsafe { block.reduce_and_i64(scratch_i64, if active { !(bit as i64) } else { -1 }) };
    let or_i64 = unsafe { block.reduce_or_i64(scratch_i64, if active { bit as i64 } else { 0 }) };
    let xor_i64 = unsafe { block.reduce_xor_i64(scratch_i64, if active { bit as i64 } else { 0 }) };

    let scan_add_u64_inclusive = unsafe { block.scan_inclusive_add_u64(scratch_u64, wide) };
    let scan_add_u64_exclusive = unsafe { block.scan_exclusive_add_u64(scratch_u64, wide) };
    let scan_add_f64_inclusive = unsafe { block.scan_inclusive_add_f64(scratch_f64, fvalue) };
    let scan_add_f64_exclusive = unsafe { block.scan_exclusive_add_f64(scratch_f64, fvalue) };
    let scan_min_u64_inclusive =
        unsafe { block.scan_inclusive_min_u64(scratch_u64, descending) };
    let scan_min_u64_exclusive =
        unsafe { block.scan_exclusive_min_u64(scratch_u64, descending) };
    let scan_max_u64_inclusive = unsafe { block.scan_inclusive_max_u64(scratch_u64, value) };
    let scan_max_u64_exclusive = unsafe { block.scan_exclusive_max_u64(scratch_u64, value) };
    let scan_or_u64_inclusive = unsafe { block.scan_inclusive_or_u64(scratch_u64, bit) };
    let scan_or_u64_exclusive = unsafe { block.scan_exclusive_or_u64(scratch_u64, bit) };
    let scan_xor_u64_inclusive = unsafe { block.scan_inclusive_xor_u64(scratch_u64, bit) };
    let scan_xor_u64_exclusive = unsafe { block.scan_exclusive_xor_u64(scratch_u64, bit) };
    let scan_and_u64_inclusive = unsafe { block.scan_inclusive_and_u64(scratch_u64, !bit) };
    let scan_and_u64_exclusive = unsafe { block.scan_exclusive_and_u64(scratch_u64, !bit) };

    if rank == 0 {
        unsafe {
            out.write_unchecked(0, add_u64);
            out.write_unchecked(1, add_i64 as u64);
            out.write_unchecked(2, add_f64.to_bits());
            out.write_unchecked(3, min_u64);
            out.write_unchecked(4, max_u64);
            out.write_unchecked(5, min_i64 as u64);
            out.write_unchecked(6, max_i64 as u64);
            out.write_unchecked(7, min_f64.to_bits());
            out.write_unchecked(8, max_f64.to_bits());
            out.write_unchecked(9, and_u64);
            out.write_unchecked(10, or_u64);
            out.write_unchecked(11, xor_u64);
            out.write_unchecked(12, and_i64 as u64);
            out.write_unchecked(13, or_i64 as u64);
            out.write_unchecked(14, xor_i64 as u64);
        }
    }
    if rank == 7 {
        unsafe {
            out.write_unchecked(15, scan_add_u64_inclusive);
            out.write_unchecked(16, scan_add_u64_exclusive);
            out.write_unchecked(17, scan_add_f64_inclusive.to_bits());
            out.write_unchecked(18, scan_add_f64_exclusive.to_bits());
            out.write_unchecked(19, scan_min_u64_inclusive);
            out.write_unchecked(20, scan_min_u64_exclusive);
            out.write_unchecked(21, scan_max_u64_inclusive);
            out.write_unchecked(22, scan_max_u64_exclusive);
            out.write_unchecked(23, scan_or_u64_inclusive);
            out.write_unchecked(24, scan_or_u64_exclusive);
            out.write_unchecked(25, scan_xor_u64_inclusive);
            out.write_unchecked(26, scan_xor_u64_exclusive);
            out.write_unchecked(27, scan_and_u64_inclusive);
            out.write_unchecked(28, scan_and_u64_exclusive);
        }
    }
    if rank == block_x - 1 {
        unsafe {
            out.write_unchecked(29, scan_add_u64_inclusive);
            out.write_unchecked(30, scan_add_u64_exclusive);
            out.write_unchecked(31, scan_min_u64_inclusive);
            out.write_unchecked(32, scan_min_u64_exclusive);
            out.write_unchecked(33, scan_or_u64_inclusive);
            out.write_unchecked(34, scan_or_u64_exclusive);
            out.write_unchecked(35, 1);
        }
    }
}

// rocm-oxide: len(out)=6
#[kernel]
pub unsafe extern "C" fn debug_helpers_probe(out: gpu::DeviceSliceMut<u32>) {
    let i = gpu::global_id_x();
    if i != 0 {
        return;
    }

    let dispatch = gpu::debug::dispatch_id();
    gpu::debug::sleep::<0>();
    let pc = gpu::debug::program_counter();
    gpu::debug::assert_or_trap(pc != 0);
    unsafe {
        out.write_unchecked(0, dispatch as u32);
        out.write_unchecked(1, (dispatch >> 32) as u32);
        out.write_unchecked(2, 1);
        out.write_unchecked(3, (pc != 0) as u32);
        out.write_unchecked(4, pc as u32);
        out.write_unchecked(5, (pc >> 32) as u32);
    }
}

// rocm-oxide: len(out)=12
#[kernel]
pub unsafe extern "C" fn cooperative_groups_probe(out: gpu::DeviceSliceMut<u32>) {
    let block = gpu::this_thread_block();
    let wave = gpu::this_wavefront();
    let tile = gpu::tiled_partition::<32>(block);
    let rank = block.thread_rank();
    let wave_any_first = wave.any(rank == 0) as u32;
    let wave_all_in_bounds = wave.all(wave.thread_rank() < wave.size()) as u32;
    let wave_max_lane = wave.reduce_max_u32(wave.thread_rank());

    if rank == 0 {
        unsafe {
            out.write_unchecked(0, block.size());
            out.write_unchecked(1, block.group_index_x());
            out.write_unchecked(2, rank);
            out.write_unchecked(3, wave.size());
            out.write_unchecked(4, tile.size());
            out.write_unchecked(8, wave_any_first);
            out.write_unchecked(9, wave_all_in_bounds);
            out.write_unchecked(10, wave_max_lane);
        }
    }
    if rank == 31 {
        unsafe {
            out.write_unchecked(5, tile.thread_rank());
            out.write_unchecked(6, tile.meta_group_rank());
        }
    }
    if rank == 32 {
        unsafe {
            out.write_unchecked(7, tile.meta_group_rank());
            out.write_unchecked(11, wave.meta_group_rank());
        }
    }
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(pairs)=n
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn compiler_parity_matrix(
    out: gpu::DeviceSliceMut<u32>,
    pairs: gpu::DeviceSliceMut<ControlPair>,
    input: gpu::DeviceSlice<u32>,
    params: ControlParams,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let value = unsafe { input.read_unchecked(i) };
    let pair = control_pair(value, params);
    let result = control_score(value, params, pair);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    let disjoint_pairs = unsafe { gpu::DisjointSliceMut::new_unchecked(pairs) };
    disjoint_out.write_for_thread(thread, result);
    disjoint_pairs.write_for_thread(thread, pair);
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn compiler_layout_probe(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    params: RustLayoutParams,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let value = unsafe { input.read_unchecked(i) };
    let result = value.wrapping_mul(params.stride).wrapping_add(params.base);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    disjoint_out.write_for_thread(thread, result);
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel(monomorphize(RustLayoutParams))]
pub unsafe extern "C" fn compiler_move_closure_probe<P: ClosureCaptureParams>(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    params: P,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let captured = params;
    let transform = move |value: u32| captured.apply(value).wrapping_add((i as u32) & 1);
    let value = unsafe { input.read_unchecked(i) };
    let result = apply_device_closure(value, transform);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    disjoint_out.write_for_thread(thread, result);
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel(monomorphize(HostAffineClosure))]
pub unsafe extern "C" fn compiler_host_closure_arg_probe<F: FnOnce(u32) -> u32 + Copy>(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    f: F,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let value = unsafe { input.read_unchecked(i) };
    let result = apply_device_closure(value.wrapping_add((i as u32) & 3), f);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    disjoint_out.write_for_thread(thread, result);
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel(monomorphize(HostReferenceClosure))]
pub unsafe extern "C" fn compiler_host_reference_closure_probe<F: FnOnce(u32) -> u32 + Copy>(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    f: F,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let value = unsafe { input.read_unchecked(i) };
    let result = apply_device_closure(value.wrapping_add((i as u32) & 1), f);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    disjoint_out.write_for_thread(thread, result);
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn compiler_flow_cast_probe(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let result = flow_cast_score(input, n, i);
    let byte_offset = i * core::mem::size_of::<u32>();
    let slot = unsafe { out.as_mut_ptr().cast::<u8>().add(byte_offset).cast::<u32>() };
    unsafe { core::ptr::write(slot, result) };
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(packets)=n
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn compiler_return_value_probe(
    out: gpu::DeviceSliceMut<u64>,
    packets: gpu::DeviceSliceMut<ReturnPacket>,
    input: gpu::DeviceSlice<u32>,
    params: ControlParams,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let value = unsafe { input.read_unchecked(i) };
    let pair = return_rust_pair(value, params);
    let packet = return_packet(value, pair);
    let score = return_packet_score(packet);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    let disjoint_packets = unsafe { gpu::DisjointSliceMut::new_unchecked(packets) };
    disjoint_out.write_for_thread(thread, score);
    disjoint_packets.write_for_thread(thread, packet);
}

// rocm-oxide: len(out)=n
// rocm-oxide: len(packets)=n
// rocm-oxide: len(input)=n
#[kernel]
pub unsafe extern "C" fn compiler_arithmetic_cast_probe(
    out: gpu::DeviceSliceMut<u64>,
    packets: gpu::DeviceSliceMut<CastPacket>,
    input: gpu::DeviceSlice<u32>,
    n: usize,
) {
    let thread = gpu::thread_index_x_witness();
    let i = thread.get();
    if i >= n {
        return;
    }

    let value = unsafe { input.read_unchecked(i) };
    let packet = cast_packet(value, i);
    let score = cast_packet_score(packet);
    let disjoint_out = unsafe { gpu::DisjointSliceMut::new_unchecked(out) };
    let disjoint_packets = unsafe { gpu::DisjointSliceMut::new_unchecked(packets) };
    disjoint_out.write_for_thread(thread, score);
    disjoint_packets.write_for_thread(thread, packet);
}

// rocm-oxide: len(out)=24
// rocm-oxide: len(i32_counter)=1
// rocm-oxide: len(u64_counter)=1
// rocm-oxide: len(i64_counter)=1
#[kernel]
pub unsafe extern "C" fn device_api_breadth_probe(
    out: gpu::DeviceSliceMut<u32>,
    i32_counter: gpu::DeviceSliceMut<i32>,
    u64_counter: gpu::DeviceSliceMut<u64>,
    i64_counter: gpu::DeviceSliceMut<i64>,
) {
    let wave = gpu::this_wavefront();
    let lane = wave.thread_rank();
    let value = lane + 1;
    let one_hot = 1u32 << (lane & 31);
    let shuffle_lane5 = wave.shuffle_u32(value, 5);
    let shuffle_down = wave.shuffle_down_u32(value, 1);
    let shuffle_up = wave.shuffle_up_u32(value, 1);
    let shuffle_xor = wave.shuffle_xor_u32(value, 1);
    let first = wave.read_first_u32(value + 9);
    let first64 = gpu::read_first_lane_u64(100 + lane as u64);
    let sum = wave.reduce_add_u32(value);
    let min_i32 = wave.reduce_min_i32(-(lane as i32));
    let max_i32 = wave.reduce_max_i32(-(lane as i32));
    let or_bits = wave.reduce_or_u32(one_hot);
    let xor_bits = wave.reduce_xor_u32(one_hot);
    let match_mask = wave.match_any_u32(lane & 3);
    let any_lane_zero = wave.any(lane == 0);
    let all_lanes_in_wave = wave.all(lane < wave.size());
    let no_lane_out_of_wave = wave.none(lane >= wave.size());
    let elected = wave.elected();

    if lane < 32 {
        unsafe {
            gpu::atomic::atomic_add_i32_scoped(
                i32_counter.as_mut_ptr(),
                -1,
                gpu::AtomicScope::Device,
                gpu::AtomicOrdering::Relaxed,
            );
            gpu::atomic::atomic_add_u64_scoped(
                u64_counter.as_mut_ptr(),
                1,
                gpu::AtomicScope::Device,
                gpu::AtomicOrdering::Relaxed,
            );
            gpu::atomic::atomic_add_i64_scoped(
                i64_counter.as_mut_ptr(),
                -2,
                gpu::AtomicScope::Device,
                gpu::AtomicOrdering::Relaxed,
            );
        }
    }

    let barrier = gpu::workgroup_barrier_token().arrive_and_wait();

    if lane == 0 {
        unsafe {
            out.write_unchecked(0, shuffle_lane5);
            out.write_unchecked(1, shuffle_down);
            out.write_unchecked(3, shuffle_xor);
            out.write_unchecked(4, sum);
            out.write_unchecked(5, min_i32 as u32);
            out.write_unchecked(6, max_i32 as u32);
            out.write_unchecked(7, or_bits);
            out.write_unchecked(8, xor_bits);
            out.write_unchecked(9, match_mask as u32);
            out.write_unchecked(10, any_lane_zero as u32);
            out.write_unchecked(11, all_lanes_in_wave as u32);
            out.write_unchecked(12, no_lane_out_of_wave as u32);
            out.write_unchecked(13, elected as u32);
            out.write_unchecked(14, first);
            out.write_unchecked(
                15,
                gpu::atomic::atomic_load_i32_scoped(
                    i32_counter.as_ptr(),
                    gpu::AtomicScope::Device,
                    gpu::AtomicOrdering::Relaxed,
                ) as u32,
            );
            out.write_unchecked(
                16,
                gpu::atomic::atomic_load_u64_scoped(
                    u64_counter.as_ptr(),
                    gpu::AtomicScope::Device,
                    gpu::AtomicOrdering::Relaxed,
                ) as u32,
            );
            let i64_total = gpu::atomic::atomic_load_i64_scoped(
                i64_counter.as_ptr(),
                gpu::AtomicScope::Device,
                gpu::AtomicOrdering::Relaxed,
            );
            out.write_unchecked(17, (-i64_total) as u32);
            out.write_unchecked(18, first64 as u32);
            out.write_unchecked(19, gpu::DeviceAtomicI32::scope() as u32);
            out.write_unchecked(20, gpu::DeviceAtomicU64::scope() as u32);
            out.write_unchecked(21, gpu::DeviceAtomicI64::scope() as u32);
        }
    }
    if lane == 1 {
        unsafe { out.write_unchecked(2, shuffle_up) };
    }

    let _ = barrier.arrive_and_wait();
}

#[kernel]
pub unsafe extern "C" fn rainbow_geometry(
    frame: gpu::DeviceSliceMut<u32>,
    width: u32,
    height: u32,
    frame_index: u32,
) {
    let i = gpu::global_id_x();
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & (width - 1);
    let y = (i as u32) >> 10;
    let cx = (width >> 1) as i32;
    let cy = (height >> 1) as i32;
    let dx = x as i32 - cx;
    let dy = y as i32 - cy;

    let ax = abs_i32(dx) as u32;
    let ay = abs_i32(dy) as u32;
    let r2 = (dx * dx + dy * dy) as u32;
    let t = frame_index.wrapping_mul(5);

    let rings = (r2 >> 7).wrapping_add(t);
    let diagonals = (x ^ y).wrapping_add(t.wrapping_mul(3));
    let diamonds = (ax + ay).wrapping_mul(3).wrapping_add(t);
    let hue = rings.wrapping_add(diagonals).wrapping_add(diamonds) & 255;

    let grid = if ((x.wrapping_add(frame_index)) & 31) < 2
        || ((y.wrapping_add(frame_index.wrapping_mul(2))) & 31) < 2
    {
        70
    } else {
        0
    };
    let pulse = 96 + ((rings ^ diagonals) & 127);
    let rgb = wheel(hue, clamp_u32(pulse + grid, 0, 255));

    unsafe { frame.write_unchecked(i, rgb) };
}

#[kernel]
pub unsafe extern "C" fn stress_pattern(
    frame: gpu::DeviceSliceMut<u32>,
    frame_index: u32,
    mode: u32,
    work_iters: u32,
) {
    let i = gpu::global_id_x();
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;
    let cx = 512i32;
    let cy = 256i32;
    let dx = x as i32 - cx;
    let dy = y as i32 - cy;
    let ax = abs_i32(dx) as u32;
    let ay = abs_i32(dy) as u32;
    let r2 = (dx * dx + dy * dy) as u32;

    let mut v = hash32((i as u32).wrapping_add(frame_index.wrapping_mul(747_796_405)));
    let mut k = 0u32;
    while k < work_iters {
        v = hash32(
            v.wrapping_add(x.wrapping_mul(1_103_515_245))
                .wrapping_add(y.wrapping_mul(12_345))
                .wrapping_add(k.wrapping_mul(2_654_435_761)),
        );
        k = k.wrapping_add(1);
    }

    let t = frame_index.wrapping_mul(3);
    let m = mode & 7;
    let hue = if m == 0 {
        (x.wrapping_mul(3) ^ y.wrapping_mul(5))
            .wrapping_add(t)
            .wrapping_add(v >> 24)
    } else if m == 1 {
        (r2 >> 6)
            .wrapping_add(t.wrapping_mul(2))
            .wrapping_add(v >> 25)
    } else if m == 2 {
        ((x.wrapping_add(t) & y.wrapping_add(t.wrapping_mul(2))) << 1).wrapping_add(v >> 24)
    } else if m == 3 {
        ax.wrapping_add(ay)
            .wrapping_mul(4)
            .wrapping_add(t)
            .wrapping_add(v >> 24)
    } else if m == 4 {
        ((x ^ y).wrapping_add((ax | ay) << 2)).wrapping_add(t.wrapping_mul(5))
    } else if m == 5 {
        v ^ (r2 >> 4) ^ t
    } else if m == 6 {
        ((x.wrapping_mul(y.wrapping_add(1))) >> 3).wrapping_add(v >> 24)
    } else {
        (ax.wrapping_mul(ax).wrapping_add(ay.wrapping_mul(ay)) >> 5)
            .wrapping_add(v >> 23)
            .wrapping_add(t)
    } & 255;

    let edge = if (x & 63) < 2 || (y & 63) < 2 { 60 } else { 0 };
    let intensity = 128 + ((v >> 24) & 95) + edge;
    let rgb = wheel(hue, clamp_u32(intensity, 0, 255));

    unsafe { frame.write_unchecked(i, rgb) };
}

#[kernel]
pub unsafe extern "C" fn stress_3d(
    frame: gpu::DeviceSliceMut<u32>,
    frame_index: u32,
    mode: u32,
    work_iters: u32,
) {
    let i = gpu::global_id_x();
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;
    let px = x as i32 - 512;
    let py = y as i32 - 256;
    let t = frame_index.wrapping_mul(7);
    let m = mode & 7;

    let cam_x = (((frame_index.wrapping_mul(5)) & 511) as i32) - 256;
    let cam_y = (((frame_index.wrapping_mul(3)) & 255) as i32) - 128;
    let mut hit = 0u32;
    let mut hit_depth = work_iters;
    let mut hit_hash = 0u32;
    let mut hit_face = 0u32;
    let mut k = 2u32;

    while k <= work_iters {
        let z = 96 + (k << 3);
        let bend = (((z + t) & 255) as i32) - 128;
        let wx = ((px * z as i32) >> 8) + cam_x + ((bend * py) >> 10);
        let wy = ((py * z as i32) >> 8) + cam_y - ((bend * px) >> 11);
        let wz = z as i32 + ((t as i32) << 2);

        let cell_x = wx >> 6;
        let cell_y = wy >> 6;
        let cell_z = wz >> 6;
        let lx = abs_i32((wx & 63) - 32) as u32;
        let ly = abs_i32((wy & 63) - 32) as u32;
        let lz = abs_i32((wz & 63) - 32) as u32;

        let tunnel = (cell_x * cell_x + cell_y * cell_y) as u32;
        let h = hash32(
            (cell_x as u32).wrapping_mul(73_856_093)
                ^ (cell_y as u32).wrapping_mul(19_349_663)
                ^ (cell_z as u32).wrapping_mul(83_492_791)
                ^ mode.wrapping_mul(912_931),
        );

        let shell = lx > 24 || ly > 24 || lz > 24;
        let occupied = if m == 0 {
            (h & 7) == 0 && tunnel > 4
        } else if m == 1 {
            ((cell_x ^ cell_y ^ cell_z) & 3) == 0 && tunnel > 9
        } else if m == 2 {
            (h & 15) < 3 && (cell_z & 1) == 0
        } else if m == 3 {
            tunnel > 16 && ((cell_z + cell_x - cell_y) & 3) == 0
        } else if m == 4 {
            (cell_x & cell_y & cell_z) != 0 && (h & 3) == 0
        } else if m == 5 {
            (h & 31) < 6
        } else if m == 6 {
            tunnel > 25 && ((cell_x | cell_y | cell_z) & 5) == 0
        } else {
            (h & 15) < 4 && tunnel > 6
        };

        if occupied && shell {
            hit = 1;
            hit_depth = k;
            hit_hash = h;
            hit_face = if lx >= ly && lx >= lz {
                0
            } else if ly >= lx && ly >= lz {
                1
            } else {
                2
            };
            break;
        }

        k = k.wrapping_add(1);
    }

    let rgb = if hit != 0 {
        let depth_fog = 255u32.saturating_sub((hit_depth.wrapping_mul(220)) >> 8);
        let face_light = if hit_face == 0 {
            70
        } else if hit_face == 1 {
            35
        } else {
            0
        };
        let value = clamp_u32(
            depth_fog
                .wrapping_add(face_light)
                .wrapping_add(hit_hash & 31),
            0,
            255,
        );
        let hue = hit_hash
            .wrapping_add(hit_depth.wrapping_mul(3))
            .wrapping_add(t)
            .wrapping_add(mode.wrapping_mul(29))
            & 255;
        wheel(hue, value)
    } else {
        let sky = ((py + 256) as u32) >> 2;
        let star = if (hash32((x << 12) ^ y ^ (t >> 2)) & 511) == 0 {
            150
        } else {
            0
        };
        let hue = (160 + (sky >> 1) + (t >> 3)) & 255;
        wheel(hue, clamp_u32(18 + sky + star, 0, 190))
    };

    unsafe { frame.write_unchecked(i, rgb) };
}

// rocm-oxide: len(frame)=pixel_count
#[kernel]
pub unsafe extern "C" fn spectral_lattice(
    frame: gpu::DeviceSliceMut<u32>,
    width: u32,
    height: u32,
    pixel_count: usize,
    frame_index: u32,
    mode: u32,
    palette_a: f32,
    palette_b: f32,
    palette_c: f32,
    warp: f32,
    gain: f32,
    work_iterations: u32,
) {
    let i = gpu::global_id_x();
    if width == 0 || height == 0 || i >= pixel_count || i >= frame.len() {
        return;
    }

    let x = (i as u32) % width;
    let y = (i as u32) / width;
    let aspect = (width as f32) / (height as f32);
    let fx = (((x as f32) + 0.5) / (width as f32) - 0.5) * 2.0 * aspect;
    let fy = (((y as f32) + 0.5) / (height as f32) - 0.5) * -2.0;
    let t = (frame_index as f32) * 0.018;
    let radius = gpu::math::sqrt_f32(fx * fx + fy * fy) + 0.0001;

    let twist = gpu::math::sin_f32(radius * 8.5 - t * 1.7) * warp;
    let st = gpu::math::sin_f32(twist);
    let ct = gpu::math::cos_f32(twist);
    let px = fx * ct - fy * st;
    let py = fx * st + fy * ct;

    let lattice_a = abs_f32(gpu::math::sin_f32(px * 18.0 + py * 3.0 + t + palette_a));
    let lattice_b = abs_f32(gpu::math::cos_f32(py * 16.0 - px * 4.0 - t * 1.3 + palette_b));
    let ridge = 1.0 - abs_f32(lattice_a - lattice_b);
    let prism = abs_f32(gpu::math::sin_f32((px + py) * 9.0 + radius * 13.0 - t * 2.1));
    let bloom = 1.0 / (1.0 + radius * radius * 4.2);
    let center = 1.0 - clamp_f32(radius * 0.72, 0.0, 1.0);
    let work_iterations = min_u32(work_iterations, 1024);
    let mut work_detail = 0.0f32;
    let mut qx = px + palette_a * 0.031;
    let mut qy = py - palette_b * 0.027;
    let mut work = 0u32;
    while work < work_iterations {
        let step = work as f32;
        let phase = t * 0.13 + step * 0.119;
        let sx = gpu::math::sin_f32(qx * (3.0 + step * 0.017) + qy * 1.7 + phase);
        let cy = gpu::math::cos_f32(qy * (2.5 + step * 0.013) - qx * 1.3 - phase);
        work_detail += sx * cy;
        qx += sx * 0.006 + cy * 0.002;
        qy += cy * 0.006 - sx * 0.002;
        work = work.wrapping_add(1);
    }
    let work_detail = if work_iterations == 0 {
        0.0
    } else {
        work_detail / work_iterations as f32
    };
    let h = hash32((x << 16) ^ y);
    let spark = if (h & 4095) < 6 + ((center * 18.0) as u32) {
        let phase = ((h >> 12) & 255) as f32 * 0.024_543_693;
        let pulse = gpu::math::sin_f32(t * 0.72 + phase) * 0.5 + 0.5;
        0.16 + pulse * 0.22
    } else {
        0.0
    };

    let energy = clamp_f32(
        (0.06 + ridge * 0.44 + prism * 0.2 + bloom * 0.5 + spark + work_detail * 0.045)
            * gain,
        0.0,
        1.0,
    );
    let red = energy
        * clamp_f32(
            0.28 + bloom * 0.72 + gpu::math::sin_f32(px * 5.0 + palette_c + t) * 0.22,
            0.0,
            1.0,
        );
    let green = energy
        * clamp_f32(
            0.22 + ridge * 0.62 + gpu::math::cos_f32(py * 4.0 + palette_a - t) * 0.18,
            0.0,
            1.0,
        );
    let blue = energy
        * clamp_f32(
            0.42 + prism * 0.55 + gpu::math::sin_f32(radius * 10.0 + palette_b) * 0.2,
            0.0,
            1.0,
        );

    let mode = mode & 3;
    let rgb = if mode == 1 {
        pack_rgbf(
            energy,
            clamp_f32(ridge * gain, 0.0, 1.0),
            clamp_f32(bloom + spark, 0.0, 1.0),
        )
    } else if mode == 2 {
        let cut = if ridge > 0.86 { 1.0 } else { ridge * 0.34 };
        let veins = if prism > 0.92 { 1.0 } else { prism * 0.18 };
        pack_rgbf(
            clamp_f32(cut + bloom * 0.35, 0.0, 1.0),
            clamp_f32(veins + center * 0.42, 0.0, 1.0),
            clamp_f32(0.22 + ridge * prism * gain, 0.0, 1.0),
        )
    } else if mode == 3 {
        let hue_mix = gpu::math::sin_f32(px * palette_a + py * palette_b + t) * 0.5 + 0.5;
        pack_rgbf(
            clamp_f32(red * 0.7 + hue_mix * energy * 0.65, 0.0, 1.0),
            clamp_f32(green * 0.55 + bloom * gain * 0.55, 0.0, 1.0),
            clamp_f32(blue * 0.85 + (1.0 - hue_mix) * prism * 0.5, 0.0, 1.0),
        )
    } else {
        pack_rgbf(red, green, blue)
    };

    unsafe { frame.write_unchecked(i, rgb) };
}

// rocm-oxide: len(out)=pixel_count
// rocm-oxide: len(input)=pixel_count
// rocm-oxide: len(tile_stats)=tile_count
#[kernel]
pub unsafe extern "C" fn spectral_lds_tiles(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    tile_stats: gpu::DeviceSliceMut<u32>,
    pixel_count: usize,
    tile_count: usize,
    block_x: u32,
    mode: u32,
) {
    let block_id = gpu::block_idx_x() as usize;
    if block_id >= tile_count {
        return;
    }

    let local = gpu::thread_idx_x() as usize;
    let block_dim = gpu::block_dim_x() as usize;
    if block_dim == 0 || block_dim != block_x as usize {
        return;
    }

    let block_base = block_id * block_dim;
    let global = block_base + local;
    let data_count = min_usize(pixel_count, min_usize(input.len(), out.len()));
    let valid = if block_base >= data_count {
        0
    } else {
        min_usize(block_dim, data_count - block_base)
    };
    let value = if global < data_count {
        luminance(unsafe { input.read_unchecked(global) }) as u32
    } else {
        0
    };

    let scratch = unsafe { gpu::DynamicSharedMem::<u32>::get() };
    unsafe { scratch.add(local).write(value) };
    gpu::workgroup_barrier();

    let mut active = block_dim;
    while active > 1 {
        let half = active.div_ceil(2);
        if local < half && local + half < active {
            let left = unsafe { scratch.add(local).read() };
            let right = unsafe { scratch.add(local + half).read() };
            unsafe { scratch.add(local).write(left + right) };
        }
        gpu::workgroup_barrier();
        active = half;
    }

    let avg = if valid == 0 {
        0
    } else {
        (unsafe { scratch.read() }) / valid as u32
    };
    if local == 0 && block_id < tile_stats.len() {
        unsafe { tile_stats.write_unchecked(block_id, avg) };
    }
    gpu::workgroup_barrier();

    if global < data_count {
        let base = unsafe { input.read_unchecked(global) };
        let hue = avg
            .wrapping_add((block_id as u32).wrapping_mul(17))
            .wrapping_add(mode.wrapping_mul(43))
            & 255;
        let block_color = wheel(hue, clamp_u32(90 + avg, 0, 255));
        let edge = if (local & 31) == 0 || local >= valid.saturating_sub(1) {
            0xffffff
        } else {
            block_color
        };
        unsafe { out.write_unchecked(global, mix_color(base, edge, 0.58)) };
    }
}

// rocm-oxide: len(counters)=256
// rocm-oxide: len(input)=pixel_count
#[kernel]
pub unsafe extern "C" fn spectral_atomic_histogram(
    counters: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    pixel_count: usize,
) {
    let i = gpu::global_id_x();
    if i >= pixel_count || i >= input.len() || counters.len() < 256 {
        return;
    }

    let rgb = unsafe { input.read_unchecked(i) };
    let bucket = luminance(rgb) as usize;
    unsafe {
        gpu::atomic::atomic_add_u32_scoped(
            counters.as_mut_ptr().add(bucket),
            1,
            gpu::AtomicScope::Device,
            gpu::AtomicOrdering::Relaxed,
        );
    }
}

// rocm-oxide: len(out)=pixel_count
// rocm-oxide: len(input)=pixel_count
// rocm-oxide: len(counters)=256
#[kernel]
pub unsafe extern "C" fn spectral_histogram_overlay(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    counters: gpu::DeviceSlice<u32>,
    width: u32,
    height: u32,
    pixel_count: usize,
    frame_index: u32,
) {
    let i = gpu::global_id_x();
    if width == 0
        || height == 0
        || i >= pixel_count
        || i >= out.len()
        || i >= input.len()
        || counters.len() < 256
    {
        return;
    }

    let x = (i as u32) % width;
    let y = (i as u32) / width;
    let base = unsafe { input.read_unchecked(i) };
    let bucket = ((x * 256) / width) as usize;
    let count = unsafe { counters.read_unchecked(bucket) };
    let bar_height = min_u32(height / 3, count >> 6);
    let from_bottom = height.saturating_sub(1).saturating_sub(y);
    let rgb = if from_bottom < bar_height {
        wheel(
            (bucket as u32).wrapping_add(frame_index >> 2) & 255,
            clamp_u32(130 + (count >> 8), 0, 255),
        )
    } else {
        let bucket_color = wheel(bucket as u32, clamp_u32(34 + (count >> 9), 34, 160));
        mix_color(base, bucket_color, 0.22)
    };
    unsafe { out.write_unchecked(i, rgb) };
}

// rocm-oxide: len(out)=pixel_count
// rocm-oxide: len(input)=pixel_count
#[kernel]
pub unsafe extern "C" fn spectral_post_fx(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    width: u32,
    height: u32,
    pixel_count: usize,
    frame_index: u32,
    mode: u32,
    intensity: f32,
) {
    let i = gpu::global_id_x();
    if width == 0 || height == 0 || i >= pixel_count || i >= out.len() || i >= input.len() {
        return;
    }

    let x = (i as u32) % width;
    let y = (i as u32) / width;
    let left = unsafe { sample_frame(input.as_ptr(), width, height, x.saturating_sub(1), y) };
    let right = unsafe { sample_frame(input.as_ptr(), width, height, min_u32(x + 1, width - 1), y) };
    let up = unsafe { sample_frame(input.as_ptr(), width, height, x, y.saturating_sub(1)) };
    let down = unsafe { sample_frame(input.as_ptr(), width, height, x, min_u32(y + 1, height - 1)) };
    let center = unsafe { input.read_unchecked(i) };
    let edge = clamp_i32(
        abs_i32(luminance(left) - luminance(right)) + abs_i32(luminance(up) - luminance(down)),
        0,
        255,
    ) as u32;
    let t = frame_index.wrapping_mul(5).wrapping_add(mode.wrapping_mul(53));
    let hue = ((x ^ y).wrapping_add(edge).wrapping_add(t)) & 255;
    let glow = wheel(hue, clamp_u32(edge + 48, 0, 255));
    let k = clamp_f32(0.18 + intensity * 0.34, 0.0, 0.75);
    let rgb = if (mode & 1) == 0 {
        mix_color(center, glow, k)
    } else {
        let sharp = sharpen_rgb(center, glow, 7);
        mix_color(sharp, glow, k)
    };
    unsafe { out.write_unchecked(i, rgb) };
}

// rocm-oxide: len(camera)=13
#[kernel]
pub unsafe extern "C" fn raytrace_world(
    frame: gpu::DeviceSliceMut<u32>,
    camera: gpu::DeviceSlice<f32>,
    frame_index: u32,
) {
    let i = gpu::global_id_x();
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;

    let cx = unsafe { camera.read_unchecked(0) };
    let cy = unsafe { camera.read_unchecked(1) };
    let cz = unsafe { camera.read_unchecked(2) };
    let rx = unsafe { camera.read_unchecked(3) };
    let ry = unsafe { camera.read_unchecked(4) };
    let rz = unsafe { camera.read_unchecked(5) };
    let ux = unsafe { camera.read_unchecked(6) };
    let uy = unsafe { camera.read_unchecked(7) };
    let uz = unsafe { camera.read_unchecked(8) };
    let fx = unsafe { camera.read_unchecked(9) };
    let fy = unsafe { camera.read_unchecked(10) };
    let fz = unsafe { camera.read_unchecked(11) };
    let flags = unsafe { camera.read_unchecked(12) } as u32;

    let sx = ((x as f32) - 512.0) * (1.0 / 512.0);
    let sy = (288.0 - (y as f32)) * (1.0 / 512.0);
    let mut dx = fx + rx * sx * 1.28 + ux * sy;
    let mut dy = fy + ry * sx * 1.28 + uy * sy;
    let mut dz = fz + rz * sx * 1.28 + uz * sy;
    let inv_len = inv_sqrt(dx * dx + dy * dy + dz * dz);
    dx *= inv_len;
    dy *= inv_len;
    dz *= inv_len;

    let (t, nx, ny, nz, mr, mg, mb, material) = scene_hit(cx, cy, cz, dx, dy, dz, frame_index, 1);
    let rgb = if t < 9_000.0 {
        let px = cx + dx * t;
        let py = cy + dy * t;
        let pz = cz + dz * t;
        let mut shade = shade_hit(px, py, pz, nx, ny, nz, mr, mg, mb, frame_index, flags);

        if material == 2 && (flags & 2) != 0 {
            let nd = nx * dx + ny * dy + nz * dz;
            let rdx = dx - nx * (2.0 * nd);
            let rdy = dy - ny * (2.0 * nd);
            let rdz = dz - nz * (2.0 * nd);
            let (rt, rnx, rny, rnz, rr, rg, rb, _) = scene_hit(
                px + nx * 0.04,
                py + ny * 0.04,
                pz + nz * 0.04,
                rdx,
                rdy,
                rdz,
                frame_index,
                1,
            );
            let reflected = if rt < 9_000.0 {
                let rpx = px + nx * 0.04 + rdx * rt;
                let rpy = py + ny * 0.04 + rdy * rt;
                let rpz = pz + nz * 0.04 + rdz * rt;
                shade_hit(rpx, rpy, rpz, rnx, rny, rnz, rr, rg, rb, frame_index, flags)
            } else {
                sky_color(rdy)
            };
            shade = mix_color(shade, reflected, 0.42);
        }

        shade
    } else {
        sky_color(dy)
    };

    unsafe { frame.write_unchecked(i, rgb) };
}

#[kernel]
pub unsafe extern "C" fn window_fx(
    frame: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    frame_index: u32,
    mode: u32,
) {
    let i = gpu::global_id_x();
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;
    let m = mode & 15;
    let sharpness = ((mode >> 4) & 7) as i32;
    let upscale = (mode >> 8) & 3;
    let src_w = if upscale == 0 {
        1024u32
    } else if upscale == 1 {
        768u32
    } else if upscale == 2 {
        640u32
    } else {
        512u32
    };
    let src_h = if upscale == 0 {
        576u32
    } else if upscale == 1 {
        432u32
    } else if upscale == 2 {
        360u32
    } else {
        288u32
    };

    let mut src_xf = ((x as f32) + 0.5) * (src_w as f32) * (1.0 / 1024.0) - 0.5;
    let mut src_yf = ((y as f32) + 0.5) * (src_h as f32) * (1.0 / 576.0) - 0.5;

    if m == 5 {
        let fx = (x as f32 - 512.0) * (1.0 / 512.0);
        let fy = (y as f32 - 288.0) * (1.0 / 512.0);
        let tilt = ((frame_index & 255) as f32 - 128.0) * (1.0 / 420.0);
        let depth = 1.0 + fx * tilt;
        let wx = fx / depth;
        let wy = fy / depth;
        let is_outside = wx < -1.0 || wx > 1.0 || wy < -0.5625 || wy > 0.5625;
        if is_outside {
            unsafe { frame.write_unchecked(i, wheel((x ^ y).wrapping_add(frame_index) & 255, 26)) };
            return;
        }
        src_xf = wx * (src_w as f32 * 0.5) + (src_w as f32 * 0.5);
        src_yf = wy * (src_h as f32 * 0.5) + (src_h as f32 * 0.5);
    }

    src_xf = clamp_f32(src_xf, 0.0, (src_w - 1) as f32);
    src_yf = clamp_f32(src_yf, 0.0, (src_h - 1) as f32);
    let sx = src_xf as u32;
    let sy = src_yf as u32;
    let fx = src_xf - sx as f32;
    let fy = src_yf - sy as f32;
    let sx1 = min_u32(sx + 1, src_w - 1);
    let sy1 = min_u32(sy + 1, src_h - 1);

    let center = unsafe { bilinear_input(input.as_ptr(), sx, sy, sx1, sy1, fx, fy) };
    let left = unsafe { sample_input(input.as_ptr(), sx.saturating_sub(1), sy) };
    let right = unsafe { sample_input(input.as_ptr(), min_u32(sx + 1, src_w - 1), sy) };
    let up = unsafe { sample_input(input.as_ptr(), sx, sy.saturating_sub(1)) };
    let down = unsafe { sample_input(input.as_ptr(), sx, min_u32(sy + 1, src_h - 1)) };

    let cr = ((center >> 16) & 255) as i32;
    let cg = ((center >> 8) & 255) as i32;
    let cb = (center & 255) as i32;
    let blur_r = avg4(left >> 16, right >> 16, up >> 16, down >> 16);
    let blur_g = avg4(left >> 8, right >> 8, up >> 8, down >> 8);
    let blur_b = avg4(left, right, up, down);

    let t = frame_index.wrapping_mul(5);
    let sharpen_num = 2 + sharpness;
    let mut r = clamp_i32(cr + ((cr - blur_r) * sharpen_num) / 8, 0, 255) as u32;
    let mut g = clamp_i32(cg + ((cg - blur_g) * sharpen_num) / 8, 0, 255) as u32;
    let mut b = clamp_i32(cb + ((cb - blur_b) * sharpen_num) / 8, 0, 255) as u32;

    if m == 4 {
        let lx = luminance(left);
        let rx = luminance(right);
        let uy = luminance(up);
        let dy = luminance(down);
        let nx = clamp_i32(128 + (lx - rx), 0, 255) as u32;
        let ny = clamp_i32(128 + (uy - dy), 0, 255) as u32;
        let nz = 220u32.saturating_sub(((abs_i32(lx - rx) + abs_i32(uy - dy)) as u32) >> 1);
        unsafe { frame.write_unchecked(i, (nx << 16) | (ny << 8) | nz) };
        return;
    } else if m == 6 {
        let depth = luminance(center) as u32;
        unsafe {
            frame.write_unchecked(
                i,
                wheel(
                    (170u32.saturating_sub(depth >> 1)).wrapping_add(t) & 255,
                    depth,
                ),
            )
        };
        return;
    }

    if m == 1 {
        let hue = ((x ^ y).wrapping_add(t)) & 255;
        let glow = 48 + ((hash32(i as u32 ^ t) >> 25) & 63);
        let tint = wheel(hue, glow);
        r = min_u32(r + ((tint >> 16) & 255), 255);
        g = min_u32(g + ((tint >> 8) & 255), 255);
        b = min_u32(b + (tint & 255), 255);
    } else if m == 2 {
        let scan = if ((y + t) & 15) < 3 { 58 } else { 0 };
        let chroma = (((x + t) & 7) as i32) - 3;
        r = clamp_i32(r as i32 + scan + chroma * 10, 0, 255) as u32;
        g = clamp_i32(g as i32 + scan / 2, 0, 255) as u32;
        b = clamp_i32(b as i32 + scan - chroma * 10, 0, 255) as u32;
    } else if m == 3 {
        let dx = x as i32 - 512;
        let dy = y as i32 - 288;
        let dist = ((dx * dx + dy * dy) as u32) >> 11;
        let wave = ((dist + t) & 255) as i32 - 128;
        r = clamp_i32(r as i32 + wave / 3, 0, 255) as u32;
        g = clamp_i32(g as i32 + abs_i32(wave) / 4, 0, 255) as u32;
        b = clamp_i32(b as i32 - wave / 3, 0, 255) as u32;
    } else if m == 5 {
        let shade = 210 + (((x as i32 - 512) * (((frame_index & 255) as i32) - 128)) >> 11);
        r = clamp_i32((r as i32 * shade) >> 8, 0, 255) as u32;
        g = clamp_i32((g as i32 * shade) >> 8, 0, 255) as u32;
        b = clamp_i32((b as i32 * shade) >> 8, 0, 255) as u32;
    }

    unsafe { frame.write_unchecked(i, (r << 16) | (g << 8) | b) };
}

// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(depth)=pixel_count/4
#[kernel]
pub unsafe extern "C" fn depth_aware_upscale(
    frame: gpu::DeviceSliceMut<u32>,
    color: gpu::DeviceSlice<u32>,
    depth: gpu::DeviceSlice<f32>,
    pixel_count: usize,
    mode: u32,
) {
    let i = gpu::global_id_x();
    let _ = pixel_count;
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;
    let src_xf = ((x as f32) + 0.5) * 0.5 - 0.5;
    let src_yf = ((y as f32) + 0.5) * 0.5 - 0.5;
    let src_xf = clamp_f32(src_xf, 0.0, 511.0);
    let src_yf = clamp_f32(src_yf, 0.0, 287.0);
    let sx = src_xf as u32;
    let sy = src_yf as u32;
    let sx1 = min_u32(sx + 1, 511);
    let sy1 = min_u32(sy + 1, 287);
    let fx = src_xf - sx as f32;
    let fy = src_yf - sy as f32;

    let d00 = unsafe { sample_depth_512(depth.as_ptr(), sx, sy) };
    let d10 = unsafe { sample_depth_512(depth.as_ptr(), sx1, sy) };
    let d01 = unsafe { sample_depth_512(depth.as_ptr(), sx, sy1) };
    let d11 = unsafe { sample_depth_512(depth.as_ptr(), sx1, sy1) };
    let min_d = min_f32(min_f32(d00, d10), min_f32(d01, d11));
    let max_d = max_f32(max_f32(d00, d10), max_f32(d01, d11));
    let edge = max_d - min_d;

    let nearest_x = if fx < 0.5 { sx } else { sx1 };
    let nearest_y = if fy < 0.5 { sy } else { sy1 };
    let nearest = unsafe { sample_color_512(color.as_ptr(), nearest_x, nearest_y) };
    let smooth = unsafe { bilinear_color_512(color.as_ptr(), sx, sy, sx1, sy1, fx, fy) };
    let mut rgb = if edge > 0.09 { nearest } else { smooth };

    if (mode & 15) == 1 {
        let d = clamp_i32(((1.0 - d00) * 255.0) as i32, 0, 255) as u32;
        rgb = (d << 16) | (d << 8) | d;
    } else if (mode & 15) == 2 {
        let e = clamp_i32((edge * 900.0) as i32, 0, 255) as u32;
        rgb = (e << 16) | ((255 - e) << 8) | 32;
    } else {
        let sharp = ((mode >> 4) & 7) as i32;
        rgb = sharpen_rgb(rgb, nearest, sharp);
    }

    unsafe { frame.write_unchecked(i, rgb) };
}

// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(history_out)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(depth)=pixel_count/4
// rocm-oxide: len(motion_reactive)=pixel_count/4*3
// rocm-oxide: len(prev_history)=pixel_count
#[kernel]
pub unsafe extern "C" fn temporal_reconstruct_upscale(
    frame: gpu::DeviceSliceMut<u32>,
    history_out: gpu::DeviceSliceMut<u32>,
    color: gpu::DeviceSlice<u32>,
    depth: gpu::DeviceSlice<f32>,
    motion_reactive: gpu::DeviceSlice<f32>,
    prev_history: gpu::DeviceSlice<u32>,
    pixel_count: usize,
    mode: u32,
) {
    let i = gpu::global_id_x();
    let _ = pixel_count;
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;
    let src_xf = ((x as f32) + 0.5) * 0.5 - 0.5;
    let src_yf = ((y as f32) + 0.5) * 0.5 - 0.5;
    let src_xf = clamp_f32(src_xf, 0.0, 511.0);
    let src_yf = clamp_f32(src_yf, 0.0, 287.0);
    let sx = src_xf as u32;
    let sy = src_yf as u32;
    let sx1 = min_u32(sx + 1, 511);
    let sy1 = min_u32(sy + 1, 287);
    let fx = src_xf - sx as f32;
    let fy = src_yf - sy as f32;

    let current_smooth = unsafe { bilinear_color_512(color.as_ptr(), sx, sy, sx1, sy1, fx, fy) };
    let current_nearest = unsafe {
        sample_color_512(
            color.as_ptr(),
            if fx < 0.5 { sx } else { sx1 },
            if fy < 0.5 { sy } else { sy1 },
        )
    };
    let current = sharpen_rgb(current_smooth, current_nearest, ((mode >> 4) & 7) as i32);

    let mut nearest_depth = 10_000.0;
    let mut farthest_depth = -10_000.0;
    let mut motion_x = 0.0;
    let mut motion_y = 0.0;
    let mut reactive = 0.0;
    let base_x = sx as i32;
    let base_y = sy as i32;
    let mut oy = -1i32;
    while oy <= 1 {
        let mut ox = -1i32;
        while ox <= 1 {
            let nx = clamp_i32(base_x + ox, 0, 511) as u32;
            let ny = clamp_i32(base_y + oy, 0, 287) as u32;
            let d = unsafe { sample_depth_512(depth.as_ptr(), nx, ny) };
            farthest_depth = max_f32(farthest_depth, d);
            if d < nearest_depth {
                nearest_depth = d;
                motion_x = unsafe { sample_aux_512(motion_reactive.as_ptr(), nx, ny, 0) };
                motion_y = unsafe { sample_aux_512(motion_reactive.as_ptr(), nx, ny, 1) };
                reactive = unsafe { sample_aux_512(motion_reactive.as_ptr(), nx, ny, 2) };
            }
            ox += 1;
        }
        oy += 1;
    }

    let edge = clamp_f32((farthest_depth - nearest_depth) * 7.0, 0.0, 1.0);
    let prev_x = clamp_f32(x as f32 + motion_x, 0.0, 1023.0);
    let prev_y = clamp_f32(y as f32 + motion_y, 0.0, 575.0);
    let history = unsafe { bilinear_history_1024(prev_history.as_ptr(), prev_x, prev_y) };

    let current_weight = clamp_f32(0.14 + reactive * 0.86 + edge * 0.48, 0.10, 1.0);
    let mut rgb = mix_color(history, current, current_weight);

    let debug = mode & 15;
    if debug == 1 {
        let mr = clamp_i32((128.0 + motion_x * 2.0) as i32, 0, 255) as u32;
        let mg = clamp_i32((128.0 + motion_y * 2.0) as i32, 0, 255) as u32;
        let mb = clamp_i32(
            (32.0 + (abs_f32(motion_x) + abs_f32(motion_y)) * 3.0) as i32,
            0,
            255,
        ) as u32;
        rgb = (mr << 16) | (mg << 8) | mb;
    } else if debug == 2 {
        let v = clamp_i32((reactive * 255.0) as i32, 0, 255) as u32;
        rgb = (v << 16) | ((v >> 1) << 8) | (255 - v);
    } else if debug == 3 {
        let near = clamp_i32(((1.0 - nearest_depth) * 255.0) as i32, 0, 255) as u32;
        let e = clamp_i32((edge * 255.0) as i32, 0, 255) as u32;
        rgb = (e << 16) | (near << 8) | (255 - near);
    } else if debug == 4 {
        let h = clamp_i32(((1.0 - current_weight) * 255.0) as i32, 0, 255) as u32;
        let c = clamp_i32((current_weight * 255.0) as i32, 0, 255) as u32;
        rgb = (c << 16) | (h << 8) | 48;
    }

    unsafe {
        frame.write_unchecked(i, rgb);
        history_out.write_unchecked(i, rgb);
    }
}

// rocm-oxide: len(frame)=pixel_count
#[kernel]
pub unsafe extern "C" fn bvh_raytrace(
    frame: gpu::DeviceSliceMut<u32>,
    scene: gpu::DeviceSlice<f32>,
    pixel_count: usize,
    mode: u32,
) {
    let i = gpu::global_id_x();
    let _ = pixel_count;
    if i >= frame.len() {
        return;
    }

    let x = (i as u32) & 1023;
    let y = (i as u32) >> 10;
    let px = (x as f32 - 512.0) * (1.0 / 512.0);
    let py = (288.0 - y as f32) * (1.0 / 512.0);
    let ox = 0.0;
    let oy = 0.2;
    let oz = -5.0;
    let mut dx = px;
    let mut dy = py;
    let mut dz = 1.35;
    let inv_len = inv_sqrt(dx * dx + dy * dy + dz * dz);
    dx *= inv_len;
    dy *= inv_len;
    dz *= inv_len;

    let scene_ptr = scene.as_ptr();
    let sphere_count = unsafe { scene.read_unchecked(0) } as u32;
    let node_count = unsafe { scene.read_unchecked(1) } as u32;
    let node_offset = unsafe { scene.read_unchecked(2) } as u32;

    let (hit_t, hit_index) = if (mode & 1) == 0 {
        trace_spheres_brute(scene_ptr, sphere_count, ox, oy, oz, dx, dy, dz)
    } else {
        trace_spheres_bvh(
            scene_ptr,
            sphere_count,
            node_count,
            node_offset,
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
        )
    };

    let rgb = if hit_index >= 0 {
        shade_scene_sphere(scene_ptr, hit_index as u32, ox, oy, oz, dx, dy, dz, hit_t)
    } else if dy < -0.0001 {
        let t = (-1.35 - oy) / dy;
        let gx = ox + dx * t;
        let gz = oz + dz * t;
        let checker = (((gx * 1.4) as i32) ^ ((gz * 1.4) as i32)) & 1;
        if checker == 0 { 0x2d3440 } else { 0x1b2028 }
    } else {
        sky_color(dy)
    };

    unsafe { frame.write_unchecked(i, rgb) };
}

fn trace_spheres_brute(
    scene: *const f32,
    sphere_count: u32,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
) -> (f32, i32) {
    let mut best_t = 10_000.0;
    let mut best_i = -1;
    let mut i = 0u32;
    while i < sphere_count {
        let t = unsafe { packed_sphere_hit(scene, i, ox, oy, oz, dx, dy, dz) };
        if t > 0.03 && t < best_t {
            best_t = t;
            best_i = i as i32;
        }
        i = i.wrapping_add(1);
    }
    (best_t, best_i)
}

#[allow(clippy::too_many_arguments)]
fn trace_spheres_bvh(
    scene: *const f32,
    sphere_count: u32,
    node_count: u32,
    node_offset: u32,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
) -> (f32, i32) {
    let mut stack = [0u32; 64];
    let mut stack_len = 1u32;
    let mut best_t = 10_000.0;
    let mut best_i = -1;
    unsafe {
        *stack.as_mut_ptr() = 0;
    }

    while stack_len > 0 {
        stack_len = stack_len.wrapping_sub(1);
        let node = unsafe { *stack.as_ptr().add(stack_len as usize) };
        if node >= node_count {
            continue;
        }
        let node_base = node_offset.wrapping_add(node.wrapping_mul(8));
        let box_t = unsafe { packed_aabb_hit(scene, node_base, ox, oy, oz, dx, dy, dz, best_t) };
        if box_t >= best_t {
            continue;
        }

        let left = unsafe { *scene.add((node_base + 6) as usize) } as u32;
        let right_or_count = unsafe { *scene.add((node_base + 7) as usize) };
        if right_or_count < 0.0 {
            let count = (-right_or_count) as u32;
            let mut j = 0u32;
            while j < count {
                let index = left.wrapping_add(j);
                if index < sphere_count {
                    let t = unsafe { packed_sphere_hit(scene, index, ox, oy, oz, dx, dy, dz) };
                    if t > 0.03 && t < best_t {
                        best_t = t;
                        best_i = index as i32;
                    }
                }
                j = j.wrapping_add(1);
            }
        } else if stack_len + 2 < 64 {
            unsafe {
                *stack.as_mut_ptr().add(stack_len as usize) = right_or_count as u32;
            }
            stack_len = stack_len.wrapping_add(1);
            unsafe {
                *stack.as_mut_ptr().add(stack_len as usize) = left;
            }
            stack_len = stack_len.wrapping_add(1);
        }
    }
    (best_t, best_i)
}

unsafe fn packed_sphere_hit(
    scene: *const f32,
    index: u32,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
) -> f32 {
    let base = 8u32.wrapping_add(index.wrapping_mul(8));
    let cx = unsafe { *scene.add(base as usize) };
    let cy = unsafe { *scene.add((base + 1) as usize) };
    let cz = unsafe { *scene.add((base + 2) as usize) };
    let radius = unsafe { *scene.add((base + 3) as usize) };
    let ocx = ox - cx;
    let ocy = oy - cy;
    let ocz = oz - cz;
    let b = ocx * dx + ocy * dy + ocz * dz;
    let c = ocx * ocx + ocy * ocy + ocz * ocz - radius * radius;
    let h = b * b - c;
    if h <= 0.0 {
        10_000.0
    } else {
        -b - h * inv_sqrt(h)
    }
}

unsafe fn packed_aabb_hit(
    scene: *const f32,
    base: u32,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    best_t: f32,
) -> f32 {
    let min_x = unsafe { *scene.add(base as usize) };
    let min_y = unsafe { *scene.add((base + 1) as usize) };
    let min_z = unsafe { *scene.add((base + 2) as usize) };
    let max_x = unsafe { *scene.add((base + 3) as usize) };
    let max_y = unsafe { *scene.add((base + 4) as usize) };
    let max_z = unsafe { *scene.add((base + 5) as usize) };
    let inv_x = 1.0 / dx;
    let inv_y = 1.0 / dy;
    let inv_z = 1.0 / dz;
    let tx0 = (min_x - ox) * inv_x;
    let tx1 = (max_x - ox) * inv_x;
    let ty0 = (min_y - oy) * inv_y;
    let ty1 = (max_y - oy) * inv_y;
    let tz0 = (min_z - oz) * inv_z;
    let tz1 = (max_z - oz) * inv_z;
    let t_near = max_f32(
        max_f32(min_f32(tx0, tx1), min_f32(ty0, ty1)),
        min_f32(tz0, tz1),
    );
    let t_far = min_f32(
        min_f32(max_f32(tx0, tx1), max_f32(ty0, ty1)),
        max_f32(tz0, tz1),
    );
    if t_far < max_f32(t_near, 0.03) || t_near > best_t {
        10_000.0
    } else {
        t_near
    }
}

unsafe fn shade_scene_sphere(
    scene: *const f32,
    index: u32,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    hit_t: f32,
) -> u32 {
    let base = 8u32.wrapping_add(index.wrapping_mul(8));
    let cx = unsafe { *scene.add(base as usize) };
    let cy = unsafe { *scene.add((base + 1) as usize) };
    let cz = unsafe { *scene.add((base + 2) as usize) };
    let radius = unsafe { *scene.add((base + 3) as usize) };
    let r = unsafe { *scene.add((base + 4) as usize) };
    let g = unsafe { *scene.add((base + 5) as usize) };
    let b = unsafe { *scene.add((base + 6) as usize) };
    let px = ox + dx * hit_t;
    let py = oy + dy * hit_t;
    let pz = oz + dz * hit_t;
    let nx = (px - cx) / radius;
    let ny = (py - cy) / radius;
    let nz = (pz - cz) / radius;
    let light = max_f32(0.0, nx * -0.48 + ny * 0.82 + nz * -0.31);
    let view_rim = pow2(max_f32(0.0, 1.0 + nx * dx + ny * dy + nz * dz)) * 0.18;
    pack_rgbf(
        r * (0.18 + light * 0.92) + view_rim,
        g * (0.18 + light * 0.92) + view_rim,
        b * (0.18 + light * 0.92) + view_rim,
    )
}

fn scene_hit(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    frame_index: u32,
    include_plane: u32,
) -> (f32, f32, f32, f32, f32, f32, f32, u32) {
    let mut best_t = 10_000.0;
    let mut best_nx = 0.0;
    let mut best_ny = 1.0;
    let mut best_nz = 0.0;
    let mut best_r = 0.7;
    let mut best_g = 0.7;
    let mut best_b = 0.7;
    let mut material = 0;

    let anim = ((frame_index & 255) as f32) * (1.0 / 255.0);
    let moving_x = -0.4 + (anim - 0.5) * 2.4;

    let (t0, nx0, ny0, nz0) = sphere_hit(ox, oy, oz, dx, dy, dz, -1.65, -0.05, 3.2, 0.58);
    if t0 > 0.03 && t0 < best_t {
        best_t = t0;
        best_nx = nx0;
        best_ny = ny0;
        best_nz = nz0;
        best_r = 0.25;
        best_g = 0.86;
        best_b = 1.0;
        material = 2;
    }

    let (t1, nx1, ny1, nz1) = sphere_hit(ox, oy, oz, dx, dy, dz, moving_x, 0.0, 4.9, 0.72);
    if t1 > 0.03 && t1 < best_t {
        best_t = t1;
        best_nx = nx1;
        best_ny = ny1;
        best_nz = nz1;
        best_r = 1.0;
        best_g = 0.25;
        best_b = 0.2;
        material = 1;
    }

    let (t2, nx2, ny2, nz2) = sphere_hit(ox, oy, oz, dx, dy, dz, 1.25, -0.18, 2.25, 0.42);
    if t2 > 0.03 && t2 < best_t {
        best_t = t2;
        best_nx = nx2;
        best_ny = ny2;
        best_nz = nz2;
        best_r = 1.0;
        best_g = 0.86;
        best_b = 0.25;
        material = 1;
    }

    let (bt0, bnx0, bny0, bnz0) =
        box_hit(ox, oy, oz, dx, dy, dz, 1.65, -0.65, 3.45, 2.55, 0.35, 4.2);
    if bt0 > 0.03 && bt0 < best_t {
        best_t = bt0;
        best_nx = bnx0;
        best_ny = bny0;
        best_nz = bnz0;
        best_r = 0.26;
        best_g = 0.9;
        best_b = 0.42;
        material = 1;
    }

    let (bt1, bnx1, bny1, bnz1) = box_hit(
        ox, oy, oz, dx, dy, dz, -2.75, -0.65, 5.05, -1.95, 1.25, 5.85,
    );
    if bt1 > 0.03 && bt1 < best_t {
        best_t = bt1;
        best_nx = bnx1;
        best_ny = bny1;
        best_nz = bnz1;
        best_r = 0.82;
        best_g = 0.28;
        best_b = 1.0;
        material = 1;
    }

    let (bt2, bnx2, bny2, bnz2) =
        box_hit(ox, oy, oz, dx, dy, dz, -0.45, -0.65, 6.8, 0.45, 1.65, 7.7);
    if bt2 > 0.03 && bt2 < best_t {
        best_t = bt2;
        best_nx = bnx2;
        best_ny = bny2;
        best_nz = bnz2;
        best_r = 0.18;
        best_g = 0.5;
        best_b = 1.0;
        material = 2;
    }

    if include_plane != 0 && dy < -0.0001 {
        let pt = (-0.68 - oy) / dy;
        if pt > 0.03 && pt < best_t {
            let px = ox + dx * pt;
            let pz = oz + dz * pt;
            best_t = pt;
            best_nx = 0.0;
            best_ny = 1.0;
            best_nz = 0.0;
            let check = (((px * 1.2) as i32) ^ ((pz * 1.2) as i32)) & 1;
            if check == 0 {
                best_r = 0.62;
                best_g = 0.62;
                best_b = 0.58;
            } else {
                best_r = 0.16;
                best_g = 0.18;
                best_b = 0.2;
            }
            material = 0;
        }
    }

    (
        best_t, best_nx, best_ny, best_nz, best_r, best_g, best_b, material,
    )
}

fn shade_hit(
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    r: f32,
    g: f32,
    b: f32,
    frame_index: u32,
    flags: u32,
) -> u32 {
    let lx = -0.48;
    let ly = 0.78;
    let lz = -0.4;
    let ndotl = max_f32(0.0, nx * lx + ny * ly + nz * lz);
    let mut light = 0.18 + ndotl * 0.9;

    if (flags & 1) != 0 {
        let (st, _, _, _, _, _, _, _) = scene_hit(
            px + nx * 0.045,
            py + ny * 0.045,
            pz + nz * 0.045,
            lx,
            ly,
            lz,
            frame_index,
            0,
        );
        if st < 16.0 {
            light *= 0.35;
        }
    }

    let rim = pow2(max_f32(0.0, 1.0 - ny)) * 0.08;
    pack_rgbf(r * light + rim, g * light + rim, b * light + rim)
}

fn sky_color(dy: f32) -> u32 {
    let t = clamp_f32(dy * 0.5 + 0.55, 0.0, 1.0);
    pack_rgbf(0.05 + t * 0.22, 0.08 + t * 0.34, 0.12 + t * 0.58)
}

fn sphere_hit(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
) -> (f32, f32, f32, f32) {
    let ocx = ox - cx;
    let ocy = oy - cy;
    let ocz = oz - cz;
    let b = ocx * dx + ocy * dy + ocz * dz;
    let c = ocx * ocx + ocy * ocy + ocz * ocz - radius * radius;
    let h = b * b - c;
    if h <= 0.0 {
        return (10_000.0, 0.0, 1.0, 0.0);
    }

    let t = -b - h * inv_sqrt(h);
    let inv_r = 1.0 / radius;
    let px = ox + dx * t;
    let py = oy + dy * t;
    let pz = oz + dz * t;
    ((t), (px - cx) * inv_r, (py - cy) * inv_r, (pz - cz) * inv_r)
}

fn box_hit(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
) -> (f32, f32, f32, f32) {
    let inv_x = 1.0 / dx;
    let inv_y = 1.0 / dy;
    let inv_z = 1.0 / dz;

    let tx0 = (min_x - ox) * inv_x;
    let tx1 = (max_x - ox) * inv_x;
    let ty0 = (min_y - oy) * inv_y;
    let ty1 = (max_y - oy) * inv_y;
    let tz0 = (min_z - oz) * inv_z;
    let tz1 = (max_z - oz) * inv_z;

    let tx_min = min_f32(tx0, tx1);
    let tx_max = max_f32(tx0, tx1);
    let ty_min = min_f32(ty0, ty1);
    let ty_max = max_f32(ty0, ty1);
    let tz_min = min_f32(tz0, tz1);
    let tz_max = max_f32(tz0, tz1);

    let t_near = max_f32(max_f32(tx_min, ty_min), tz_min);
    let t_far = min_f32(min_f32(tx_max, ty_max), tz_max);
    if t_far < max_f32(t_near, 0.03) {
        return (10_000.0, 0.0, 1.0, 0.0);
    }

    let px = ox + dx * t_near;
    let py = oy + dy * t_near;
    let pz = oz + dz * t_near;
    let bias = 0.004;
    if abs_f32(px - min_x) < bias {
        (t_near, -1.0, 0.0, 0.0)
    } else if abs_f32(px - max_x) < bias {
        (t_near, 1.0, 0.0, 0.0)
    } else if abs_f32(py - min_y) < bias {
        (t_near, 0.0, -1.0, 0.0)
    } else if abs_f32(py - max_y) < bias {
        (t_near, 0.0, 1.0, 0.0)
    } else if abs_f32(pz - min_z) < bias {
        (t_near, 0.0, 0.0, -1.0)
    } else {
        (t_near, 0.0, 0.0, 1.0)
    }
}

fn classify_control(value: u32) -> ControlKind {
    match value & 3 {
        0 => ControlKind::Zero,
        1 => ControlKind::One,
        2 => ControlKind::Many,
        _ => ControlKind::Custom,
    }
}

fn control_option(value: u32) -> Option<u32> {
    if (value & 1) == 0 {
        Some(value / 2 + 3)
    } else {
        None
    }
}

fn control_result(value: u32) -> Result<u32, u32> {
    if value < 12 {
        Ok(value.wrapping_mul(3).wrapping_add(1))
    } else {
        Err(value - 12)
    }
}

#[inline(always)]
fn apply_device_closure<F>(value: u32, f: F) -> u32
where
    F: FnOnce(u32) -> u32,
{
    f(value)
}

fn control_pair(value: u32, params: ControlParams) -> ControlPair {
    let scale = abs_i32(params.scale) as u32;
    let kind = classify_control(value) as u32;
    ControlPair {
        left: value.wrapping_add(params.seed),
        right: kind.wrapping_add(scale),
    }
}

fn control_score(value: u32, params: ControlParams, pair: ControlPair) -> u32 {
    let scale = abs_i32(params.scale) as u32;
    let kind_score = match classify_control(value) {
        ControlKind::Zero => 17u32,
        ControlKind::One => 31u32,
        ControlKind::Many => 47u32,
        ControlKind::Custom => 61u32,
    };
    let option_score = match control_option(value) {
        Some(inner) => inner.wrapping_mul(5),
        None => 23,
    };
    let result_score = match control_result(value) {
        Ok(ok) => ok,
        Err(err) => err.wrapping_add(101),
    };
    let fixed = [value, value.wrapping_add(1), params.seed, scale];
    let runtime_index = (value as usize) & 3;
    let mut mutable = [0u32; 4];
    mutable[0] = fixed[runtime_index];
    mutable[1] = fixed[0].wrapping_add(fixed[1]);
    mutable[2] = pair.left;
    mutable[3] = pair.right;

    let mut array_score = 0u32;
    for item in fixed {
        array_score = array_score.wrapping_add(item & 15);
    }
    for j in 0..4 {
        if j == runtime_index {
            continue;
        }
        array_score = array_score.wrapping_add(mutable[j]);
    }

    let mut loop_score = 0u32;
    let mut countdown = value & 3;
    while countdown > 0 {
        loop_score = loop_score.wrapping_add(countdown);
        countdown -= 1;
    }
    let mut step = 0u32;
    loop {
        if step >= 3 {
            break;
        }
        step = step.wrapping_add(1);
        if step == 2 {
            continue;
        }
        loop_score = loop_score.wrapping_add(step);
    }

    let signed = params.scale.wrapping_add(value as i32);
    let float_score = ((signed as f32) * 0.5 + 2.0) as u32;
    let bitcast_score = ((float_score as f32).to_bits() >> 20) & 31;

    kind_score
        .wrapping_add(option_score)
        .wrapping_add(result_score)
        .wrapping_add(array_score)
        .wrapping_add(loop_score)
        .wrapping_add(float_score)
        .wrapping_add(bitcast_score)
        .wrapping_add((pair.left ^ pair.right) & 31)
}

fn return_rust_pair(value: u32, params: ControlParams) -> ReturnRustPair {
    let scale = abs_i32(params.scale) as u32;
    let rotation = (value & 7).wrapping_add(1);
    let left = value.wrapping_add(params.seed).rotate_left(rotation);
    let right = ((value as u64) << 32)
        .wrapping_add(scale as u64)
        .wrapping_add((params.seed as u64).wrapping_mul(17));
    ReturnRustPair { left, right }
}

fn return_packet(value: u32, pair: ReturnRustPair) -> ReturnPacket {
    let shift = (value & 3) * 8;
    let lane_mix = (pair.right >> shift) as u32;
    ReturnPacket {
        sum: pair.right.wrapping_add(pair.left as u64),
        folded: pair.left ^ lane_mix.rotate_right(value & 15),
        tag: 0xc0de_0000u32 ^ (value & 0xff) ^ ((pair.right >> 48) as u32),
    }
}

fn return_packet_score(packet: ReturnPacket) -> u64 {
    packet.sum ^ ((packet.folded as u64) << 16) ^ packet.tag as u64
}

fn read_input_by_byte_cast(input: gpu::DeviceSlice<u32>, index: usize) -> u32 {
    let byte_offset = index * core::mem::size_of::<u32>();
    let slot = unsafe { input.as_ptr().cast::<u8>().add(byte_offset).cast::<u32>() };
    unsafe { core::ptr::read(slot) }
}

fn flow_cast_score(input: gpu::DeviceSlice<u32>, len: usize, index: usize) -> u32 {
    let value = read_input_by_byte_cast(input, index);
    let mut score = 0u32;

    if (value & 1) == 0 {
        if (value & 4) == 0 {
            score = score.wrapping_add(11);
        } else {
            score = score.wrapping_add(17);
        }
    } else if value % 3 == 0 {
        score = score.wrapping_add(23);
    } else if value > 10 {
        score = score.wrapping_add(31);
    } else {
        score = score.wrapping_add(37);
    }

    for outer in 0u32..3 {
        for inner in 0u32..4 {
            let candidate = value.wrapping_add(outer * 7 + inner);
            if (candidate & 1) == 0 {
                continue;
            }
            if inner == 3 && (value & 2) != 0 {
                break;
            }
            score = score.wrapping_add(candidate & 15);
        }
    }

    for step in 0..((value & 3) + 1) {
        score = score.wrapping_add((step + 1) * 3);
    }

    let mut offset = 0usize;
    while offset < 4 && index + offset < len {
        let lane = read_input_by_byte_cast(input, index + offset);
        score = score.wrapping_add((lane & 7).wrapping_mul(offset as u32 + 1));
        offset += 1;
    }

    let signed = (score as i32).wrapping_sub(value as i32);
    score.wrapping_add((signed as u32) & 31)
}

fn cast_packet(value: u32, index: usize) -> CastPacket {
    let signed = (value as i64).wrapping_sub(0x1234).wrapping_mul(-33);
    let wide = ((value as u64) << 37)
        .wrapping_add((index as u64).wrapping_mul(0x1f1f_0101))
        .rotate_left(value & 31);
    let float_value = (signed as f32) * 0.125 + index as f32;
    let double_value = f64::from_bits(0x3ff0_0000_0000_0000u64 | (wide & 0x000f_ffff_ffff_ffff));
    let narrow = (wide as u32)
        .wrapping_add(signed as u32)
        .wrapping_add(float_value as i32 as u32);
    let float_bits = float_value.to_bits() ^ (double_value.to_bits() as u32).rotate_left(11);
    CastPacket {
        wide,
        signed_bits: signed as u64,
        float_bits,
        narrow,
    }
}

fn cast_packet_score(packet: CastPacket) -> u64 {
    packet
        .wide
        .wrapping_add(packet.signed_bits.rotate_left(7))
        .wrapping_add((packet.float_bits as u64) << 1)
        .wrapping_add(packet.narrow as u64)
}

fn abs_i32(value: i32) -> i32 {
    if value < 0 { -value } else { value }
}

fn abs_f32(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}

fn min_f32(a: f32, b: f32) -> f32 {
    gpu::math::min_f32(a, b)
}

fn max_f32(a: f32, b: f32) -> f32 {
    gpu::math::max_f32(a, b)
}

fn clamp_f32(value: f32, lo: f32, hi: f32) -> f32 {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}

fn inv_sqrt(value: f32) -> f32 {
    gpu::math::rsqrt_f32(value)
}

fn pow2(value: f32) -> f32 {
    value * value
}

fn clamp_u32(value: u32, lo: u32, hi: u32) -> u32 {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}

fn clamp_i32(value: i32, lo: i32, hi: i32) -> i32 {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}

fn min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

fn min_usize(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}

fn avg4(a: u32, b: u32, c: u32, d: u32) -> i32 {
    (((a & 255) + (b & 255) + (c & 255) + (d & 255)) >> 2) as i32
}

unsafe fn bilinear_input(
    input: *const u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    fx: f32,
    fy: f32,
) -> u32 {
    let c00 = unsafe { sample_input(input, x0, y0) };
    let c10 = unsafe { sample_input(input, x1, y0) };
    let c01 = unsafe { sample_input(input, x0, y1) };
    let c11 = unsafe { sample_input(input, x1, y1) };
    let r = bilinear_channel(c00 >> 16, c10 >> 16, c01 >> 16, c11 >> 16, fx, fy);
    let g = bilinear_channel(c00 >> 8, c10 >> 8, c01 >> 8, c11 >> 8, fx, fy);
    let b = bilinear_channel(c00, c10, c01, c11, fx, fy);
    (r << 16) | (g << 8) | b
}

unsafe fn bilinear_color_512(
    input: *const u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    fx: f32,
    fy: f32,
) -> u32 {
    let c00 = unsafe { sample_color_512(input, x0, y0) };
    let c10 = unsafe { sample_color_512(input, x1, y0) };
    let c01 = unsafe { sample_color_512(input, x0, y1) };
    let c11 = unsafe { sample_color_512(input, x1, y1) };
    let r = bilinear_channel(c00 >> 16, c10 >> 16, c01 >> 16, c11 >> 16, fx, fy);
    let g = bilinear_channel(c00 >> 8, c10 >> 8, c01 >> 8, c11 >> 8, fx, fy);
    let b = bilinear_channel(c00, c10, c01, c11, fx, fy);
    (r << 16) | (g << 8) | b
}

unsafe fn bilinear_history_1024(input: *const u32, x: f32, y: f32) -> u32 {
    let x = clamp_f32(x, 0.0, 1023.0);
    let y = clamp_f32(y, 0.0, 575.0);
    let x0 = x as u32;
    let y0 = y as u32;
    let x1 = min_u32(x0 + 1, 1023);
    let y1 = min_u32(y0 + 1, 575);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let c00 = unsafe { sample_history_1024(input, x0, y0) };
    let c10 = unsafe { sample_history_1024(input, x1, y0) };
    let c01 = unsafe { sample_history_1024(input, x0, y1) };
    let c11 = unsafe { sample_history_1024(input, x1, y1) };
    let r = bilinear_channel(c00 >> 16, c10 >> 16, c01 >> 16, c11 >> 16, fx, fy);
    let g = bilinear_channel(c00 >> 8, c10 >> 8, c01 >> 8, c11 >> 8, fx, fy);
    let b = bilinear_channel(c00, c10, c01, c11, fx, fy);
    (r << 16) | (g << 8) | b
}

fn bilinear_channel(c00: u32, c10: u32, c01: u32, c11: u32, fx: f32, fy: f32) -> u32 {
    let v00 = (c00 & 255) as f32;
    let v10 = (c10 & 255) as f32;
    let v01 = (c01 & 255) as f32;
    let v11 = (c11 & 255) as f32;
    let top = v00 + (v10 - v00) * fx;
    let bottom = v01 + (v11 - v01) * fx;
    clamp_i32((top + (bottom - top) * fy) as i32, 0, 255) as u32
}

unsafe fn sample_input(input: *const u32, x: u32, y: u32) -> u32 {
    unsafe { *input.add((min_u32(y, 575) << 10).wrapping_add(min_u32(x, 1023)) as usize) }
}

unsafe fn sample_color_512(input: *const u32, x: u32, y: u32) -> u32 {
    unsafe { *input.add((min_u32(y, 287) << 9).wrapping_add(min_u32(x, 511)) as usize) }
}

unsafe fn sample_depth_512(input: *const f32, x: u32, y: u32) -> f32 {
    unsafe { *input.add((min_u32(y, 287) << 9).wrapping_add(min_u32(x, 511)) as usize) }
}

unsafe fn sample_aux_512(input: *const f32, x: u32, y: u32, channel: u32) -> f32 {
    unsafe {
        *input.add(
            (((min_u32(y, 287) << 9).wrapping_add(min_u32(x, 511))) * 3 + min_u32(channel, 2))
                as usize,
        )
    }
}

unsafe fn sample_history_1024(input: *const u32, x: u32, y: u32) -> u32 {
    unsafe { *input.add((min_u32(y, 575) << 10).wrapping_add(min_u32(x, 1023)) as usize) }
}

unsafe fn sample_frame(input: *const u32, width: u32, height: u32, x: u32, y: u32) -> u32 {
    let sx = min_u32(x, width.saturating_sub(1));
    let sy = min_u32(y, height.saturating_sub(1));
    unsafe { *input.add(sy.wrapping_mul(width).wrapping_add(sx) as usize) }
}

fn luminance(rgb: u32) -> i32 {
    let r = ((rgb >> 16) & 255) as i32;
    let g = ((rgb >> 8) & 255) as i32;
    let b = (rgb & 255) as i32;
    (r * 54 + g * 183 + b * 19) >> 8
}

fn sharpen_rgb(base: u32, reference: u32, sharpness: i32) -> u32 {
    let amount = 2 + sharpness;
    let br = ((base >> 16) & 255) as i32;
    let bg = ((base >> 8) & 255) as i32;
    let bb = (base & 255) as i32;
    let rr = ((reference >> 16) & 255) as i32;
    let rg = ((reference >> 8) & 255) as i32;
    let rb = (reference & 255) as i32;
    let r = clamp_i32(br + ((br - rr) * amount) / 16, 0, 255) as u32;
    let g = clamp_i32(bg + ((bg - rg) * amount) / 16, 0, 255) as u32;
    let b = clamp_i32(bb + ((bb - rb) * amount) / 16, 0, 255) as u32;
    (r << 16) | (g << 8) | b
}

fn mix_color(a: u32, b: u32, t: f32) -> u32 {
    let ar = ((a >> 16) & 255) as f32;
    let ag = ((a >> 8) & 255) as f32;
    let ab = (a & 255) as f32;
    let br = ((b >> 16) & 255) as f32;
    let bg = ((b >> 8) & 255) as f32;
    let bb = (b & 255) as f32;
    let s = 1.0 - t;
    (((ar * s + br * t) as u32) << 16)
        | (((ag * s + bg * t) as u32) << 8)
        | ((ab * s + bb * t) as u32)
}

fn pack_rgbf(r: f32, g: f32, b: f32) -> u32 {
    let ri = (clamp_f32(r, 0.0, 1.0) * 255.0) as u32;
    let gi = (clamp_f32(g, 0.0, 1.0) * 255.0) as u32;
    let bi = (clamp_f32(b, 0.0, 1.0) * 255.0) as u32;
    (ri << 16) | (gi << 8) | bi
}

fn wheel(hue: u32, value: u32) -> u32 {
    let r = (tri(hue) * value) >> 8;
    let g = (tri(hue.wrapping_add(85)) * value) >> 8;
    let b = (tri(hue.wrapping_add(170)) * value) >> 8;

    (r << 16) | (g << 8) | b
}

fn tri(phase: u32) -> u32 {
    let x = (phase & 255) as i32 - 128;
    let v = 255 - clamp_u32((abs_i32(x) as u32) << 1, 0, 255);
    v
}

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
