use rocm_oxide_kernel::{constant, device_global, kernel, shared};

#[kernel]
pub unsafe fn plain_kernel(_out: *mut f32, _len: usize) {}

#[kernel(monomorphize(f32), monomorphize(u32))]
unsafe fn typed_kernel<T>(_out: *mut T, _len: usize) {}

#[device_global]
pub static mut DEVICE_GLOBAL_VALUE: u32 = 0;

#[constant]
pub static CONSTANT_VALUE: u32 = 7;

#[shared]
pub static mut SHARED_VALUE: [u32; 4] = [0; 4];

#[test]
fn attribute_macros_compile() {}
