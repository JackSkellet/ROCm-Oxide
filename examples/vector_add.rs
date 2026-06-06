use rocm_oxide::{Device, DeviceBuffer};

const KERNEL: &str = r#"
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long n) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let module = device.compile_hip_source(KERNEL)?;
    let kernel = module.kernel(c"vector_add")?;

    let n = 65_536;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (i * 2) as f32).collect::<Vec<_>>();

    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(n)?;

    unsafe {
        rocm_oxide::launch_1d!(
            kernel,
            n,
            d_out.as_mut_ptr(),
            d_a.as_ptr(),
            d_b.as_ptr(),
            n as u64,
        )?;
    }
    rocm_oxide::hip::synchronize()?;

    let out = d_out.copy_to_vec()?;
    assert!(
        out.iter()
            .zip(a.iter().zip(&b))
            .all(|(got, (lhs, rhs))| (*got - (lhs + rhs)).abs() <= f32::EPSILON)
    );

    println!("vector_add example passed on {}", device.arch());
    Ok(())
}
