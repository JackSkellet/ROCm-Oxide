use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

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
    println!("ROCm-Oxide: compiling HIPRTC kernel for {}", device.arch());

    let module = device.compile_hip_source(KERNEL)?;
    let kernel = module.kernel(c"vector_add")?;

    let n = 1 << 20;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (n - i) as f32).collect::<Vec<_>>();

    let d_a = DeviceBuffer::<f32>::from_slice(&a)?;
    let d_b = DeviceBuffer::<f32>::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(n)?;

    let config = LaunchConfig::for_num_elems(n);
    unsafe {
        rocm_oxide::launch!(
            kernel,
            config,
            d_out.as_mut_ptr(),
            d_a.as_ptr(),
            d_b.as_ptr(),
            n as u64,
        )?;
    }

    rocm_oxide::hip::synchronize()?;
    let out = d_out.copy_to_vec()?;

    for (idx, ((got, lhs), rhs)) in out.iter().zip(&a).zip(&b).enumerate() {
        let expected = lhs + rhs;
        if (*got - expected).abs() > f32::EPSILON {
            return Err(format!("mismatch at {idx}: got {got}, expected {expected}").into());
        }
    }

    println!("OK: vector_add verified for {n} elements");
    Ok(())
}
