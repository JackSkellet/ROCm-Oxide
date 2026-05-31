use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

const KERNEL: &str = r#"
extern "C" {
__device__ float scale_factor = 1.0f;
}

extern "C" __global__
void scale_vec(float* out, const float* input, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] * scale_factor;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let module = device.compile_hip_source(KERNEL)?;
    let kernel = module.kernel(c"scale_vec")?;
    let scale = module.global::<f32>(c"scale_factor")?;
    scale.set(2.5)?;

    let n = 65_536;
    let input = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let d_input = DeviceBuffer::from_slice(&input)?;
    let d_out = DeviceBuffer::<f32>::new(n)?;

    unsafe {
        rocm_oxide::launch!(
            kernel,
            LaunchConfig::for_num_elems(n, 256),
            d_out.as_mut_ptr(),
            d_input.as_ptr(),
            n as u64,
        )?;
    }
    rocm_oxide::hip::synchronize()?;

    let out = d_out.copy_to_vec()?;
    assert_eq!(out[123], input[123] * 2.5);
    assert_eq!(scale.copy_to_vec()?, vec![2.5]);

    println!(
        "module global `{}` ({} bytes) scale_vec passed on {}",
        scale.name(),
        scale.bytes(),
        device.arch()
    );
    Ok(())
}
