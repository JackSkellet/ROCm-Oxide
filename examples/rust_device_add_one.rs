use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let module = device.load_code_object_file(env!("ROCM_OXIDE_DEVICE_HSACO"))?;
    let kernel = module.kernel(c"add_one")?;

    let n = 256usize;
    let input = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let d_input = DeviceBuffer::from_slice(&input)?;
    let d_output = DeviceBuffer::<f32>::new(n)?;

    unsafe {
        rocm_oxide::launch!(
            kernel,
            LaunchConfig::for_num_elems(n),
            d_output.as_mut_ptr(),
            d_output.len(),
            d_input.as_ptr(),
            d_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;

    let output = d_output.copy_to_vec()?;
    for (index, (got, input)) in output.iter().zip(&input).enumerate() {
        let expected = input + 1.0;
        if (*got - expected).abs() > f32::EPSILON {
            return Err(format!("mismatch at {index}: got {got}, expected {expected}").into());
        }
    }

    println!("Rust-authored AMDGPU kernel passed on {}", device.arch());
    Ok(())
}
