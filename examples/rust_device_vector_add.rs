use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let module = device.load_code_object_file(env!("ROCM_OXIDE_DEVICE_HSACO"))?;
    let kernel = module.kernel(c"vector_add")?;

    let n = 1 << 20;
    let block_x = 256u32;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (n - i) as f32).collect::<Vec<_>>();

    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(n)?;

    unsafe {
        rocm_oxide::launch!(
            kernel,
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            d_out.as_mut_ptr(),
            d_out.len(),
            d_a.as_ptr(),
            d_a.len(),
            d_b.as_ptr(),
            d_b.len()
        )?;
    }
    rocm_oxide::hip::synchronize()?;

    let out = d_out.copy_to_vec()?;
    for (index, ((got, lhs), rhs)) in out.iter().zip(&a).zip(&b).enumerate() {
        let expected = lhs + rhs;
        if (*got - expected).abs() > f32::EPSILON {
            return Err(format!("mismatch at {index}: got {got}, expected {expected}").into());
        }
    }

    println!(
        "Rust-authored vector_add kernel passed on {}",
        device.arch()
    );
    Ok(())
}
