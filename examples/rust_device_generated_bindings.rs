use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;

    let n = 1 << 20;
    let block_x = 256u32;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (n - i) as f32).collect::<Vec<_>>();

    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(n)?;

    let short = DeviceBuffer::from_slice(&a[..n / 2])?;
    let validation = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems(n, block_x),
            &d_out,
            &short,
            &d_b,
            n,
            block_x,
        )
    };
    match validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected short buffer: {message}");
        }
        Err(err) => return Err(format!("unexpected validation error: {err}").into()),
        Ok(()) => return Err("short buffer launch unexpectedly succeeded".into()),
    }

    let block_validation = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems(n, block_x),
            &d_out,
            &d_a,
            &d_b,
            n,
            block_x / 2,
        )
    };
    match block_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected block_x mismatch: {message}");
        }
        Err(err) => return Err(format!("unexpected block_x validation error: {err}").into()),
        Ok(()) => return Err("block_x mismatch launch unexpectedly succeeded".into()),
    }

    let small_frame = DeviceBuffer::<u32>::new(512)?;
    let small_color = DeviceBuffer::<u32>::new(127)?;
    let small_depth = DeviceBuffer::<f32>::new(128)?;
    let contract_validation = unsafe {
        kernels.depth_aware_upscale(
            LaunchConfig::for_num_elems(512, block_x),
            &small_frame,
            &small_color,
            &small_depth,
            512,
            0,
            block_x,
        )
    };
    match contract_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected explicit buffer contract: {message}");
        }
        Err(err) => return Err(format!("unexpected contract validation error: {err}").into()),
        Ok(()) => return Err("explicit contract violation unexpectedly succeeded".into()),
    }

    unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems(n, block_x),
            &d_out,
            &d_a,
            &d_b,
            n,
            block_x,
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

    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 2.0,
        bias: 3.0,
    }])?;
    unsafe {
        kernels.affine_transform(
            LaunchConfig::for_num_elems(n, block_x),
            &d_out,
            &d_a,
            &params,
            n,
            block_x,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let affine = d_out.copy_to_vec()?;
    assert_eq!(affine[17], a[17] * 2.0 + 3.0);

    println!("Generated binding vector_add passed on {}", device.arch());
    Ok(())
}
