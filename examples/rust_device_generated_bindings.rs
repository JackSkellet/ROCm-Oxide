use rocm_oxide::{Device, DeviceBuffer, DeviceOperation, Dim3, LaunchConfig, StreamPool};
use std::sync::Arc;

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

    let delta = kernels.global_add_one_delta()?;
    assert_eq!(delta.copy_to_vec()?, vec![1.0]);
    delta.set(2.0)?;
    let add_input = DeviceBuffer::from_slice(&[1.0f32, 5.5, -3.0, 0.25])?;
    let add_out = DeviceBuffer::<f32>::new(add_input.len())?;
    unsafe {
        kernels.add_one(
            LaunchConfig::for_num_elems_with_block_size(add_input.len(), block_x),
            &add_out,
            &add_input,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(add_out.copy_to_vec()?, vec![3.0, 7.5, -1.0, 2.25]);

    let math_input = DeviceBuffer::from_slice(&[4.0f32, 0.0, 1.0, -1.0])?;
    let math_out = DeviceBuffer::<f32>::new(16)?;
    unsafe {
        kernels.math_intrinsics(LaunchConfig::for_num_elems(1), &math_out, &math_input)?;
    }
    rocm_oxide::hip::synchronize()?;
    let math = math_out.copy_to_vec()?;
    assert_close("sqrt_f32", math[0], 2.0, 0.0001)?;
    assert_close("rsqrt_f32", math[1], 0.5, 0.0001)?;
    assert_close("sin_f32", math[2], 0.0, 0.0001)?;
    assert_close("cos_f32", math[3], 1.0, 0.0001)?;
    assert_close("atan_f32", math[4], std::f32::consts::FRAC_PI_4, 0.002)?;
    assert_close("min_f32", math[5], -2.0, 0.0001)?;
    assert_close("max_f32", math[6], 3.0, 0.0001)?;
    assert_close("sqrt_f64", math[7], 2.0, 0.0001)?;
    assert_close("rsqrt_f64", math[8], 0.5, 0.0001)?;
    assert_close("sin_f64", math[9], 0.0, 0.0001)?;
    assert_close("cos_f64", math[10], 1.0, 0.0001)?;
    assert_close("atan_f64", math[11], std::f32::consts::FRAC_PI_4, 0.002)?;
    assert_eq!(math[12], 1.0, "sqrt_f32(-1) should produce NaN");
    assert_eq!(math[13], 1.0, "min_f32 should propagate NaN");
    assert_eq!(math[14], 1.0, "max_f32 should propagate NaN");
    assert_close("min_f64", math[15], -2.0, 0.0001)?;

    let atomic_scope_out = DeviceBuffer::<u32>::new(4)?;
    let atomic_counters = DeviceBuffer::from_slice(&[0u32; 3])?;
    unsafe {
        kernels.scoped_atomics(
            LaunchConfig::new(Dim3::x(1), Dim3::x(256)),
            &atomic_scope_out,
            &atomic_counters,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(atomic_scope_out.copy_to_vec()?, vec![0, 1, 2, 0]);
    assert_eq!(atomic_counters.copy_to_vec()?, vec![256, 256, 256]);

    let short = DeviceBuffer::from_slice(&a[..n / 2])?;
    let validation = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &short,
            &d_b,
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
            LaunchConfig::new(Dim3::x(1), Dim3::x(0)),
            &d_out,
            &d_a,
            &d_b,
        )
    };
    match block_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected invalid launch shape: {message}");
        }
        Err(err) => return Err(format!("unexpected launch validation error: {err}").into()),
        Ok(()) => return Err("invalid launch unexpectedly succeeded".into()),
    }

    let alias_validation = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &d_out,
            &d_b,
        )
    };
    match alias_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected aliased mutable buffer: {message}");
        }
        Err(err) => return Err(format!("unexpected alias validation error: {err}").into()),
        Ok(()) => return Err("aliased mutable buffer launch unexpectedly succeeded".into()),
    }

    let small_frame = DeviceBuffer::<u32>::new(512)?;
    let small_color = DeviceBuffer::<u32>::new(127)?;
    let small_depth = DeviceBuffer::<f32>::new(128)?;
    let contract_validation = unsafe {
        kernels.depth_aware_upscale(
            LaunchConfig::for_num_elems_with_block_size(512, block_x),
            &small_frame,
            &small_color,
            &small_depth,
            512,
            0,
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
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &d_a,
            &d_b,
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

    let pool = StreamPool::new(&device, 2)?;
    let lazy_a = Arc::new(DeviceBuffer::from_slice(&a)?);
    let lazy_b = Arc::new(DeviceBuffer::from_slice(&b)?);
    let lazy_out = Arc::new(DeviceBuffer::<f32>::new(n)?);
    let lazy_completion = unsafe {
        kernels.vector_add_operation(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            Arc::clone(&lazy_out),
            Arc::clone(&lazy_a),
            Arc::clone(&lazy_b),
        )?
    }
    .async_in(&pool)
    .wait()?;
    assert_eq!(lazy_completion.retained_count(), 4);
    let lazy = lazy_out.copy_to_vec()?;
    assert_eq!(lazy[4096], a[4096] + b[4096]);

    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 2.0,
        bias: 3.0,
    }])?;
    unsafe {
        kernels.affine_transform(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &d_a,
            &params,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let affine = d_out.copy_to_vec()?;
    assert_eq!(affine[17], a[17] * 2.0 + 3.0);

    println!("Generated binding vector_add passed on {}", device.arch());
    Ok(())
}

fn assert_close(
    label: &str,
    got: f32,
    expected: f32,
    tolerance: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if (got - expected).abs() > tolerance {
        Err(format!("{label}: got {got}, expected {expected} +/- {tolerance}").into())
    } else {
        Ok(())
    }
}
