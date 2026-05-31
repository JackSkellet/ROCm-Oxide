use rocm_oxide::{
    Device, DeviceBuffer, DeviceOperation, ExecutionContext, LaunchConfig, Result, StreamPool,
};
use std::sync::{Arc, mpsc};
use std::time::Duration;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const VECTOR_ADD_HIPRTC: &str = r#"
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

const MODULE_GLOBAL_HIPRTC: &str = r#"
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

fn vector_add_operation(
    device: Device,
    a: Vec<f32>,
    b: Vec<f32>,
) -> impl DeviceOperation<Output = Vec<f32>> {
    move |context: &ExecutionContext| -> Result<Vec<f32>> {
        let n = a.len();
        let module = device.compile_hip_source(VECTOR_ADD_HIPRTC)?;
        let kernel = module.kernel(c"vector_add")?;

        let d_a = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        let d_b = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        let d_out = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        d_a.copy_from_host_async(context.stream(), &a)?;
        d_b.copy_from_host_async(context.stream(), &b)?;

        let config = LaunchConfig::for_num_elems(n);
        let mut out_ptr = d_out.as_mut_ptr();
        let mut a_ptr = d_a.as_ptr();
        let mut b_ptr = d_b.as_ptr();
        let mut n_arg = n as u64;
        let mut params = [
            rocm_oxide::__private::arg_ptr(&mut out_ptr),
            rocm_oxide::__private::arg_ptr(&mut a_ptr),
            rocm_oxide::__private::arg_ptr(&mut b_ptr),
            rocm_oxide::__private::arg_ptr(&mut n_arg),
        ];
        unsafe {
            kernel.launch_raw_on_stream(context.stream(), config, &mut params)?;
        }

        let mut out = vec![0.0f32; n];
        d_out.copy_to_host_async(context.stream(), &mut out)?;
        Ok(out)
    }
}

fn main() -> Result<()> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let n = 1 << 16;
    let block_x = 256;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (i as f32) * 0.5).collect::<Vec<_>>();

    println!("ROCm-Oxide feature showcase on {}", device.arch());

    let d_a = Arc::new(DeviceBuffer::from_slice(&a)?);
    let d_b = Arc::new(DeviceBuffer::from_slice(&b)?);
    let d_out = Arc::new(DeviceBuffer::<f32>::new(n)?);

    let short = Arc::new(DeviceBuffer::from_slice(&a[..n / 2])?);
    let rejected = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &short,
            &d_b,
        )
    };
    assert!(matches!(rejected, Err(rocm_oxide::Error::InvalidLaunch(_))));
    println!("ok: generated bindings rejected a short buffer before launch");

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
    assert_eq!(out[4096], a[4096] + b[4096]);
    println!("ok: Rust-authored AMDGPU vector_add launched from generated host bindings");

    let completion = unsafe {
        kernels.vector_add_operation(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            Arc::clone(&d_out),
            Arc::clone(&d_a),
            Arc::clone(&d_b),
        )?
    }
    .sync_on(&device.execution_context()?)?;
    assert_eq!(completion.retained_count(), 3);
    let lazy_out = d_out.copy_to_vec()?;
    assert_eq!(lazy_out[8192], a[8192] + b[8192]);
    println!("ok: generated DeviceOperation binding launched on an execution stream");

    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 3.0,
        bias: -7.0,
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
    assert_eq!(affine[1234], a[1234] * 3.0 - 7.0);
    println!("ok: mirrored repr(C) environment struct reached GPU code");

    let module = device.compile_hip_source(MODULE_GLOBAL_HIPRTC)?;
    let kernel = module.kernel(c"scale_vec")?;
    let scale = module.global::<f32>(c"scale_factor")?;
    scale.set(4.0)?;
    unsafe {
        rocm_oxide::launch!(
            kernel,
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            d_out.as_mut_ptr(),
            d_a.as_ptr(),
            n as u64,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let scaled = d_out.copy_to_vec()?;
    assert_eq!(scaled[321], a[321] * 4.0);
    assert_eq!(scale.copy_to_vec()?, vec![4.0]);
    println!("ok: HIP module global lookup/set/get updated GPU behavior");

    let pool = StreamPool::new(&device, 2)?;
    let future_a = vector_add_operation(device.clone(), a.clone(), b.clone()).async_in(&pool);
    let future_b = vector_add_operation(device.clone(), b.clone(), b.clone()).async_in(&pool);
    let async_a = future_a.wait()?;
    let async_b = future_b.wait()?;
    assert_eq!(async_a[n - 1], a[n - 1] + b[n - 1]);
    assert_eq!(async_b[n - 1], b[n - 1] * 2.0);
    println!("ok: lazy DeviceOperation jobs completed through a 2-stream pool");

    let (sent, received) = mpsc::channel();
    let dropped = (move |_context: &ExecutionContext| -> Result<()> {
        let _ = sent.send(());
        Ok(())
    })
    .async_in(&pool);
    drop(dropped);
    received
        .recv_timeout(Duration::from_secs(2))
        .expect("dropped DeviceFuture should not cancel already-started work");
    println!("ok: dropping DeviceFuture did not cancel in-flight work");

    println!("feature showcase passed");
    Ok(())
}
