use rocm_oxide::{
    Device, DeviceBuffer, DeviceOperation, ExecutionContext, LaunchConfig, Result, StreamPool,
};
use std::sync::mpsc;
use std::time::Duration;

const KERNEL: &str = r#"
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
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
        assert_eq!(n, b.len(), "input vectors must have matching lengths");

        let module = device.compile_hip_source(KERNEL)?;
        let kernel = module.kernel(c"vector_add")?;

        let d_a = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        let d_b = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        let d_out = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        d_a.copy_from_host_async(context.stream(), &a)?;
        d_b.copy_from_host_async(context.stream(), &b)?;

        let config = LaunchConfig::for_num_elems(n, 256);
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
    let pool = StreamPool::new(&device, 2)?;
    let n = 1 << 18;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| i as f32 * 2.0).collect::<Vec<_>>();

    let chained_sum = vector_add_operation(device.clone(), a.clone(), b.clone())
        .map(|out| out.into_iter().sum::<f32>())
        .sync_on(&device.execution_context()?)?;
    let expected_sum = (0..n).map(|i| i as f32 * 3.0).sum::<f32>();
    assert!((chained_sum - expected_sum).abs() <= 64.0);

    let future_a = vector_add_operation(device.clone(), a.clone(), b.clone()).async_in(&pool);
    let future_b = vector_add_operation(device.clone(), b.clone(), b.clone()).async_in(&pool);
    let out_a = future_a.wait()?;
    let out_b = future_b.wait()?;
    assert_eq!(out_a[n - 1], (n - 1) as f32 * 3.0);
    assert_eq!(out_b[n - 1], (n - 1) as f32 * 4.0);

    let (sent, received) = mpsc::channel();
    let dropped_future = (move |_context: &ExecutionContext| -> Result<()> {
        let _ = sent.send(());
        Ok(())
    })
    .async_in(&pool);
    drop(dropped_future);
    received
        .recv_timeout(Duration::from_secs(2))
        .expect("dropped DeviceFuture should not cancel in-flight work");

    println!(
        "DeviceOperation chain passed on {} with {} streams",
        device.arch(),
        pool.len()
    );
    Ok(())
}
