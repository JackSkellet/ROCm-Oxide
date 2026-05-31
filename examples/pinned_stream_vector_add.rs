use rocm_oxide::{Device, DeviceBuffer, LaunchConfig, PinnedHostBuffer, Stream};

const KERNEL: &str = r#"
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let module = device.compile_hip_source(KERNEL)?;
    let kernel = module.kernel(c"vector_add")?;
    let stream = Stream::new()?;
    let n = 1 << 20;

    let a_host = PinnedHostBuffer::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>())?;
    let b_host =
        PinnedHostBuffer::from_slice(&(0..n).map(|i| (i as f32) * 3.0).collect::<Vec<_>>())?;
    let mut out_host = PinnedHostBuffer::<f32>::new_zeroed(n)?;

    let d_a = DeviceBuffer::<f32>::new_async(&stream, n)?;
    let d_b = DeviceBuffer::<f32>::new(n)?;
    let d_out = DeviceBuffer::<f32>::new_async(&stream, n)?;
    d_a.copy_from_pinned_host(&a_host)?;
    d_b.copy_from_pinned_host_async(&stream, &b_host)?;

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
        kernel.launch_raw_on_stream(&stream, config, &mut params)?;
    }

    d_out.copy_to_pinned_host_async(&stream, &mut out_host)?;
    stream.synchronize()?;

    for (i, value) in out_host.as_slice().iter().enumerate() {
        let expected = i as f32 * 4.0;
        assert_eq!(*value, expected, "mismatch at {i}");
    }
    let mut sync_copy = PinnedHostBuffer::<f32>::new_zeroed(n)?;
    d_out.copy_to_pinned_host(&mut sync_copy)?;
    assert_eq!(sync_copy.as_slice()[n - 1], (n - 1) as f32 * 4.0);
    unsafe {
        d_a.free_async(&stream)?;
        d_out.free_async(&stream)?;
    }
    stream.synchronize()?;
    println!(
        "pinned host + HIP stream vector_add passed on {}",
        device.arch()
    );
    Ok(())
}
