//! Minimal end-to-end "Hello GPU" example for ROCm Oxide.
//!
//! This example shows the complete lifecycle of a GPU workload using the
//! HIPRTC runtime-compilation path, which requires no separate device-crate
//! build step — just ROCm installed at /opt/rocm and a visible AMD GPU.
//!
//! Steps:
//!   1. Open the first AMD GPU.
//!   2. Compile an inline HIP C++ kernel at runtime (result is cached by hash).
//!   3. Upload input data to the GPU.
//!   4. Launch the kernel.
//!   5. Synchronize (wait for the GPU to finish).
//!   6. Download results and verify correctness.
//!
//! Run with:
//!   cargo run --example hello_gpu

use rocm_oxide::prelude::*;

/// A simple element-wise linear transform: `out[i] = a[i] + b[i]`.
///
/// Written in HIP C++ and compiled at runtime via HIPRTC. The `extern "C"`
/// linkage and `__global__` qualifier mark it as a GPU kernel entry point.
const KERNEL_SOURCE: &str = r#"
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long n) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

fn main() -> Result<()> {
    // ── Step 1: open the first AMD GPU ──────────────────────────────────────
    let device = Device::first()?;
    println!("hello_gpu: device {} ({})", device.ordinal(), device.arch());

    // ── Step 2: compile the kernel from HIP C++ source ───────────────────────
    // HIPRTC compiles to a device code object at runtime.
    // Repeated calls with the same source return a cached result.
    let module = device.compile_hip_source(KERNEL_SOURCE)?;
    let kernel = module.kernel(c"vector_add")?;

    // ── Step 3: prepare input data and upload to the GPU ─────────────────────
    let n: usize = 1 << 20; // 1 048 576 elements

    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let d_a = DeviceBuffer::from_slice(&a)?; // host → device copy
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(n)?; // uninitialized output buffer

    // ── Step 4: launch the kernel ─────────────────────────────────────────────
    // `LaunchConfig::for_num_elems` picks a grid/block shape for a 1-D workload.
    // The kernel arguments must exactly match the C++ signature above.
    //
    // SAFETY: argument types and order match `vector_add`'s C++ signature.
    //         The device buffers are live for the duration of the launch.
    let config = LaunchConfig::for_num_elems(n);
    unsafe {
        rocm_oxide::launch!(
            kernel,
            config,
            d_out.as_mut_ptr(), // float* out
            d_a.as_ptr(),       // const float* a
            d_b.as_ptr(),       // const float* b
            n as u64,           // unsigned long n
        )?;
    }

    // ── Step 5: wait for the GPU to finish ────────────────────────────────────
    rocm_oxide::hip::synchronize()?;

    // ── Step 6: download and verify the results ───────────────────────────────
    let out = d_out.copy_to_vec()?;

    for (i, ((&got, &a_i), &b_i)) in out.iter().zip(&a).zip(&b).enumerate() {
        let expected = a_i + b_i;
        if (got - expected).abs() > f32::EPSILON {
            return Err(Error::InvalidLaunch(format!(
                "mismatch at index {i}: got {got}, expected {expected}"
            )));
        }
    }

    println!(
        "hello_gpu: {} elements verified — all correct on {}",
        n,
        device.arch()
    );
    Ok(())
}
