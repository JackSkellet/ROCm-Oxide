# Hello GPU — Walkthrough

`examples/hello_gpu.rs` is the smallest self-contained workload in this
repository. It demonstrates the complete GPU execution lifecycle using the
HIPRTC runtime-compilation path: no separate device-crate build is needed.

## What it does

1. Opens the first AMD GPU with `Device::first()`.
2. Compiles an inline HIP C++ kernel at runtime with `device.compile_hip_source()`.
3. Uploads two `f32` vectors to the GPU with `DeviceBuffer::from_slice()`.
4. Launches the kernel with `launch!` and `LaunchConfig::for_num_elems()`.
5. Synchronizes the device with `rocm_oxide::hip::synchronize()`.
6. Downloads the result with `DeviceBuffer::copy_to_vec()` and verifies correctness.

## Run it

```sh
cargo run --example hello_gpu
```

Expected output:

```
hello_gpu: device 0 (gfx1100)
hello_gpu: 1048576 elements verified — all correct on gfx1100
```

The GPU architecture string (`gfx1100`, `gfx1201`, etc.) matches your hardware.

## Requirements

| Requirement | Detail |
|-------------|--------|
| AMD GPU | Any ROCm-supported GPU (RDNA 2+, CDNA 2+) |
| ROCm | 6.0+ installed at `/opt/rocm` (or pointed to by `ROCM_PATH`) |
| Rust nightly | Required for this workspace (`rust-toolchain.toml` pins it) |
| `libamdhip64.so` | Provided by ROCm; linked automatically by `build.rs` |
| `libhiprtc.so` | Provided by ROCm; used to compile the kernel at runtime |

The kernel is compiled the first time and the result is cached in memory for
the process lifetime. HIPRTC compilation typically takes < 500 ms the first
time; subsequent runs reuse the in-memory cache.

## How it works

### The kernel

```c
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long n) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
```

This is standard HIP C++. Each GPU thread computes one output element. The
bounds check `i < n` handles the case where the grid is slightly larger than
`n` (which is normal when `n` is not a multiple of the block size).

### Device and module

```rust
let device = Device::first()?;
let module = device.compile_hip_source(KERNEL_SOURCE)?;
let kernel = module.kernel(c"vector_add")?;
```

`Device::first()` opens GPU ordinal 0. If you have multiple GPUs, use
`Device::at(1)` to select by index, or `Device::all()` to open all devices.

`compile_hip_source` calls HIPRTC (or COMGR when the HIPRTC path is
unavailable) and returns a loaded `Module`. `module.kernel(c"name")` looks up
the `extern "C"` function in the compiled code object.

### Buffers

```rust
let d_a = DeviceBuffer::from_slice(&a)?;  // allocate + copy host → device
let d_b = DeviceBuffer::from_slice(&b)?;
let d_out = DeviceBuffer::<f32>::new(n)?; // allocate only; contents undefined
```

`DeviceBuffer` owns a `hipMalloc` allocation. `from_slice` allocates and does
a synchronous host-to-device copy. `new(n)` allocates without initializing
(the kernel writes every element, so initialization would be wasted work).

### Launch

```rust
unsafe {
    rocm_oxide::launch!(
        kernel,
        LaunchConfig::for_num_elems(n),
        d_out.as_mut_ptr(),
        d_a.as_ptr(),
        d_b.as_ptr(),
        n as u64,
    )?;
}
```

The `launch!` macro builds the HIP kernel argument array and calls
`hipLaunchKernel`. It is `unsafe` because argument types are checked at
the programmer's discretion — they must match the HIP C++ function signature.
`LaunchConfig::for_num_elems(n)` computes a 1-D grid with 256 threads per
block, which is a sensible default for most element-wise kernels.

Kernel launch is asynchronous — control returns to the host before the GPU
has finished. The following `synchronize()` call blocks until the GPU is idle.

### Verify

```rust
for (i, ((&got, &a_i), &b_i)) in out.iter().zip(&a).zip(&b).enumerate() {
    let expected = a_i + b_i;
    if (got - expected).abs() > f32::EPSILON {
        return Err(format!("mismatch at index {i}").into());
    }
}
```

Each output element is checked against the expected `a[i] + b[i]` within
floating-point tolerance. A failure here indicates a kernel logic error or a
buffer aliasing problem.

## Next steps

- **Rust device kernels**: see `examples/rust_device_vector_add.rs` for the
  same workload written as a `#![no_std]` Rust kernel compiled with
  `rocm-oxide-build`.
- **Generated typed bindings**: see `examples/rust_device_generated_bindings.rs`
  for the `DeviceKernels` struct pattern.
- **Streams and events**: add `Stream::new()` and use
  `Kernel::launch_raw_on_stream` for pipelined execution.
- **Full API**: see [api_overview.md](api_overview.md).
