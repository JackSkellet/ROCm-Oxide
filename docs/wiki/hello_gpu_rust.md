# Hello GPU Rust — Rust-Authored Kernel Walkthrough

`examples/hello_gpu_rust.rs` is the main SDK example. It demonstrates the full
Rust-to-GPU pipeline: a kernel written in Rust, compiled to an AMD `.hsaco`
code object at build time, and launched with automatically generated typed host
bindings.

Compare this with [hello_gpu.md](hello_gpu.md), which uses HIPRTC to compile a
HIP C++ kernel at runtime. That path is the fastest smoke test. This path is the
production SDK vision.

---

## Run it

```sh
cargo run --features device-spike --example hello_gpu_rust
```

Expected output:

```
hello_gpu_rust: device 0 (gfx1100)
hello_gpu_rust: 1048576 elements verified — Rust-authored kernel passed on gfx1100
```

The first build triggers the full pipeline (usually 20–60 s). Subsequent builds
are incremental and very fast unless `device-spike/src` changes.

The feature flag is intentional: it keeps normal source-workspace host builds
and external path dependencies from compiling the repository's reference
`device-spike` crate during setup. Generated projects created by
`cargo rocm-oxide new` compile their own local `device-spike/` crate and do not
need this flag.

---

## How the two-file structure works

The SDK separates GPU code from host code into two Rust crates:

```
device-spike/          ← GPU crate (#![no_std], compiled for amdgcn-amd-amdhsa)
  src/lib.rs           ← #[kernel] functions run on the GPU
  Cargo.toml

examples/
  hello_gpu_rust.rs    ← Host code: loads the kernel, allocates buffers, launches

build.rs               ← Orchestrates the build pipeline
```

`cargo build --features device-spike` on the source workspace triggers
`build.rs`, which runs `rocm-oxide-build` as a subprocess. The build tool:

1. Scans `device-spike/src/lib.rs` for `#[kernel]` functions.
2. Compiles the device crate with `cargo rustc -Z build-std=core --target amdgcn-amd-amdhsa`.
3. Rewrites the LLVM IR (`transform_ir`): adds `amdgpu_kernel` calling convention, fixes address spaces.
4. Lowers the IR to a `.o` file with ROCm `llc`.
5. Links the object into a `.hsaco` code object with ROCm `clang`.
6. Validates the output with `llvm-readelf`.
7. Generates a typed `DeviceKernels` struct in `bindings.rs` — one method per `#[kernel]`.
8. Generates `metadata.json` with argument layout records.

`build.rs` then copies these artifacts to `OUT_DIR` and sets env vars:

| Env var | Points to |
|---------|----------|
| `ROCM_OXIDE_DEVICE_HSACO` | Compiled `.hsaco` code object |
| `ROCM_OXIDE_DEVICE_BINDINGS` | Generated `bindings.rs` with `DeviceKernels` |
| `ROCM_OXIDE_DEVICE_METADATA` | `metadata.json` with argument type records |
| `ROCM_OXIDE_DEVICE_MANIFEST` | `manifest.json` with all artifact paths |

The host example includes the bindings with:

```rust
mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}
```

---

## The kernel (device-spike/src/lib.rs)

```rust
use rocm_oxide_device as gpu;
use rocm_oxide_kernel::kernel;

#[kernel]
pub unsafe extern "C" fn vector_add(
    out: gpu::DeviceSliceMut<f32>,
    a:   gpu::DeviceSlice<f32>,
    b:   gpu::DeviceSlice<f32>,
) {
    let i = gpu::global_id_x();
    if i < out.len() {
        let lhs = unsafe { a.read_unchecked(i) };
        let rhs = unsafe { b.read_unchecked(i) };
        unsafe { out.write_unchecked(i, lhs + rhs) };
    }
}
```

**Key points:**

- `#![no_std]` — no heap allocator, no standard library. Only `core` and the
  `rocm-oxide-device` GPU API are available.
- `#[kernel]` — marks the function as a GPU entry point. The build tool scans
  for this attribute to know which functions to compile and expose.
- `pub unsafe extern "C"` — required calling convention for GPU kernels. The
  `extern "C"` keeps the symbol name unmangled so the host can look it up by name.
- `gpu::DeviceSliceMut<f32>` / `gpu::DeviceSlice<f32>` — `#[repr(C)]` fat
  pointers carrying `(ptr, len)`. They are ABI-safe to pass through the kernel
  argument list and appear on the host side as `&DeviceBuffer<f32>`.
- `gpu::global_id_x()` — returns `blockIdx.x * blockDim.x + threadIdx.x`.
  The bounds check `i < out.len()` handles grids that are slightly larger than `n`.

---

## The generated typed binding

`rocm-oxide-build` generates this method from the kernel signature:

```rust
pub unsafe fn vector_add(
    &self,
    config: rocm_oxide::LaunchConfig,
    out: &rocm_oxide::DeviceBuffer<f32>,
    a:   &rocm_oxide::DeviceBuffer<f32>,
    b:   &rocm_oxide::DeviceBuffer<f32>,
) -> rocm_oxide::Result<()>
```

Compared to using `launch!` with raw pointers, the generated method:
- Takes `&DeviceBuffer<f32>` instead of `(*mut f32, usize)` pairs.
- Validates that `a.len() == out.len()` and `b.len() == out.len()`.
- Validates that `out`, `a`, `b` do not overlap in device memory.
- Expands each `DeviceSlice` into the `(ptr, len)` pair the ABI expects.

---

## The host code (examples/hello_gpu_rust.rs)

```rust
// Load the HSACO embedded in the binary and validate against metadata.
let kernels = generated::DeviceKernels::load_embedded(&device)?;

// Allocate GPU buffers.
let d_a = DeviceBuffer::from_slice(&a)?;    // host → device copy
let d_b = DeviceBuffer::from_slice(&b)?;
let d_out = DeviceBuffer::<f32>::new(n)?;   // uninitialized output

// Launch via the generated typed method.
unsafe {
    kernels.vector_add(LaunchConfig::for_num_elems(n), &d_out, &d_a, &d_b)?;
}

// Synchronize and read back.
rocm_oxide::hip::synchronize()?;
let out = d_out.copy_to_vec()?;
```

`LaunchConfig::for_num_elems(n)` computes a 1-D grid with 256 threads per block
(the default). The grid has enough blocks to cover all `n` elements.

---

## Requirements

| Requirement | Detail |
|-------------|--------|
| AMD GPU | Any ROCm-supported GPU (RDNA 2+, CDNA 2+) |
| ROCm | 6.0+ at `/opt/rocm` or `ROCM_PATH` |
| Rust nightly | Selected by `rust-toolchain.toml` |
| `rust-src` component | Required by `-Z build-std=core` |
| ROCm `llc` | Lowers LLVM IR to object file |
| ROCm `clang` | Links objects to `.hsaco` |
| ROCm `llvm-readelf` | Validates code object |

All ROCm tools are discovered automatically by the build tool. `clang` and
`rocminfo` are at `/opt/rocm/bin/`; `llc`, `llvm-readelf`, and `llvm-objdump`
are at `/opt/rocm/lib/llvm/bin/`. Run `cargo rocm-oxide doctor` to verify all
locations.

---

## Troubleshooting

### `error[E0463]: can't find crate for 'core'`

The `rust-src` component is missing. Install it for the active nightly:

```sh
rustup component add rust-src --toolchain nightly
```

### `could not compile device crate: error[E0658]: use of unstable library feature`

The device crate uses nightly-only features (`stdarch_amdgpu`, `fn_traits`,
`unboxed_closures`). Use the nightly toolchain selected by
`rust-toolchain.toml`, then re-install `rust-src` if doctor reports it missing.

### `llc: command not found` / `clang: command not found`

ROCm tools are not on `PATH`. Note that `llc` lives under `lib/llvm/bin/`, not
`bin/`:

```sh
# Fish shell — permanent
fish_add_path /opt/rocm/bin
fish_add_path /opt/rocm/lib/llvm/bin

# Bash — for the current session
export PATH="/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH"
```

Alternatively, set explicit overrides or `ROCM_PATH` and let the build tool
find them:

```sh
export ROCM_PATH=/opt/rocm                              # preferred
export ROCM_OXIDE_LLC=/opt/rocm/lib/llvm/bin/llc        # or override directly
ROCM_PATH=/path/to/rocm cargo run --features device-spike --example hello_gpu_rust
```

### `rocm-oxide-build failed: no #[kernel] functions found in device crate bundle`

`build.rs` could not find any `#[kernel]` functions. Check that
`device-spike/src/lib.rs` has functions annotated with `#[kernel]` and that the
device crate compiles. Try:

```sh
cargo build --manifest-path device-spike/Cargo.toml --target amdgcn-amd-amdhsa \
  -Z build-std=core 2>&1 | head -40
```

### `rocm-oxide-build failed: failed to detect ROCm GPU architecture`

No AMD GPU was found. Either:
- The GPU driver is not loaded (`lsmod | grep amdgpu`).
- Your user lacks access to `/dev/kfd` (`ls -la /dev/kfd` — should be `crw-rw----`).
- Add your user to the `render` and `video` groups: `sudo usermod -aG render,video $USER`.

Force the architecture manually if the GPU is present but not detected:

```sh
ROCM_OXIDE_ARCH=gfx1100 cargo run --features device-spike --example hello_gpu_rust
```

### `Architecture mismatch: compiled for gfx1100, device is gfx1201`

The cached `.hsaco` was compiled for a different GPU. Clean and rebuild:

```sh
cargo clean
ROCM_OXIDE_ARCH=gfx1201 cargo run --features device-spike --example hello_gpu_rust
```

### `.hsaco` validation failed: symbol `vector_add` not found

The linker step produced an empty or corrupt code object. This usually means
the IR rewrite step dropped the kernel (e.g. the `#[kernel]` annotation was
not on a `pub unsafe extern "C"` function). Check the function signature in
`device-spike/src/lib.rs` against the requirements listed under "The kernel" above.

### Build succeeds but output is wrong

The kernel ran but produced incorrect results. Possible causes:
- A mismatch between the kernel's bounds check (`i < out.len()`) and the
  actual buffer size — make sure `DeviceBuffer::new(n)` and
  `LaunchConfig::for_num_elems(n)` both use the same `n`.
- Uninitialized output buffer elements that the kernel did not write (should
  not happen for a 1-D kernel with the correct bounds check).

Enable device debug builds to get source-level correlation in crash reports:

```sh
ROCM_OXIDE_DEVICE_DEBUG=1 cargo run --features device-spike --example hello_gpu_rust
```

---

## What gets generated (artifact tour)

After a successful build, these files exist in `device-spike/target/amdgcn-amd-amdhsa/release/`:

| File | Description |
|------|-------------|
| `rocm_oxide_device_spike.hsaco` | GPU code object, loadable by `hipModuleLoadData` |
| `rocm_oxide_device_spike.kernel.ll` | Transformed LLVM IR (human-readable, useful for debugging) |
| `rocm_oxide_device_spike.o` | Intermediate object file |
| `rocm_oxide_device_spike.bindings.rs` | Generated `DeviceKernels` struct |
| `rocm_oxide_device_spike.metadata.json` | Argument types, sizes, alignment records |
| `rocm_oxide_device_spike.manifest.json` | Build provenance: tools, versions, paths |

Inspect the metadata:

```sh
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- \
  --inspect-metadata device-spike/target/amdgcn-amd-amdhsa/release/rocm_oxide_device_spike.metadata.json
```

Disassemble the HSACO to verify it contains correct AMDGPU ISA:

```sh
/opt/rocm/lib/llvm/bin/llvm-objdump -d \
  device-spike/target/amdgcn-amd-amdhsa/release/rocm_oxide_device_spike.hsaco \
  | grep -A 20 "vector_add"
```

---

## What still feels complex

- **Two crates, one project**: new users expect the kernel to live in the same
  file as the host code. The two-crate structure is necessary for separate
  compilation targets but is the biggest onboarding friction point.
- **`env!("ROCM_OXIDE_DEVICE_BINDINGS")` is magic**: the generated bindings are
  included via an env var set by `build.rs`. There is no file to inspect in the
  source tree — you have to know to look in `target/`.
- **Build time on first run**: the device crate compiles from scratch the first
  time (20–60 s depending on machine), which feels slow with no output.
- **Nightly-only**: the device crate depends on unstable Rust features
  (`stdarch_amdgpu`, `-Z build-std`). There is no stable-Rust GPU path yet.

---

## Next steps

- **Add your own kernel**: open `device-spike/src/lib.rs`, add a `#[kernel]` function,
  rebuild. `DeviceKernels` will gain a matching typed method automatically.
- **Raw launch API**: see `examples/rust_device_add_one.rs` for the same pattern
  using `launch!` with explicit pointer/length pairs instead of the generated binding.
- **Device globals**: see `examples/module_global.rs` for reading and writing
  a `#[device_global]` variable from the host.
- **Full API**: see [api_overview.md](api_overview.md).
