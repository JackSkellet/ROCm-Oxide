# ROCm-Oxide

**Rust-first GPU development on AMD/ROCm.**

<img width="1925" height="1113" alt="image" src="https://github.com/user-attachments/assets/a1b877e9-af51-44e4-bd73-de1c0be8cd25" />

ROCm-Oxide is an experimental SDK preview for writing, building, launching,
testing, and profiling AMD GPU workloads from Rust.

It currently provides:

- Rust-authored AMD GPU kernels
- Generated typed host bindings
- `.hsaco` code object generation
- Kernel metadata generation
- HIPRTC runtime compilation
- COMGR/toolchain integration
- Device buffers, streams, events, and graph helpers
- `cargo rocm-oxide` diagnostics, build, verification, pipeline, and profiling tools

> **Status:** Experimental SDK preview  
> **Branch:** `sdk-preview-0`  
> **Validated profiles:** `gfx1100`, `gfx1201`

ROCm-Oxide is not production-stable yet. See
[docs/stability-policy.md](docs/stability-policy.md).

---

## How it works

```text
Rust device kernel
        ↓
rocm-oxide-build
        ↓
HSACO + metadata + typed bindings
        ↓
ROCm-Oxide runtime
        ↓
ROCm / HIP
        ↓
AMD GPU
```

The main goal is to make AMD GPU development feel like a Rust workflow:

```text
write Rust
cargo build
cargo run
verify with typed bindings
```

---

## Start here

There are two recommended first runs.

### 1. Quick GPU smoke test

This verifies that ROCm, GPU access, runtime compilation, and kernel launch work.

```sh
cargo run --example hello_gpu
```

Expected output:

```text
hello_gpu: device 0 (gfx1100)
hello_gpu: 1048576 elements verified — vector add passed
```

This path uses HIPRTC runtime compilation with an inline HIP C++ kernel. It does
not require the Rust device-kernel build pipeline.

Walkthrough:

- [docs/hello_gpu.md](docs/hello_gpu.md)

---

### 2. Rust-authored GPU kernel

This is the main SDK path.

```sh
cargo run --example hello_gpu_rust
```

Expected output:

```text
hello_gpu_rust: device 0 (gfx1100)
hello_gpu_rust: 1048576 elements verified — Rust-authored kernel passed on gfx1100
```

This path demonstrates:

- Rust device-kernel source
- device-crate compilation
- `.hsaco` generation
- metadata generation
- generated typed host bindings
- host-side launch and verification

The first build can take 20–60 seconds while the device crate compiles.
Subsequent builds are incremental.

Walkthrough:

- [docs/hello_gpu_rust.md](docs/hello_gpu_rust.md)

---

## Requirements

Required:

- ROCm 6.0+
- Rust installed with `rustup`
- ROCm-supported AMD GPU
- access to `/dev/kfd`

For Rust-authored device kernels:

```sh
rustup component add rust-src
```

If something fails, run:

```sh
cargo rocm-oxide doctor
```

The doctor command checks the ROCm tools, GPU visibility, `/dev/kfd`,
Rust toolchain, `rust-src`, workspace/scaffold context, and detected GPU target.
It prints PASS/WARN/FAIL results plus a copy-pasteable GitHub issue block.

Full troubleshooting guide:

- [docs/troubleshooting.md](docs/troubleshooting.md)

---

## Example

A Rust-authored GPU kernel looks like this:

```rust
#[kernel]
pub unsafe extern "C" fn vector_add(
    out: gpu::DeviceSliceMut<f32>,
    a: gpu::DeviceSlice<f32>,
    b: gpu::DeviceSlice<f32>,
    n: usize,
) {
    let idx = gpu::global_id_x();

    if idx < n {
        out[idx] = a[idx] + b[idx];
    }
}
```

Host code loads the generated kernel bindings and launches the kernel:

```rust
let kernels = generated::DeviceKernels::load_embedded(&device)?;

unsafe {
    kernels.vector_add(
        LaunchConfig::for_num_elems(n),
        &out,
        &a,
        &b,
        n,
    )?;
}
```

Kernel launch remains `unsafe` because host code cannot fully prove every GPU ABI
and memory-safety condition at compile time. Generated bindings reduce launch
mistakes, but they do not make GPU code magically safe.

---

## Creating a local scaffold project

Install the cargo subcommand from the repository root:

```sh
cargo install --path tools/cargo-rocm-oxide
```

Then create a local scaffold from within or adjacent to the ROCm-Oxide workspace:

```sh
cargo rocm-oxide new my-gpu-project
cd my-gpu-project
cargo run
```

Important:

> `cargo rocm-oxide new` currently creates a **local scaffold**, not a fully
> standalone crates.io-ready project.

Generated projects use relative `path` links back to the ROCm-Oxide workspace.
They can be moved together with the workspace, but they are not yet designed to
be cloned and built independently on another machine.

Details:

- [docs/project_generation.md](docs/project_generation.md)
- [docs/getting-started.md](docs/getting-started.md)

---

## Tooling

After installing `cargo-rocm-oxide`, these commands are available:

```sh
cargo rocm-oxide doctor
cargo rocm-oxide build
cargo rocm-oxide verify --quick
cargo rocm-oxide verify --full
cargo rocm-oxide pipeline
cargo rocm-oxide profile
```

Common uses:

| Command | Purpose |
|---|---|
| `cargo rocm-oxide doctor` | Check ROCm, Rust, GPU, and workspace setup |
| `cargo rocm-oxide build` | Build device artifacts |
| `cargo rocm-oxide verify --quick` | Run the source-workspace quick verification gate |
| `cargo rocm-oxide verify --full` | Run the full verification gate |
| `cargo rocm-oxide pipeline` | Inspect generated kernels and pipeline artifacts |
| `cargo rocm-oxide profile` | Run profiling/inspection helpers |

`verify` is a repository gate and should be run from the ROCm-Oxide source
workspace. To check a generated scaffold project, run `cargo build` or
`cargo run` inside that project.

---

## What works today

### Runtime path

ROCm-Oxide can currently:

- discover AMD devices
- allocate and copy device memory
- compile HIP kernels at runtime
- load code objects
- launch kernels
- synchronize execution
- use streams and events
- use pinned and managed memory
- exercise graph helper paths

### Rust device-kernel path

The Rust device-kernel pipeline can currently:

- compile `#![no_std]` Rust device crates for `amdgcn-amd-amdhsa`
- use `#[kernel]`, `#[device_global]`, `#[constant]`, and `#[shared]`
- expose device intrinsics and atomics
- generate `.hsaco` code objects
- generate metadata JSON
- generate typed host bindings
- validate kernel contracts where metadata exists

### Validation

The SDK preview has passed:

- host CI gate
- offline gate
- quick hardware gate
- runtime tests
- doctor checks
- pipeline inspection
- generated-bindings example
- feature showcase
- consumer-smoke downstream compile
- validation profile
- performance probe
- spectral lattice artifact generation

Validated machine profiles:

- `gfx1100` — AMD Radeon RX 7900 XT
- `gfx1201` — AMD Radeon RX 9070 XT

See:

- [docs/release_checklist.md](docs/release_checklist.md)
- [docs/supported-rocm-gpu-matrix.md](docs/supported-rocm-gpu-matrix.md)

---

## Documentation

### Start here

- [docs/getting-started.md](docs/getting-started.md)
- [docs/api_overview.md](docs/api_overview.md)
- [docs/hello_gpu.md](docs/hello_gpu.md)
- [docs/hello_gpu_rust.md](docs/hello_gpu_rust.md)

### Troubleshooting and setup

- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/project_generation.md](docs/project_generation.md)
- [docs/toolchain-discovery.md](docs/toolchain-discovery.md)
- [docs/supported-rocm-gpu-matrix.md](docs/supported-rocm-gpu-matrix.md)

### Architecture and safety

- [DESIGN.md](DESIGN.md)
- [docs/sdk_direction.md](docs/sdk_direction.md)
- [docs/api-stability.md](docs/api-stability.md)
- [docs/compiler-path.md](docs/compiler-path.md)
- [docs/code-object-linking.md](docs/code-object-linking.md)
- [docs/unsafe-audit.md](docs/unsafe-audit.md)
- [docs/atomic-scopes.md](docs/atomic-scopes.md)
- [docs/host-memory-coherence.md](docs/host-memory-coherence.md)
- [docs/stream-ordered-allocation.md](docs/stream-ordered-allocation.md)
- [docs/rocm-library-interop.md](docs/rocm-library-interop.md)

### Release and contribution

- [CHANGELOG.md](CHANGELOG.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/release_checklist.md](docs/release_checklist.md)
- [SECURITY.md](SECURITY.md)

---

## Examples

Beginner examples:

```sh
cargo run --example hello_gpu
cargo run --example hello_gpu_rust
cargo run --example vector_add
cargo run --example rust_device_generated_bindings
```

Runtime and feature examples:

```sh
cargo run --example feature_showcase
cargo run --example validation_profile
cargo run --example performance_probe -- --json target/performance_probe.json
```

Visual and experimental demos:

```sh
cargo run --example spectral_lattice
cargo run --example spectral_lattice -- --frames 3 --resolution 4k --fps-limit 120
cargo run --example matrix_lens -- --resolution 720p --mode matrix
cargo run --example compiler_feature_lab
cargo run --example stress_test_gui
cargo run --example stress_3d_gui
```

For the full visual demo table, see:

- [docs/visual-demos.md](docs/visual-demos.md)

---

## Current priorities

The current focus is the SDK preview:

- first-user experience
- Rust-authored GPU kernels
- generated bindings
- diagnostics
- verification
- documentation
- release gates

Not current priorities:

- CUDA binary compatibility
- pretending to be `libcuda.so`
- multi-backend abstraction
- production stability guarantees
- crates.io distribution for generated projects

Those may be explored later, but they are not the purpose of this preview.

---

## Repository layout

```text
src/                  host runtime/library
crates/               device/runtime support crates
device-spike/         Rust-authored device kernel spike
tools/                build tool and cargo wrapper
examples/             examples, demos, feature labs, probes
docs/                 design, SDK, runtime, troubleshooting, release docs
scripts/              verification and compatibility scripts
```

---

## Contributing

See:

- [CONTRIBUTING.md](CONTRIBUTING.md)

Before filing a bug, please run:

```sh
cargo rocm-oxide doctor
```

and include the copy-pasteable diagnostic block in the issue.

Issue templates are provided for:

- bug reports
- GPU compatibility reports
- documentation issues

---

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

See:

- [LICENSE-APACHE](LICENSE-APACHE)
- [LICENSE-MIT](LICENSE-MIT)
