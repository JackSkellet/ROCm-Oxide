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
> **Branch:** `main`
> **Validated profiles:** `gfx1100`, `gfx1201`

ROCm-Oxide is not production-stable yet. See
[docs/api-stability.md](docs/api-stability.md).

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

- [docs/wiki/hello_gpu.md](docs/wiki/hello_gpu.md)

---

### 2. Rust-authored GPU kernel

This is the main SDK path.

```sh
cargo run --features device-spike --example hello_gpu_rust
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

The `device-spike` feature is required only for source-workspace examples that
embed the repository's reference Rust device crate. Normal host/runtime builds
and generated consumer projects do not compile this reference kernel code.

Walkthrough:

- [docs/wiki/hello_gpu_rust.md](docs/wiki/hello_gpu_rust.md)

---

## Creating a consumer project

> **Manual project creation is not supported.** Hand-authoring `Cargo.toml`,
> `build.rs`, or `rust-toolchain.toml` will produce incorrect path dependencies
> and opaque compiler errors. Always use `cargo rocm-oxide new`.

Install the cargo subcommand once from the repository root:

```sh
cargo install --path tools/cargo-rocm-oxide
```

Then run the four-command sequence:

```sh
# 1. Clone (or cd into) the ROCm-Oxide workspace
git clone https://github.com/JackSkellet/ROCm-Oxide.git
cd ROCm-Oxide

# 2. Generate the scaffold — paths are computed automatically
cargo rocm-oxide new ../my-project

# 3. Validate the scaffold paths
cd ../my-project
cargo rocm-oxide check-consumer

# 4. Build and run
cargo run
```

The generated project uses relative `path` links back to this workspace.
Move both together and the build stays intact. See
[docs/project-generation.md](docs/project-generation.md) for portability options.

If you run the generator from outside the ROCm-Oxide checkout, point it at the
workspace explicitly:

```sh
cargo rocm-oxide new my-project --local /path/to/ROCm-Oxide
```

`--standalone` is reserved until the runtime, device API, proc macro, and build
tool can be consumed through crates.io or release artifacts.

---

## Requirements

Required:

- ROCm 7.2 for release-gating validation
- Rust installed with `rustup`
- ROCm-supported AMD GPU
- access to `/dev/kfd`

ROCm 6.x and earlier 7.x releases may work for some host/runtime paths, but the
current preview release gates are validated on ROCm 7.2.

For Rust-authored device kernels:

```sh
rustup component add rust-src --toolchain nightly
```

If something fails, run:

```sh
cargo rocm-oxide doctor
```

The doctor command checks the ROCm tools, GPU visibility, `/dev/kfd`,
Rust toolchain, `rust-src`, workspace/scaffold context, and detected GPU target.
It prints PASS/WARN/FAIL results plus a copy-pasteable GitHub issue block. Use
`cargo rocm-oxide doctor --json` for machine-readable output or
`cargo rocm-oxide doctor --github` for only the issue-ready report block.

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
    kernels
        .vector_add_launcher()
        .grid_for(n)
        .launch(&out, &a, &b, n)?;
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

- [docs/project-generation.md](docs/project-generation.md)
- [docs/getting-started.md](docs/getting-started.md)

---

## Tooling

After installing `cargo-rocm-oxide`, these commands are available:

```sh
cargo rocm-oxide doctor
cargo rocm-oxide build
cargo rocm-oxide check-consumer
cargo rocm-oxide verify --quick
cargo rocm-oxide verify --full
cargo rocm-oxide pipeline
cargo rocm-oxide profile
```

Common uses:

| Command | Purpose |
|---|---|
| `cargo rocm-oxide doctor` | Check ROCm, Rust, GPU, and workspace setup |
| `cargo rocm-oxide doctor --json` | Emit the same doctor report as JSON |
| `cargo rocm-oxide doctor --github` | Emit only the issue-ready doctor report block |
| `cargo rocm-oxide build` | Build device artifacts |
| `cargo rocm-oxide check-consumer` | Validate a generated scaffold project |
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

- [docs/release.md](docs/release.md)
- [docs/sdk-preview-restructure-plan.md](docs/sdk-preview-restructure-plan.md)

---

## Documentation

Start with [docs/index.md](docs/index.md).

Maintained docs:

- [docs/getting-started.md](docs/getting-started.md)
- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/api-stability.md](docs/api-stability.md)
- [docs/project-generation.md](docs/project-generation.md)
- [docs/release.md](docs/release.md)
- [docs/visual-demos.md](docs/visual-demos.md)

Long-form design notes and historical checklists are kept as wiki source under
[docs/wiki/](docs/wiki/README.md).

---

## Examples

Beginner examples:

```sh
cargo run --example hello_gpu
cargo run --features device-spike --example hello_gpu_rust
cargo run --example vector_add
cargo run --features device-spike --example rust_device_generated_bindings
```

Runtime and feature examples:

```sh
cargo run --features device-spike --example feature_showcase
cargo run --example validation_profile
cargo run --features device-spike --example performance_probe -- --json target/performance_probe.json
```

Visual and experimental demos:

```sh
cd demo-projects/spectral-lattice && cargo run -- --present vulkan --frames 3 --resolution 4k --fps-limit 120
cd ../matrix-lens && cargo run -- --resolution 720p --mode matrix
cd ../compiler-feature-lab && cargo run -- --present vulkan --frames 1
cd ../stress-gui && cargo run --bin stress_test_gui -- --present vulkan --frames 300
```

Examples that use generated Rust-device bindings require
`device-spike` when run from the source root. Separated demo projects own their
manifests, README files, and demo-only dependencies. HIPRTC-only examples such
as `hello_gpu` and `vector_add` do not need demo features.

For the full visual demo table, see:

- [docs/visual-demos.md](docs/visual-demos.md)
- [examples/README.md](examples/README.md)
- [demo-projects/README.md](demo-projects/README.md)

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

and include the copy-pasteable diagnostic block in the issue. `cargo rocm-oxide
doctor --github` prints only that block.

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
