# ROCm-Oxide

Rust-first GPU development on AMD/ROCm, with a practical path toward CUDA Oxide-style ergonomics.

ROCm-Oxide is an active project that combines:

- a Rust host/runtime layer over HIP,
- runtime HIP kernel compilation and launch,
- and a Rust-authored device-kernel pipeline that produces launchable `.hsaco` artifacts.

---

## Start here

Two paths. Pick one.

### Path 1 — Quick smoke test (HIPRTC runtime compilation)

Compiles an inline HIP C++ kernel at runtime. No build pipeline, no nightly
`rust-src`, no extra tools — just a ROCm GPU and the Rust toolchain:

```sh
cargo run --example hello_gpu
```

Expected output:

```
hello_gpu: device 0 (gfx1100)
hello_gpu: 1048576 elements verified — vector add passed
```

**Annotated walkthrough →** [docs/hello_gpu.md](docs/hello_gpu.md)

---

### Path 2 — Main SDK path (Rust-authored GPU kernel)

Writes the kernel in Rust, compiles it to a `.hsaco` code object at build time,
and launches it with automatically generated typed host bindings.

This is the production vision: GPU code reviewed by `cargo`, typed, testable,
no HIP C++ required.

```sh
cargo run --example hello_gpu_rust
```

The first build takes 20–60 s while the device crate compiles. Subsequent builds
are incremental and fast.

Expected output:

```
hello_gpu_rust: device 0 (gfx1100)
hello_gpu_rust: 1048576 elements verified — Rust-authored kernel passed on gfx1100
```

**Annotated walkthrough →** [docs/hello_gpu_rust.md](docs/hello_gpu_rust.md)

---

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `can't find crate for 'core'` | `rustup component add rust-src --toolchain nightly` |
| `llc: command not found` | Add `/opt/rocm/bin` to `PATH` (or set `ROCM_PATH`) |
| No GPU detected | `sudo usermod -aG render,video $USER`, then log out and back in |
| Wrong GPU architecture | `ROCM_OXIDE_ARCH=gfx1100 cargo run --example hello_gpu_rust` |
| Stale HSACO after GPU change | `cargo clean ; cargo run --example hello_gpu_rust` |
| Slow first build (expected) | The device crate compiles with `-Z build-std=core`; caching kicks in on rebuild |

For a full prerequisites checklist, first-project scaffold, and common error
messages, see [docs/getting-started.md](docs/getting-started.md).

---

### Local scaffold (develop alongside this workspace)

> **Local scaffold only.** `cargo rocm-oxide new` creates a project that
> depends on this ROCm-Oxide workspace via relative `path` links. It is not a
> standalone project and cannot be published to crates.io. Generated projects
> and the workspace must be moved together to remain functional.
>
> **Full details →** [docs/project_generation.md](docs/project_generation.md)

Install the tool from the **repo root** once:

```sh
cargo install --path tools/cargo-rocm-oxide
```

Then, from within the ROCm-Oxide workspace:

**1. Check prerequisites**

```sh
cargo rocm-oxide doctor
```

This validates your ROCm installation, GPU visibility, and Rust nightly toolchain.

**2. Create a local scaffold project**

```sh
cargo rocm-oxide new my-gpu-project
cd my-gpu-project
```

The generated project contains:
- A `Cargo.toml` with a relative `path` dependency on `rocm-oxide`
- A `build.rs` that invokes `rocm-oxide-build` from the workspace
- A `rust-toolchain.toml` that pins nightly + `rust-src`
- A `README.md` explaining the scaffold's portability constraints
- A sample `#[kernel]` function and host program

**3. Build and run**

```sh
cargo run
```

This compiles your Rust kernel for `amdgcn-amd-amdhsa`, links a `.hsaco` code
object, and launches it on your AMD GPU.

**4. Verify a repo-wide build (source workspace only)**

```sh
# Run this from the ROCm-Oxide repo root, not from the generated project
cargo rocm-oxide verify --quick
```

Note: `verify` is a repository gate and only works inside the ROCm-Oxide source
tree. To verify your generated project builds, use `cargo build` inside it.

**Full walkthrough →** [docs/getting-started.md](docs/getting-started.md)

**API reference →** [docs/api_overview.md](docs/api_overview.md)

**Project generation details →** [docs/project_generation.md](docs/project_generation.md)

---

## What ROCm-Oxide does today

### Runtime path (host-side Rust)

- Compiles HIP kernels at runtime (HIPRTC / COMGR backend support)
- Loads GPU modules and launches kernels through Rust APIs
- Provides typed GPU buffers, streams, events, pinned memory, managed memory, and explicit graph helpers for empty/dependency, memcpy, memset, kernel, allocation/free, replay, and update paths

### Rust device-kernel path

- Compiles `#![no_std]` Rust device crates for `amdgcn-amd-amdhsa` (nightly)
- Rewrites emitted IR into launchable AMDGPU kernels where required
- Links code objects (`.hsaco`) and generates typed host bindings + metadata

---

## Current status

ROCm-Oxide is **not yet** a complete Rust-to-AMDGPU compiler stack, but it already demonstrates that:

1. Rust can cleanly own the ROCm runtime surface.
2. Rust-authored kernels can be compiled, packaged, and launched on AMD GPUs.

---

## Validated machine profiles

- **2026-06-01** — `gfx1100`, AMD Radeon RX 7900 XT, HIP `7.2.53211-364a905`, AMD clang `22.0.0git`
- **2026-05-31** — `gfx1201`, AMD Radeon RX 9070 XT, HIP `7.2.53211-364a905`, AMD clang `22.0.0git`

---

## Quick start

### Run the default demo

```bash
cargo run
```

### Force a target GPU architecture

```bash
ROCM_OXIDE_ARCH=gfx1100 cargo run
ROCM_OXIDE_ARCH=gfx1201 cargo run
```

If unset, ROCm-Oxide tries to detect the first `gfx*` target via `/opt/rocm/bin/rocminfo`.

### Run the "Hello GPU" example

```bash
cargo run --example hello_gpu
```

This is the recommended first example. It compiles an inline HIP C++ kernel at
runtime, runs a vector add on the GPU, and verifies the result. No separate
build step needed beyond ROCm and nightly Rust.

See [docs/hello_gpu.md](docs/hello_gpu.md) for a full walkthrough.

### Run the default demo

```bash
cargo run
```

### Force a target GPU architecture

```bash
ROCM_OXIDE_ARCH=gfx1100 cargo run
ROCM_OXIDE_ARCH=gfx1201 cargo run
```

If unset, ROCm-Oxide tries to detect the first `gfx*` target via `/opt/rocm/bin/rocminfo`.

### Run a simple HIP C++ example

```bash
cargo run --example vector_add
```

### Run a Rust-device example

```bash
cargo run --example rust_device_add_one
cargo run --example rust_device_vector_add
cargo run --example rust_device_generated_bindings
```

---

## Notable examples

- **`spectral_lattice`**: interactive visual GPU workbench (multiple render/compute paths, GUI controls, headless frame export, CPU/OpenGL/Vulkan present paths)
- **`matrix_lens`**: Vulkan-only pass-through lens demo that reads the monitor
  region under the window through a wlroots/GBM dma-buf path when available and
  otherwise crops the latest desktop video-stream frame before rendering
  matrix/glass/thermal/xray effects on the GPU
- **`compiler_feature_lab`**: GUI for probing compiler/runtime/device feature slices
- **`stress_test_gui` / `stress_3d_gui`**: interactive stress controls with
  bounded work-iteration presets
- **`performance_probe`**: emits timing/resource snapshots and JSON benchmark output

Example commands:

```bash
cargo run --example spectral_lattice
cargo run --example spectral_lattice -- --frames 3 --resolution 4k --fps-limit 120
cargo run --example spectral_lattice -- --present gl --resolution 1440p --fps-limit 120
cargo run --example spectral_lattice -- --present vulkan --resolution 1440p --fps-limit 120

cargo run --example matrix_lens -- --resolution 720p --mode matrix
cargo run --example matrix_lens -- --capture video --resolution 720p --mode matrix
cargo run --example matrix_lens -- --frames 30 --capture pattern --resolution 540p --output target/matrix_lens.png

cargo run --example compiler_feature_lab
cargo run --example compiler_feature_lab -- --frames 1

cargo run --example stress_test_gui
cargo run --example stress_3d_gui
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example stress_3d_gui
cargo run --example rainbow_geometry_window -- --present vulkan
cargo run --example raytrace_world_gui -- --present vulkan

cargo run --example performance_probe -- --json target/performance_probe.json
```

See [docs/visual-demos.md](docs/visual-demos.md) for a table of every visual
demo, its launch options, bounded-run environment variables, and presentation
path.

The smaller windowed visual demos (`rainbow_geometry_window`,
`raytrace_world_gui`, `stress_test_gui`, `stress_3d_gui`,
`possibilities_window`, `compiler_feature_lab`, and `window_effects_lab`) share
an example-only presenter. They keep the default `minifb` path and accept
`--present vulkan` or `ROCM_OXIDE_VISUAL_PRESENT=vulkan` for Vulkan swapchain
presentation. In Vulkan mode, GPU-rendered frames are copied device-to-device
into exportable Vulkan memory imported by HIP; demos with CPU UI overlays copy
only the overlay rectangles after the GPU frame, rather than reading the full
frame back to host.

The interactive stress examples clamp work-iteration controls to bounded safe
presets, currently no more than 4096 iterations per launch. Treat uncapped FPS or
resolution experiments as manual profiling only; regression runs should use
finite frame counts, explicit FPS limits, and the smallest resolution that
exercises the path under test.

`spectral_lattice --present vulkan` allocates exportable Vulkan device memory,
imports its `OPAQUE_FD` handle into HIP, and copies the rendered frame
device-to-device into that shared buffer before Vulkan blits it to the swapchain.
`--present gl` keeps the HIP/OpenGL pixel-buffer path, while the default remains
the CPU-readback compatibility path. GL and Vulkan presentation keep the frame
on the GPU and composite the same interactive controls through a small textured
overlay panel instead of reading the full frame back to the host every frame.
The Vulkan overlay panel is rasterized on a bounded worker and presents the
latest ready texture so CPU UI drawing does not block the GPU presentation path.
`matrix_lens` is Vulkan-only. On Wayland/wlroots it asks the compositor for the
window-sized region at the SDL window position as a GBM dma-buf, imports that
image into Vulkan, copies it into a HIP-imported Vulkan input buffer, and renders
directly into the HIP-imported Vulkan output buffer before presentation. If the
compositor or driver cannot provide that GPU path, the demo uses a persistent
xcap video recorder and crops the newest streamed frame instead of issuing
per-frame screenshots. Use `--capture auto|dmabuf|video|pattern` to force a
specific input path; `pattern` avoids compositor capture entirely for bounded
test runs.

---

## Rust API snapshot

```rust
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

let device = Device::first()?;
let module = device.compile_hip_source(KERNEL)?;
let kernel = module.kernel(c"vector_add")?;

let d_a = DeviceBuffer::from_slice(&a)?;
let d_b = DeviceBuffer::from_slice(&b)?;
let d_out = DeviceBuffer::<f32>::new(a.len())?;

unsafe {
    rocm_oxide::launch!(
        kernel,
        LaunchConfig::for_num_elems(a.len()),
        d_out.as_mut_ptr(),
        d_a.as_ptr(),
        d_b.as_ptr(),
        a.len() as u64,
    )?;
}
rocm_oxide::hip::synchronize()?;
let out = d_out.copy_to_vec()?;
```

Kernel launch remains `unsafe` because host code cannot fully prove ABI compatibility for raw kernel argument lists.

---

## Build pipeline and generated artifacts

The root `build.rs` generates device artifacts before host compilation and exposes:

- `ROCM_OXIDE_DEVICE_HSACO`
- `ROCM_OXIDE_DEVICE_BINDINGS`
- `ROCM_OXIDE_DEVICE_METADATA`

Manual compatibility script:

```bash
./scripts/compile-device-spike.sh
```

The build tool (`tools/rocm-oxide-build`) handles architecture detection, nightly device compilation, IR transformation, object generation/linking, and descriptor validation.

---

## Tooling commands

```bash
# Build tool diagnostics/inspection
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- --doctor
cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- --inspect-metadata device-spike/target/amdgcn-amd-amdhsa/release/rocm_oxide_device_spike.metadata.json

# Cargo subcommand wrapper
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide build
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide inspect
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide run --example rust_device_generated_bindings
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide pipeline
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide profile
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --host-ci
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --offline
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --quick
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --full
```

When installed as `cargo-rocm-oxide`, these become `cargo rocm-oxide ...`.

---

## Documentation map

### Core architecture and compiler path

- [DESIGN.md](DESIGN.md)
- [API stability](docs/api-stability.md)
- [Supported ROCm and GPU matrix](docs/supported-rocm-gpu-matrix.md)
- [Compiler path](docs/compiler-path.md)
- [Code object linking](docs/code-object-linking.md)
- [Toolchain discovery](docs/toolchain-discovery.md)
- [Unsafe and FFI audit](docs/unsafe-audit.md)

### Runtime semantics and memory model

- [Atomic scopes](docs/atomic-scopes.md)
- [Host-memory coherence](docs/host-memory-coherence.md)
- [Stream-ordered allocation](docs/stream-ordered-allocation.md)
- [Unsafe and FFI audit](docs/unsafe-audit.md)
- [ROCm library interop](docs/rocm-library-interop.md)

### SDK direction and roadmap

- [SDK direction](docs/sdk_direction.md) — current state, product identity, architecture layers, what to build next, and phased roadmap
- [CUDA-Oxide parity checklist](docs/cuda-oxide-parity-checklist.md)
- [ROCm feature parity notes](docs/rocm-feature-parity.md)
- [CUDA book adaptation notes](docs/cuda-oxide-book-rocm-adaptation.md)
- [Future work](docs/cuda-future-work.md)
- [Implementation tasks](docs/implementation-tasks.md)
- [Production readiness](docs/production-readiness.md)
- [CI and release gates](docs/release-gates.md)

### Debugging

- [Debugger workflow](docs/debugger-workflow.md)

### Project and release

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [MIT license](LICENSE-MIT)
- [Apache-2.0 license](LICENSE-APACHE)

---

## Roadmap focus

The current objective is production readiness. ROCm-Oxide should keep practical
parity with the user-facing CUDA-Oxide feature set, but new work now needs to
harden APIs, safety contracts, diagnostics, validation, and release practices.

High-priority areas:

- repeatable verification through `cargo rocm-oxide verify`;
- public API stability boundaries;
- unsafe/FFI safety contracts and negative tests;
- cross-machine validation for `gfx1100` and `gfx1201`;
- actionable diagnostics for toolchain, ABI, HIP, COMGR, and optional-library
  failures.

See:

- [CUDA-Oxide supported features matrix](https://nvlabs.github.io/cuda-oxide/appendix/supported-features.html)
- [Parity checklist](docs/cuda-oxide-parity-checklist.md)
- [Production readiness](docs/production-readiness.md)

---

## Repository layout

- `src/` — host runtime/library
- `crates/` — device/runtime support crates
- `examples/` — demos, feature labs, and probes
- `device-spike/` — Rust-authored device kernel spike
- `tools/` — build + cargo tooling wrappers
- `docs/` — detailed design, parity, and runtime docs

---

## Notes

- Nightly Rust is required for the Rust device-kernel path (`rust-toolchain.toml` pins toolchain expectations).
- Stable Rust alone cannot currently provide `core` for `amdgcn-amd-amdhsa`.
