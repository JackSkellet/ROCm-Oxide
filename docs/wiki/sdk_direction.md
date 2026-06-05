# ROCm-Oxide SDK Direction

This document records the current state of the repository, defines the intended
product identity, maps the architecture into layers, and lays out a phased
roadmap toward a polished Rust GPU Kernel SDK for AMD/ROCm.

---

## 1. Current State

ROCm-Oxide is past proof-of-concept. The two major pipelines are both functional
and verified on `gfx1100` and `gfx1201`.

### 1.1 Runtime pipeline (HIP host-side)

| Capability | Source location |
|---|---|
| Device enumeration and arch detection | `src/runtime.rs` — `Device::first/at/all`, `detect_arch` via `rocminfo` |
| Device properties and limits | `src/runtime.rs` — `DeviceProperties`, `DeviceLimits` |
| GPU memory allocation | `src/hip.rs` — `DeviceBuffer<T>`, `ManagedBuffer<T>`, `PinnedHostBuffer`, `DeviceVirtualMemory`, `MemPool` |
| Streams and events | `src/hip.rs` — `Stream`, `Event`, stream capture modes |
| HIP graph execution | `src/hip.rs` — empty/dependency, 1D memcpy, typed H2D/D2H/D2D, memset, kernel, alloc/free nodes; instantiate/replay/update |
| HIPRTC runtime compilation | `src/hiprtc.rs` — `SpecializationCache`, cached `.hsaco` emission |
| COMGR compilation backend | `src/hiprtc.rs` + `src/libraries.rs` — alternate backend in specialization cache |
| `.hsaco` loading | `src/runtime.rs` — `Device::load_code_object`, `load_code_object_file` |
| Kernel launch | `src/runtime.rs` + `src/lib.rs` — `Kernel`, `launch!` macro, `LaunchConfig`, cooperative launch |
| Operation composition | `src/operation.rs` — `DeviceOperation` trait, `ExecutionContext`, `StreamPool`, `CapturedGraph`, `DeviceFuture` |
| Optional ROCm libraries | `src/libraries.rs` — rocBLAS, rocFFT, hipBLASLt, COMGR, rocPRIM/hipCUB (dynamic `dlopen`) |
| Profiling markers | `src/profiling.rs` — rocTX marker/range wrappers |
| CUDA concept mapping | `src/parity.rs` — `RocmFeaturePlan`, `RocmFeatureSet`, interop plan |
| Peer access | `src/runtime.rs` + `src/hip.rs` — `can_access_peer`, `enable_peer_access`, `disable_peer_access` |

### 1.2 Rust device-kernel pipeline

| Capability | Source location |
|---|---|
| `#[kernel]` proc-macro with monomorphization | `crates/rocm-oxide-kernel/src/lib.rs` |
| `#[device_global]`, `#[constant]`, `#[shared]` statics | `crates/rocm-oxide-kernel/src/lib.rs` |
| `#![no_std]` device support library | `crates/rocm-oxide-device/src/lib.rs` — math, scoped atomics, LDS slice helpers, dispatch-ptr intrinsics |
| Build tool (`rocm-oxide-build`) | `tools/rocm-oxide-build/src/main.rs` — kernel discovery, LLVM IR rewrite, `llc` lowering, AMDGPU `clang` link, metadata/bindings/manifest emission |
| Cargo wrapper (`cargo-rocm-oxide`) | `tools/cargo-rocm-oxide/src/main.rs` — `doctor`, `build`, `run`, `debug`, `inspect`, `pipeline`, `profile`, `verify`, `new` sub-commands |
| Build-script integration | `build.rs` — invokes `rocm-oxide-build`, copies HSACO + generated bindings into `OUT_DIR`, sets `ROCM_OXIDE_DEVICE_HSACO` and `ROCM_OXIDE_DEVICE_BINDINGS` env vars |
| Kernel length and disjointness contracts | `docs/kernel-contracts.md` — `// rocm-oxide: len(...)=...` / `disjoint(...)` parsed by build tool |
| Concrete device crate / test kernels | `device-spike/src/lib.rs` — `add_one`, `vector_add`, math intrinsics, LDS, stress kernels |

### 1.3 Examples and tests

Root SDK examples stay in `examples/`; larger visual/capture/artifact demos are
separated crates under `demo-projects/`.

| Name | Kind | Notes |
|---|---|---|
| `vector_add` | HIP source | Compiles C++ HIP inline via HIPRTC |
| `rust_device_add_one` | Rust device | Loads embedded HSACO, raw `launch!` |
| `rust_device_vector_add` | Rust device | Generated HSACO, raw `launch!` |
| `rust_device_generated_bindings` | Rust device | Uses `DeviceKernels` generated struct for typed launch |
| `demo-projects/spectral-lattice` | Visual / headless | Multi-path render/compute, GUI controls, PNG export |
| `demo-projects/matrix-lens` | Visual | Vulkan lens demo, GBM dma-buf, wlroots capture |
| `demo-projects/compiler-feature-lab` | Feature probe GUI | Compiler/runtime/device feature slices |
| `performance_probe` | Benchmark | Timing/resource JSON output |
| `validation_profile` | Verification artifact | ROCm version, device caps, library availability |
| `pinned_stream_vector_add` | Runtime | Pinned host memory + stream ordering |
| `module_global` | Runtime | Device global read/write |
| `feature_showcase` | Smoke | Tests many runtime features in one run |
| `crates/rocm-oxide-kernel/tests/smoke.rs` | Proc-macro | Compile-time check of attribute macros |

### 1.4 Infrastructure

- **Verification gates**: `scripts/verify.sh` — `--host-ci`, `--offline`, `--quick`, `--full` profiles
- **Consumer smoke test**: `scripts/consumer-smoke.sh` — compiles a temporary downstream crate against public `rocm_oxide::*`
- **Toolchain doctor**: `cargo rocm-oxide doctor` / `rocm-oxide-build --doctor`
- **API classification**: `docs/api-stability.md` — stable / experimental / internal tiers
- **Safety audit**: `docs/unsafe-audit.md` — FFI contracts for all public unsafe surfaces
- **Validated machine profiles**: `gfx1100` (RX 7900 XT) and `gfx1201` (RX 9070 XT)

---

## 2. Intended Product Identity

ROCm-Oxide should become the **Rust-native SDK for writing and running GPU
kernels on AMD/ROCm hardware** — not a HIP binding layer, and not a
CUDA-compatibility shim.

The product contract is:

- **Rust in, GPU out.** Kernels are authored in `#![no_std]` Rust, not C++,
  and the toolchain produces verified `.hsaco` artifacts.
- **Safe by default.** Host code that interacts with device memory, streams, and
  graphs should be safe Rust unless the caller is intentionally bypassing a
  validated contract.
- **AMD-first.** The design vocabulary is ROCm: workgroups, wavefronts, LDS,
  HSACO, COMGR, rocBLAS, rocFFT. There is no CUDA ABI compatibility promised.
- **Toolchain-honest.** The SDK surfaces what the AMD toolchain actually
  supports today, validated against real GPUs, not projected from CUDA's
  feature list.
- **Cargo-native.** Device crates are ordinary Rust crates. `cargo rocm-oxide
  build` wraps the extra compilation step; host projects stay in normal Cargo
  workspace conventions.

---

## 3. Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  User application / example                             │
│  (Cargo workspace, normal Rust)                         │
├─────────────────────────────────────────────────────────┤
│  SDK layer (this is the gap to fill)                    │
│  - typed kernel handles with safe launch signatures     │
│  - buffer/view types with Rust lifetime tracking        │
│  - allocation scopes and memory-kind policies           │
│  - ergonomic stream/graph composition                   │
│  - device-crate project scaffold                        │
├─────────────────────────────────────────────────────────┤
│  Generated layer (build tool output)                    │
│  - DeviceKernels struct (typed launch methods)          │
│  - kernel metadata JSON                                 │
│  - manifest JSON (link provenance)                      │
│  - Rust host bindings (include! into user crate)        │
├─────────────────────────────────────────────────────────┤
│  Kernel authoring layer                                 │
│  - rocm-oxide-kernel (proc-macros)                      │
│  - rocm-oxide-device (no_std device library)            │
│  - device-spike (reference device crate)                │
│  - rocm-oxide-build (compiler pipeline)                 │
│  - cargo-rocm-oxide (workspace tool)                    │
├─────────────────────────────────────────────────────────┤
│  Runtime layer (already solid)                          │
│  - src/hip.rs         (HIP FFI wrappers)                │
│  - src/hiprtc.rs      (HIPRTC / COMGR backends)         │
│  - src/runtime.rs     (Device, Module, Kernel, launch)  │
│  - src/operation.rs   (streams, graphs, futures)        │
│  - src/libraries.rs   (rocBLAS, rocFFT, hipBLASLt…)     │
│  - src/profiling.rs   (rocTX)                           │
│  - src/parity.rs      (ROCm concept mapping)            │
├─────────────────────────────────────────────────────────┤
│  AMD/ROCm system (not owned by this project)            │
│  HIP runtime, HIPRTC, COMGR, llc, clang, rocminfo…     │
└─────────────────────────────────────────────────────────┘
```

---

## 4. What Belongs in the SDK Layer

The SDK layer is the surface that a user who is **not** reading ROCm-Oxide
internals should interact with. It sits between the generated bindings and the
user's application code.

**Must land before SDK label is accurate:**

- **Documented `rocm-oxide-kernel` proc-macro API.** `#[kernel]`,
  `#[device_global]`, `#[constant]`, and `#[shared]` need user-facing doc
  comments with examples. Currently the crate has no rustdoc.
- **Documented `rocm-oxide-device` library API.** The `math` submodule, scoped
  atomics, LDS slice helpers, and dispatch intrinsics are real user-facing
  primitives with no public documentation.
- **`cargo rocm-oxide new` scaffold.** The command exists in
  `tools/cargo-rocm-oxide/src/main.rs` but its output is not yet documented.
  It should emit a minimal working device crate + host binary so a new user can
  reach their first kernel in under five minutes.
- **SDK getting-started guide.** A `docs/getting-started.md` walking from
  `cargo rocm-oxide new` through `cargo rocm-oxide build` to a running kernel.
- **Public crate publishability.** `crates/rocm-oxide-kernel` and
  `crates/rocm-oxide-device` are `publish = false`. For SDK use they should be
  publishable from `crates.io` (or clearly explained as workspace-only).

**Nice-to-have at SDK milestone:**

- **Typed dynamic LDS allocation.** `#[shared]` statics cover the static case.
  A typed wrapper around `gpu_launch_sized_workgroup_mem` would cover the
  dynamic case without requiring raw pointer casts.
- **Warp/wavefront reduction and scan library in `rocm-oxide-device`.** The
  intrinsics are there; a small typed API over them (`warp_reduce_sum`,
  `warp_ballot`, etc.) would remove boilerplate from every kernel.
- **GPU testing harness.** A minimal `#[gpu_test]` attribute or equivalent so
  device code can be unit-tested without writing a full host example.

---

## 5. What Belongs in the Runtime Layer

The runtime layer is already the strongest part of the project. It should
remain focused on:

- **Safe, well-documented wrappers over HIP, HIPRTC, COMGR**, with explicit
  lifetime and safety contracts (continued from `docs/unsafe-audit.md`).
- **Graph and stream composition** through `src/operation.rs`, keeping
  `DeviceOperation`, `CapturedGraph`, and `StreamPool` as the primary
  user-facing async primitives.
- **Optional library interop** (`src/libraries.rs`) staying behind availability
  probes — never assumed present, always a graceful degradation path.
- **Specialization cache** (`src/hiprtc.rs`) — HIPRTC and COMGR compilation
  paths for users who prefer HIP source kernels or for runtime-generated
  specializations.
- **Cooperative launch, peer access, VMM** — AMD-specific hardware capabilities
  that belong here, not in application code.

The runtime layer should **not** grow direct dependencies on the proc-macro or
build-tool crates. The current direction (generated bindings are emitted as Rust
source files, not linked libraries) is correct.

---

## 6. What Should Not Be Built Yet

These areas either have active design risk, depend on upstream Rust features not
yet stable, or would complicate the core SDK story without clear user demand:

- **Async-Rust executor integration for GPU work.** `DeviceFuture` in
  `src/operation.rs` is a sketch. Integrating with `tokio` or `async-std` at
  the GPU stream level is a real design problem and should not be standardized
  until there is a concrete use-case driving the API shape.
- **Multi-GPU scheduling / topology-aware dispatch.** `Device::all()` and peer
  access exist. A higher-level multi-GPU task graph is premature before the
  single-device SDK surface is ergonomic.
- **CUDA binary/PTX compatibility.** `src/parity.rs` maps CUDA concepts to
  ROCm replacements at the source level. Accepting PTX, cubin, LTOIR, or NVVM
  as runtime inputs is out of scope and explicitly rejected in
  `docs/code-object-linking.md`.
- **A `std`-enabled device crate.** The `amdgcn-amd-amdhsa` target does not
  have a `std` implementation; `#![no_std]` is the correct model and should
  stay that way until upstream Rust changes.
- **Texture/image memory.** HIP texture objects exist but are not surfaced here.
  This is a valid future extension but should not block SDK progress.
- **Hot-reload / live kernel patching.** The HIPRTC specialization cache
  supports recompilation, but a user-facing hot-reload API needs a clear
  deployment story before it is worth designing.
- **Composable Kernel / rocWMMA execution wrappers.** `src/parity.rs` reports
  these as candidates. They should remain candidate reports until a maintainer
  can validate the execution wrapper on real hardware.

---

## 7. Phased Roadmap

### Phase 0 — Foundation (done)

- HIP runtime wrappers: device, memory, streams, events, graphs ✓
- HIPRTC and COMGR compilation backends ✓
- `#[kernel]` proc-macro with monomorphization ✓
- `#![no_std]` device library (math, atomics, LDS, dispatch) ✓
- `rocm-oxide-build` full pipeline: IR → llc → clang → HSACO + metadata + bindings ✓
- `cargo-rocm-oxide` tool with doctor, build, verify, pipeline ✓
- Production verification gates (offline, quick, full) ✓
- Safety and FFI audit of all public surfaces ✓
- Validated on `gfx1100` and `gfx1201` ✓

### Phase 1 — SDK Documentation and Ergonomics (next)

Goal: a new user can write, build, and run a Rust GPU kernel without reading
source code.

- [ ] Write rustdoc for every public item in `crates/rocm-oxide-kernel`.
- [ ] Write rustdoc for every public item in `crates/rocm-oxide-device`
  (especially `math`, `atomics`, and LDS helpers).
- [ ] Write `docs/getting-started.md`: from `cargo rocm-oxide new` to a running
  kernel.
- [ ] Document `cargo rocm-oxide new` output: what files are generated, how to
  add a kernel, how to call it from the host.
- [ ] Publish `rocm-oxide-kernel` and `rocm-oxide-device` to `crates.io` or
  clearly document the workspace-only rationale.
- [ ] Add at least one end-to-end integration test that exercises
  `cargo rocm-oxide new` output in CI.

### Phase 2 — Kernel Authoring Quality

Goal: writing device code in Rust feels idiomatic, not like wrapping C ABI.

- [ ] Typed dynamic LDS wrapper over `gpu_launch_sized_workgroup_mem`.
- [ ] Wavefront reduction/scan library in `rocm-oxide-device` (`warp_reduce_*`,
  `warp_scan_*`, `warp_ballot`).
- [ ] Minimal GPU testing harness — `#[gpu_test]` or equivalent, running device
  assertions without a host example wrapper.
- [ ] Extend `#[kernel]` to support `DeviceSlice<T>` / `DeviceSliceMut<T>`
  arguments directly (currently raw pointer + length pairs).
- [ ] Kernel contract coverage for shared-memory size (`lds`) expressions.

### Phase 3 — SDK Polish and Release

Goal: stable API, published crates, usable as a library dependency.

- [ ] Promote root `rocm_oxide::*` re-exports to stable per `docs/api-stability.md`
  rules.
- [ ] Tag `0.2.0` with changelog, migration guide from `0.1`, and GPU-validated
  release artifacts.
- [ ] Publish root crate to `crates.io`.
- [ ] GitHub Actions GPU runner (self-hosted) producing `gfx1100` and `gfx1201`
  release artifacts automatically on tag.
- [ ] `cargo rocm-oxide verify --release-candidate` gate that requires all three
  GPU profiles and consumer smoke test before promotion.

### Phase 4 — Ecosystem Extensions (future, no design yet)

These are deliberately vague until Phase 3 is done:

- Typed dynamic LDS tuning API (template-parameter-style LDS sizing from host).
- Async-Rust GPU stream integration (design-first before code).
- Multi-device dispatch helpers built on `Device::all()` + peer access.
- Texture / image memory wrappers.

---

## 8. Files Changed by This Document

This is a planning and documentation PR. No source code was modified.

New file: `docs/sdk_direction.md` (this file)
Updated file: `README.md` — added a short pointer to this document.
