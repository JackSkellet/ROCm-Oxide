# Stability Policy

ROCm-Oxide is an **experimental SDK** at `0.x` version. This document defines
what "experimental" means in practice, what you can rely on, and what may
change.

---

## What "Experimental" Means

- ROCm-Oxide is not production-ready.
- APIs may break in any `0.x` minor release without a deprecation period.
- Generated bindings may change format between releases.
- The scaffold (`cargo rocm-oxide new`) produces local-path-dependent projects
  that are not portable to crates.io.
- There is no LTS branch. No backport policy. No security patch commitment
  beyond best-effort.

If you build on ROCm-Oxide today, pin an exact commit or tag. A minor version
bump may require code changes.

---

## Stability Tiers

### Tier 1 — More Stable (runtime host API)

The root `rocm_oxide::*` re-exports listed below have the most inertia. They
may still change, but changes will come with a migration note in `CHANGELOG.md`:

- `Device`, `Module`, `Kernel`
- `DeviceBuffer<T>`, `ManagedBuffer<T>`, `PinnedHostBuffer`
- `Stream`, `Event`
- `LaunchConfig`, `Dim3`
- `launch!`, `launch_1d!`, and `launch_1d_with_block!` macros
- `Error`, `Result`
- `DevicePod`

### Tier 2 — Experimental (public but not stable)

These are public for examples, generated bindings, and feature exploration.
Shape and names may change without migration notes:

- `hip::*` — raw HIP wrappers, graph APIs, VMM, memory pools
- `hiprtc::*` — runtime compiler and specialization cache
- `libraries::*` — rocBLAS, rocFFT, hipBLASLt, COMGR, rocPRIM interop
- `operation::*` — lazy stream/graph composition
- `profiling::*` — rocTX markers
- `parity::*` — CUDA concept mapping

### Tier 3 — No Stability (tooling and generated artifacts)

No API stability guarantees:

- `tools/rocm-oxide-build` — internal build pipeline tool
- `tools/cargo-rocm-oxide` — `cargo rocm-oxide` subcommands
- `crates/rocm-oxide-device` — `no_std` device support library
- `crates/rocm-oxide-kernel` — `#[kernel]` proc macro
- Generated `DeviceKernels` struct and method signatures
- Generated `bindings.rs` format
- `build.rs` env-var protocol (`ROCM_OXIDE_*`)
- Scaffold project layout from `cargo rocm-oxide new`

---

## Generated Bindings Stability

The `rocm-oxide-build` tool generates a `DeviceKernels` struct with typed
launch methods from `#[kernel]`-annotated Rust functions. This format is
**not stable**:

- Method signatures may change (e.g., if new argument kinds are added).
- The generated `impl` block structure may change.
- The `load_embedded` method name may change.
- The env vars used to locate the bindings may change.

When upgrading ROCm-Oxide, regenerate bindings by running `cargo clean` in
your scaffold project and rebuilding.

---

## crates.io

ROCm-Oxide is **not published to crates.io** and has no current plan to be.
Reasons:

- The crate depends on ROCm shared libraries (`libamdhip64.so`, `libhiprtc.so`)
  that cannot be bundled in a crate.
- Generated scaffold projects use local `path` dependencies that are
  incompatible with crates.io.
- The API is not stable enough to warrant a registry publication.

Use a Git dependency or local path dependency in your own projects:

```toml
[dependencies]
rocm-oxide = { path = "../ROCm-Oxide" }
```

---

## What Counts as a Breaking Change

For Tier 1 APIs, these are breaking changes that will appear in `CHANGELOG.md`:

- Renaming or removing a public type or function
- Changing a function signature in a backwards-incompatible way
- Changing the memory layout of a public type
- Changing the `launch!`, `launch_1d!`, or `launch_1d_with_block!` macro syntax

For Tier 2 and Tier 3, any change may happen silently. Check `CHANGELOG.md`
and `git log` when upgrading.

---

## Reporting Regressions

If an upgrade breaks your code and the breakage is in Tier 1, please file an
issue. Use the Bug Report template and note which API was broken and which
versions were affected.
