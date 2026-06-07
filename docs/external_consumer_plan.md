# External Consumer Plan

This document records how downstream projects can consume ROCm-Oxide today and
what must change before the SDK can support normal crates.io-style dependency
flows.

## Current Consumer Options

### Local scaffold

The most complete preview path is still a generated local scaffold:

```sh
cargo install --path tools/cargo-rocm-oxide
cargo rocm-oxide new ../my-project
cd ../my-project
cargo rocm-oxide check-consumer
cargo run
```

The generated project contains a host crate, a local `device-spike/` device
crate, `rust-toolchain.toml`, editor tasks, and a `build.rs` that invokes
`rocm-oxide-build`. Its `Cargo.toml` files use relative `path` dependencies back
to the ROCm-Oxide source workspace.

Local scaffolds work well for development beside the SDK checkout. They are not
standalone: moving only the downstream project or only the ROCm-Oxide workspace
breaks the relative paths.

### Git scaffold

For projects that should not keep a fixed local path to ROCm-Oxide, use the git
scaffold mode:

```sh
cargo install --git https://github.com/JackSkellet/ROCm-Oxide rocm-oxide-build
cargo rocm-oxide new my-project --git https://github.com/JackSkellet/ROCm-Oxide --rev <commit>
cd my-project
cargo rocm-oxide check-consumer
cargo run
```

This assumes the `cargo-rocm-oxide` wrapper is already installed from a
ROCm-Oxide checkout. Git scaffolds put git dependencies in generated manifests,
but generated builds still need the `rocm-oxide-build` binary through
`ROCM_OXIDE_BUILD` or `PATH`.

### Manual setup

Manual project creation is not currently supported. Hand-written manifests and
build scripts are easy to get subtly wrong because the host crate, device crate,
nightly toolchain, build tool, generated bindings, and ROCm tools all have to
line up.

## Local Path Dependency Limitations

Local scaffolds currently depend on:

- `rocm-oxide` through a relative path from the generated host crate.
- `rocm-oxide-device` through a relative path from the generated device crate.
- `rocm-oxide-kernel` through a relative path from the generated device crate.
- `rocm-oxide-build` through the source workspace, `ROCM_OXIDE_BUILD`, or
  `PATH`.

This means a downstream repo can be developed locally, but it cannot yet be
published as a normal crates.io consumer or cloned into an arbitrary layout
without preserving the expected dependency locations.

## Local Scaffold Status

The scaffold generator is the honest preview workflow. It creates the required
host/device split, inserts the correct Rust toolchain file, wires the generated
bindings path, and provides `cargo rocm-oxide check-consumer` as a first
diagnostic step.

Generated scaffolds should be treated as source-workspace consumers. They are a
good starting point for downstream demos and application repos, but not a final
distribution format.

## Before crates.io Publication

Before ROCm-Oxide can support normal crates.io dependency flows, the project
needs:

- Publishable crate metadata, license/readme/package filtering, and dependency
  hygiene for each public crate.
- A distribution story for `rocm-oxide-build`, either as a binary crate or
  release artifact.
- Generated `build.rs` templates that no longer require a source-workspace
  `RUNTIME_PATH`.
- Generated manifests that use version dependencies instead of local paths.
- CI/release gates that validate a generated project without a local workspace
  path.
- Clear docs for ROCm system requirements and unsupported platforms.

The current repository should not claim crates.io support until those items are
complete.

## Likely Publishing Order

1. `rocm-oxide-kernel`: proc-macro crate for kernel attributes and contracts.
2. `rocm-oxide-device`: `#![no_std]` device support crate used by device
   kernels.
3. `rocm-oxide`: host runtime crate that links to ROCm/HIP at runtime.
4. `rocm-oxide-build`: build tool, published as a binary crate or distributed
   through release artifacts.
5. `cargo-rocm-oxide`: cargo wrapper for generation, diagnostics, and
   verification commands.

`rocm-oxide-kernel` should be published before `rocm-oxide-device` only if the
device crate or generated downstream device crates depend on the proc macro by
version. The root runtime crate should come after the supporting device crates
so generated projects can resolve their full dependency graph.

## Public-Facing And Internal Packages

Public-facing packages:

- `rocm-oxide`: host runtime, memory, launch, stream/event/graph, HIPRTC, COMGR,
  generated-binding support, and high-level helper APIs.
- `rocm-oxide-device`: device-side support library for Rust-authored kernels.
- `rocm-oxide-kernel`: proc macros such as `#[kernel]` and
  `#[kernel_contract(...)]`.

Tooling packages:

- `rocm-oxide-build`: compiler pipeline, HSACO generation, metadata generation,
  and generated binding emission.
- `cargo-rocm-oxide`: cargo subcommand for doctor, build, verify, pipeline,
  profile, and scaffold generation.

Internal or workspace-only surfaces:

- `device-spike/`: source-workspace reference device crate and test kernels.
- `demo-projects/shared/device_build.rs`: helper for repository demo crates.
- `scripts/`: repository verification and smoke-test scripts.
- `docs/wiki/`: long-form design notes and historical implementation records.
- Private root modules such as `runtime` internals and `__private` macro
  helpers.

## Downstream Project Checklist

For a downstream repo such as `rocm-oxide-sim`:

- Generate the starting point with `cargo rocm-oxide new` rather than
  hand-writing the scaffold.
- Keep `rust-toolchain.toml` with nightly and `rust-src`; Rust-authored device
  kernels require the AMDGPU target and standard library source.
- Install ROCm and ensure GPU access, including `/dev/kfd` permissions.
- Run `cargo rocm-oxide doctor` from the ROCm-Oxide source workspace for local
  scaffolds, or from the generated project for git scaffolds.
- Run `cargo rocm-oxide check-consumer` immediately after generation and after
  moving either repository.
- Keep `build.rs` wired to the generated device crate and
  `rocm-oxide-build`; do not replace it with a generic Cargo build script.
- Include generated bindings from the path emitted by the build tool instead of
  copying generated files into source control.
- Preserve local `path` dependency layout, or use `--git` plus installed
  `cargo-rocm-oxide` and `rocm-oxide-build` binaries.
- Keep simulation, scene, physics, dataset, ROS2, and app UX code in the
  downstream repo. Add only reusable SDK primitives back to ROCm-Oxide.
