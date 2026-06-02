# API Stability

ROCm-Oxide is still a `0.x` crate. This document defines the intended API
boundaries before downstream projects start depending on every public symbol by
accident.

## Stable User-Facing Surface

The root `rocm_oxide::*` re-exports are the preferred user-facing API for host
runtime work:

- device discovery and loading: `Device`, `Module`, `Kernel`;
- launch configuration and validation: `Dim3`, `LaunchConfig`,
  `LaunchRecommendation`, `KernelMetadata`, `KernelResource`, and validation
  helpers used by generated bindings;
- memory and synchronization: `DeviceBuffer`, `ManagedBuffer`,
  `PinnedHostBuffer`, `DevicePod`, `Stream`, `Event`;
- launch entry point: `launch!`;
- shared error/result types: `Error`, `Result`.

These names should be kept source-compatible where practical. Breaking changes
need migration notes and should be tied to a production-readiness reason.

## Experimental Public Surface

These modules and re-exports are public for examples, generated bindings, and
ROCm feature exploration, but their exact shape is not stable yet:

- `hip`: low-level HIP wrappers, explicit graph APIs, memory pools, VMM, module
  globals, and raw launch escape hatches;
- `hiprtc`: runtime compiler backends and specialization cache internals;
- `libraries`: optional ROCm library interop for rocBLAS, rocFFT, rocPRIM,
  hipBLASLt, COMGR, and related handles/descriptors;
- `operation`: lazy stream/graph operation composition;
- `profiling`: rocTX marker/range support;
- `parity`: CUDA-to-ROCm planning structs and feature reports.

Experimental APIs must still maintain memory-safety contracts and deterministic
diagnostics, but they can be renamed or reshaped before a stable release.

## Internal Surface

The private `runtime` module and `__private` macro helpers are crate
implementation detail. Build-tool-generated bindings should prefer documented
root re-exports over reaching into private helpers. Tools under `tools/` and
device support crates under `crates/` have their own package boundaries and are
not part of the root crate's stable API.

## Stabilization Rules

Before promoting an experimental API:

- document every public `unsafe` function with a precise safety contract and
  keep [Unsafe and FFI audit](unsafe-audit.md) current;
- add invalid-input or invalid-order tests for the relevant FFI/lifetime edge;
- verify the API through `cargo rocm-oxide verify --quick` on a live ROCm
  machine;
- record any GPU/ROCm capability assumptions in the validation artifacts or
  release notes.
