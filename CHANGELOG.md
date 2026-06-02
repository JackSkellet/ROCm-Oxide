# Changelog

All notable ROCm-Oxide changes should be recorded here before a tagged release.

## Unreleased

- Added production-readiness gates through `scripts/verify.sh` and
  `cargo rocm-oxide verify` with offline, quick GPU, and full GPU profiles.
- Added checked validation profiles for the local `gfx1100` and `gfx1201`
  machines, including ROCm capability differences and skipped-test reasons.
- Hardened host/runtime safety contracts for raw launches, graph nodes, VMM,
  stream-ordered memory, module-owned function/global lifetimes, optional ROCm
  libraries, and generated device helpers.
- Added negative tests for graph dependency misuse, graph allocation/free
  ordering, VMM validation, rocPRIM temporary storage, COMGR/HIPRTC cache keys,
  and hipBLASLt descriptor inputs.
- Added release gates, API stability notes, diagnostics hardening, and release
  basics for the first production-oriented development phase.

## 0.1.0

- Initial local prototype version for Rust-hosted ROCm/HIP runtime work,
  runtime HIPRTC/COMGR compilation, generated Rust device bindings, examples,
  and CUDA-Oxide parity exploration.
