# Production Readiness

ROCm-Oxide has moved past proof-of-concept work. New changes should be judged
against whether they make the runtime, compiler pipeline, and examples reliable
enough for other projects to build on.

## Verification Gate

The canonical local gate is:

```bash
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --offline
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --quick
cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide verify --full
```

The direct script remains available for CI and shell use:

```bash
scripts/verify.sh --offline
scripts/verify.sh --quick
scripts/verify.sh --full
```

All profiles write logs and artifacts under
`target/production-readiness/`.

The offline profile covers:

- formatting checks;
- verification script syntax;
- proc-macro, build-tool, and cargo-wrapper tests that do not need a live GPU.

The quick profile covers:

- host/runtime unit tests;
- proc-macro and build-tool tests;
- doctor and pipeline inspection;
- core generated-binding GPU smoke coverage;
- `feature_showcase`;
- `performance_probe` JSON output;
- one headless `spectral_lattice` visual artifact.

The full profile adds the heavier examples and all headless `spectral_lattice`
mode artifacts, including the 4K path.

GPU profiles run Rust tests with one test thread. HIP, COMGR, graph, VMM, and
optional-library tests can share process-wide runtime state, and production
verification should avoid relying on default test parallelism.

## CI Gates

The standard GitHub Actions workflow runs `scripts/verify.sh --offline` on
pull requests and pushes to `main`.

The manual GPU workflow runs quick and full verification on self-hosted Linux
ROCm runners labeled:

- `self-hosted`, `linux`, `rocm`, `gfx1100`;
- `self-hosted`, `linux`, `rocm`, `gfx1201`.

GPU jobs upload the `target/production-readiness/` directory so logs, JSON
performance output, and headless visual artifacts are available from the run.

## Hardening Priorities

1. Safety audit:
   - every public `unsafe` API needs a precise safety contract;
   - every FFI wrapper needs ownership, lifetime, and thread-safety notes;
   - graph allocations, VMM mappings, COMGR/HIPRTC cache entries, rocPRIM
     temporary storage, and hipBLASLt handles need invalid-order tests.
   - host-visible zeroed buffers use `DevicePod` so safe Rust slices are only
     exposed for plain-data types with no hidden validity or ownership
     invariants.
   - stream-enqueued async allocation, copy, and memset APIs are `unsafe` when
     the caller must keep buffers, streams, memory pools, or output slices alive
     until the stream reaches the work.
2. API stability:
   - decide which modules are stable public API, experimental public API, or
     crate-private implementation detail;
   - hide or rename unstable helpers before downstream users depend on them;
   - document feature gates and capability probes for optional ROCm paths.
3. Cross-machine validation:
   - keep separate `gfx1100` and `gfx1201` capability profiles;
   - record exact ROCm version, architecture, supported libraries, skipped
     tests, and topology limits for every validation machine;
   - never promote a feature based on a single GPU profile when ROCm reports a
     relevant capability difference.
4. Diagnostics:
   - errors should name the missing tool, bad ABI, unsupported memory kind,
     failed library symbol, or unsupported HIP/COMGR feature;
   - user-facing commands should suggest the next command to run.
5. CI and release:
   - CPU-only checks should run without a GPU;
   - GPU checks should use the verification gate;
   - release notes should include supported ROCm versions, GPU architectures,
     feature matrix status, and known limitations.

## Current Non-Negotiables

- No CUDA binary compatibility promise.
- No PTX, cubin, NVVM, TMA, WGMMA, or DSMEM ABI promise.
- CUDA-only concepts remain source-level rewrite targets with ROCm-native
  implementations.
- Demos are regression coverage, not the product boundary.
