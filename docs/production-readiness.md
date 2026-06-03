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

Each verifier command is wrapped in a per-command timeout, defaulting to 1200
seconds. Set `ROCM_OXIDE_VERIFY_TIMEOUT=<duration>` to tune it, or
`ROCM_OXIDE_VERIFY_TIMEOUT=0` to disable the wrapper for a deliberate manual
run.

The host-only CI profile is:

```bash
scripts/verify.sh --host-ci
```

It covers:

- formatting checks;
- verification script syntax;
- cargo package dry-run coverage;
- proc-macro, build-tool, and cargo-wrapper tests that do not need ROCm.

The ROCm offline profile adds:

- documentation generation with `cargo doc --no-deps`;
- strict `cargo clippy --all-targets` warnings-as-errors coverage;
- root crate build-script coverage without running GPU examples.

The quick profile covers:

- host/runtime unit tests;
- proc-macro and build-tool tests;
- doctor and pipeline inspection;
- core generated-binding GPU smoke coverage;
- `feature_showcase`;
- external downstream crate compilation through `scripts/consumer-smoke.sh`;
- `validation_profile` JSON output with ROCm version, selected architecture,
  device capabilities, optional-library availability, known-profile deviations,
  and skipped-test reasons;
- `performance_probe` JSON output;
- one headless `spectral_lattice` visual artifact.

The full profile adds the heavier examples and all headless `spectral_lattice`
mode artifacts, including the 4K path.

GPU profiles run Rust tests with one test thread. HIP, COMGR, graph, VMM, and
optional-library tests can share process-wide runtime state, and production
verification should avoid relying on default test parallelism.

## Performance And Demo Safety

Interactive stress demos are bounded probes, not open-ended burn-in tools.
`stress_test_gui`, `stress_3d_gui`, and `possibilities_window` clamp interactive
work-iteration controls to a maximum of 4096 before launch. Keep that cap, keep
increment controls saturating, and prefer finite frame counts, bounded
resolutions, and explicit FPS limits for regression runs.

The performance path should remain honest about what executes. hipBLASLt SGEMM
has a checked execution wrapper and workspace/heuristic caps; Composable Kernel
and rocWMMA are candidate/probe-only until ROCm-Oxide has real execution
wrappers for them. The graph runtime currently covers empty/dependency, memcpy,
memset, kernel, memory allocation/free, instantiate/replay, node retargeting, and
exec update. Event and host-callback graph nodes are future work.

## CI Gates

Detailed promotion rules are captured in [CI and release gates](release-gates.md).

The standard GitHub Actions workflow runs `scripts/verify.sh --host-ci` on pull
requests and pushes to `main`.

The manual GPU workflow runs the ROCm offline quality gate on the `gfx1201`
runner, then runs quick and full GPU verification on self-hosted Linux ROCm
runners labeled:

- `self-hosted`, `linux`, `rocm`, `gfx1100`;
- `self-hosted`, `linux`, `rocm`, `gfx1201`.

GPU jobs upload the `target/production-readiness/` directory so logs, JSON
performance output, and headless visual artifacts are available from the run.
The `validation_profile.json` artifact is the checked machine profile for that
run; keep separate artifacts for `gfx1100` and `gfx1201` and compare the
`known_profile.deviations` and `skipped_tests` fields before promoting a feature
that depends on topology-specific ROCm capabilities. The current release-gating
profiles are listed in [Supported ROCm and GPU matrix](supported-rocm-gpu-matrix.md).

## Hardening Priorities

1. Safety audit:
   - detailed audit status is captured in
     [Unsafe and FFI audit](unsafe-audit.md);
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
   - async-created `DeviceBuffer` Drop is blocking cleanup on the retained
     allocation stream; latency-sensitive paths should free explicitly.
   - `StreamPool` is capped at 64 streams, but callers still need to bound
     outstanding async operations because pooled execution is not an unlimited
     queue.
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
