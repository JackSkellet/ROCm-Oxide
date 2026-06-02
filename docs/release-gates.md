# CI and Release Gates

ROCm-Oxide release decisions should be based on repeatable verification
artifacts, not on a single demo run.

## Pull Request Gate

Every pull request must pass the host-only production gate:

```bash
scripts/verify.sh --host-ci
```

This gate runs formatting, shell syntax checks, a cargo package dry-run, and
CPU-only Rust tests for the proc macro, build tool, and cargo wrapper. It must
not require ROCm tools, ROCm libraries, or a visible ROCm GPU.

## ROCm Offline Quality Gate

Before tagging, run the ROCm-installed offline gate:

```bash
scripts/verify.sh --offline
```

This gate adds documentation generation and strict clippy for the root crate.
It requires the ROCm toolchain and libraries used by `build.rs`, and it needs
either a visible ROCm GPU or `ROCM_OXIDE_ARCH=gfx...` so device artifacts can be
generated without guessing an architecture.

## GPU Promotion Gate

Changes that affect runtime behavior, generated bindings, device code, graph
execution, memory semantics, optional ROCm libraries, or performance-sensitive
examples need at least the quick GPU gate:

```bash
scripts/verify.sh --quick
```

Run the full gate before release candidates and before promoting a feature that
depends on examples outside the quick profile:

```bash
scripts/verify.sh --full
```

GPU verification should be run separately on `gfx1100` and `gfx1201` self-hosted
ROCm runners. Upload and retain the entire `target/production-readiness/`
directory for each architecture/profile pair.

The quick and full gates also run `scripts/consumer-smoke.sh`, which creates a
temporary crate outside the repository and verifies that downstream projects can
compile against root `rocm_oxide::*` exports without reaching into internal
modules.

## Required Artifacts

Release reviewers should inspect these artifacts from every GPU run:

- `verify-quick.log` or `verify-full.log`: command transcript and failure
  context;
- `validation_profile.json`: ROCm runtime version, selected architecture,
  device capability flags, optional-library availability, known-profile
  deviations, and skipped-test reasons;
- `performance_probe.json`: kernel timings, occupancy summaries, resource
  pressure, and limiter flags;
- `spectral_lattice*.png`: headless visual smoke artifacts for presentation and
  render-path regressions.

If `validation_profile.json` contains `known_profile.deviations`, treat the run
as a capability drift signal. Do not promote features that depend on the changed
capability until the deviation is understood and documented.

## Versioning and Known Limitations

Before a tagged release:

- update release notes with the supported ROCm version, GPU architectures, and
  required Rust nightly;
- keep [Supported ROCm and GPU matrix](supported-rocm-gpu-matrix.md) current;
- include the `gfx1100` and `gfx1201` validation artifact summaries;
- list skipped tests and unavailable optional libraries from
  `validation_profile.json`;
- call out unsupported CUDA-only ABI surfaces: PTX, cubin, NVVM, TMA, WGMMA,
  DSMEM, and CUDA cluster launch;
- keep public API stability notes aligned with
  [API stability](api-stability.md).

## Failure Policy

Host-only CI failures block pull requests. ROCm offline failures block tags.
Quick GPU failures block runtime, compiler, generated-binding, and
library-interoperability promotion. Full GPU failures block releases unless the
failure is documented as an accepted known limitation with an owner and
follow-up task.
