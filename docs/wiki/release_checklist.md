# Release Checklist — First Experimental Tag

This document is the gate for tagging the first experimental SDK release
(`v0.2.0-alpha` or similar). Nothing here implies production readiness.
The project is **experimental**. APIs will change. Generated bindings will
change. Do not rely on this for production use.

See [stability-policy.md](stability-policy.md) for the full stability
commitment.

---

## Supported Platforms

| Platform | Status |
|---|---|
| Linux x86-64 | Supported |
| Linux aarch64 | Untested — may work |
| Windows | Not supported |
| macOS | Not supported |

ROCm requires Linux. The Rust device-kernel path requires nightly Rust with
`rust-src`, which is available on Linux x86-64.

---

## Supported ROCm Versions

| ROCm version | Status |
|---|---|
| 7.2.x (`7.2.53211-364a905`) | Validated on `gfx1100` and `gfx1201` |
| 7.1.x | Untested — likely works |
| 7.0.x | Untested |
| 6.x | Not supported |

ROCm 7.2 is the minimum validated version. Older releases lack the AMDGPU
target features and `clang` ABI used by the Rust device-kernel pipeline.

---

## Supported GPU Architectures

| Architecture | GPU | Status | Gate |
|---|---|---|---|
| `gfx1100` | AMD Radeon RX 7900 XT | Release-gating | quick + full |
| `gfx1201` | AMD Radeon RX 9070 XT | Release-gating | quick + full |
| Other RDNA 3 / RDNA 4 | — | May work — not validated |
| CDNA (datacenter) | — | Untested |
| `gfx906`, `gfx908`, `gfx90a` | — | Not tested in this project |

Only `gfx1100` and `gfx1201` are release-gating profiles. A feature blocked on
a single profile must document an accepted skip reason in
`validation_profile.json` before the tag.

---

## Required Examples

These examples must build and run cleanly before tagging:

### Tier 1 — Must pass (blocks tag)

| Example | Command | What it tests |
|---|---|---|
| `hello_gpu` | `cargo run --example hello_gpu` | HIPRTC runtime path, vector add |
| `hello_gpu_rust` | `cargo run --features device-spike --example hello_gpu_rust` | Full Rust device-kernel pipeline |
| `rust_device_generated_bindings` | `cargo run --features device-spike --example rust_device_generated_bindings` | Generated `DeviceKernels` struct, typed launch |
| `feature_showcase` | `cargo run --features device-spike --example feature_showcase` | Runtime feature coverage sweep |
| `validation_profile` | `cargo run --example validation_profile` | Device caps, ROCm version, library availability |
| `performance_probe` | `cargo run --features device-spike --example performance_probe -- --json target/performance_probe.json` | Timing/resource JSON artifact |

### Tier 2 — Should pass (documents skip if not)

| Example | Notes |
|---|---|
| `rust_device_add_one` | Raw `launch!` smoke test |
| `rust_device_vector_add` | Raw `launch!` with generated HSACO |
| `vector_add` | HIP C++ via HIPRTC |
| `pinned_stream_vector_add` | Pinned host memory + stream ordering |
| `module_global` | Device global read/write |
| `cargo run --manifest-path demo-projects/spectral-lattice/Cargo.toml -- --frames 3` | Headless visual smoke (PNG artifact) |
| `cargo run --manifest-path demo-projects/compiler-feature-lab/Cargo.toml -- --frames 1` | Feature probe GUI headless path |

### Tier 3 — Run manually, not blocking

All other root SDK examples in `examples/` and separated demo crates in
`demo-projects/`. Visual demos may require a display or accept `--frames N` for
headless runs.

---

## Required Documentation

These docs must exist, be accurate, and link correctly:

| Doc | Status | Notes |
|---|---|---|
| `README.md` | ✓ | Links all major paths |
| `docs/getting-started.md` | ✓ | Fixed API examples, correct llc paths |
| `docs/wiki/hello_gpu.md` | ✓ | HIPRTC walkthrough |
| `docs/wiki/hello_gpu_rust.md` | ✓ | Rust device-kernel walkthrough |
| `docs/wiki/api_overview.md` | ✓ | Fixed generated binding types |
| `docs/troubleshooting.md` | ✓ | Error-by-error guide |
| `docs/project-generation.md` | ✓ | Scaffold docs |
| `docs/api-stability.md` | ✓ | Stable vs experimental surface |
| `docs/release-profile-template.md` | ✓ | Known-good machine record format |
| `docs/wiki/stability-policy.md` | ✓ | Experimental SDK commitment |
| `docs/wiki/supported-rocm-gpu-matrix.md` | ✓ | GPU/ROCm matrix |
| `docs/wiki/release-gates.md` | ✓ | CI and promotion rules |
| `docs/wiki/release_checklist.md` | ✓ | This document |
| `CHANGELOG.md` | ✓ | Unreleased section current |
| `CONTRIBUTING.md` | ✓ | Issue templates, dev setup |
| `SECURITY.md` | ✓ | |
| `DESIGN.md` | ✓ | |

---

## Required Commands

These must all pass before tagging:

### Host-only (no GPU required)

```sh
scripts/verify.sh --host-ci
```

Covers: formatting, script syntax, first-user script syntax, cargo package
dry-run, proc-macro tests, build-tool unit tests, cargo-wrapper unit tests.

### ROCm offline (ROCm tools + libs required, GPU optional)

```sh
ROCM_OXIDE_ARCH=gfx1100 scripts/verify.sh --offline
```

Covers: `cargo doc --no-deps`, strict clippy, build-script integration.

### Quick GPU (live GPU required)

```sh
scripts/verify.sh --quick
```

Must pass on **both** `gfx1100` and `gfx1201`.

Covers: host/runtime unit tests, doctor, pipeline inspection, generated-binding
GPU smoke, `feature_showcase`, consumer-smoke downstream compilation,
the README first-user path, and `validation_profile` JSON output.

### Full GPU (live GPU required)

```sh
scripts/verify.sh --full
```

Must pass on **both** `gfx1100` and `gfx1201`.

Covers: all quick targets plus the separated `spectral-lattice`,
`compiler-feature-lab`, `upscale-artifacts`, and `bvh-raytrace-benchmark` demo
crates. Produces `release_manifest.json`.

---

## Pre-Release Test Matrix

Before the tag, collect these artifacts for each release-gating profile:

| Profile | `gfx1100` | `gfx1201` |
|---|---|---|
| `--host-ci` | pass | (host-only, once) |
| `--offline` | pass | (arch-parameterized) |
| `--quick` | ☐ | ☐ |
| `--full` | ☐ | ☐ |
| `validation_profile.json` retained | ☐ | ☐ |
| `performance_probe.json` retained | ☐ | ☐ |
| `release_manifest.json` retained | ☐ | ☐ |
| `spectral_lattice*.png` retained | ☐ | ☐ |

Artifacts must be kept with the release notes. Do not tag without both full-GPU
runs retained.

---

## What Must Pass Before Tagging

All of the following must be true:

- [ ] `scripts/verify.sh --host-ci` passes (zero failures)
- [ ] `scripts/verify.sh --offline` passes on the release machine
- [ ] `scripts/verify.sh --quick` passes on `gfx1100`
- [ ] `scripts/verify.sh --quick` passes on `gfx1201`
- [ ] `scripts/first-user-path.sh` passes on both release-gating profiles
- [ ] `scripts/verify.sh --full` passes on `gfx1100` (or every failure documented with owner + follow-up)
- [ ] `scripts/verify.sh --full` passes on `gfx1201` (same)
- [ ] `validation_profile.json` retained for each GPU run
- [ ] `performance_probe.json` retained for each GPU run
- [ ] `release_manifest.json` retained for at least one GPU run
- [ ] Tier 1 examples all pass on both profiles
- [ ] All Tier 2 skips documented with accepted skip reason
- [ ] `CHANGELOG.md` Unreleased section is current and accurate
- [ ] `docs/wiki/supported-rocm-gpu-matrix.md` reflects the validated ROCm version
- [ ] `cargo rocm-oxide doctor` exits `[PASS]` or `[WARN]`-only on both release machines
- [ ] `cargo doc --no-deps` generates without errors
- [ ] No publicly facing docs contain `new_zeroed`, `&mut DeviceBuffer`, `✗`, or `HIPRTC/COMGR` in the doctor description (consistency audit passed)
- [ ] Issue templates exist in `.github/ISSUE_TEMPLATE/`
- [ ] `docs/wiki/stability-policy.md` exists and disclaims experimental status
- [ ] CHANGELOG, CONTRIBUTING, troubleshooting, and this checklist are linked from README

---

## Known Limitations (as of first experimental tag)

- **No crates.io publication.** The crate depends on ROCm dynamic libraries
  that cannot be bundled. Local path dependencies in generated scaffold projects
  are not crates.io-compatible.
- **Stable Rust not supported for the device-kernel path.** `-Z build-std=core`
  for `amdgcn-amd-amdhsa` requires nightly. The HIPRTC/runtime path works on
  stable.
- **Windows and macOS are not supported.** ROCm is Linux-only. Cross-compilation
  for AMD GPUs from non-Linux hosts is not supported.
- **Generated scaffold projects are not standalone.** `cargo rocm-oxide new`
  creates a project with a relative `path` dependency on this workspace. Moving
  the scaffold without moving the workspace breaks the build.
- **Generated bindings format is not stable.** The `DeviceKernels` struct and
  method signatures may change in any `0.x` release.
- **CUDA PTX, cubin, NVVM, TMA, WGMMA, DSMEM, and cluster launch** are
  CUDA/NVIDIA-only. They are not supported on ROCm.
- **No native GPU debugger integration.** `rocgdb` can attach but the workflow
  is not documented end-to-end.
- **No coverage of CDNA / datacenter GPUs** in the current validation matrix.
- **Optional libraries (rocBLAS, rocFFT, hipBLASLt, COMGR) are runtime-loaded.**
  Their absence is a `[WARN]`, not a `[FAIL]`, in `cargo rocm-oxide doctor`.
  Features that require them are unavailable if the library is absent.
- **`cargo rocm-oxide verify` works only inside the ROCm-Oxide source workspace.**
  It does not apply to generated scaffold projects.

---

## How to File a Release Blocker

Use the **Bug Report** issue template. Include:

1. Full `cargo rocm-oxide doctor` output (copy the GitHub-issue block).
2. The exact command that failed.
3. Full terminal output.
4. ROCm version (`/opt/rocm/.info/version` or `rocminfo | head -5`).
5. GPU architecture (`rocminfo | grep 'gfx'`).
