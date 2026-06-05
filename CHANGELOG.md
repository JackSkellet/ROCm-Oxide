# Changelog

All notable ROCm-Oxide changes should be recorded here before a tagged release.

## Unreleased

### SDK preview restructure and release gates (2026-06)

- Split large visual, capture, artifact, benchmark, and experimental demos out
  of root `examples/` into standalone crates under `demo-projects/`.
- Slimmed the root crate back to SDK/tutorial/diagnostic examples and removed
  visual-demo dependencies from the root manifest.
- Moved long-form design and historical docs into `docs/wiki/`, added
  `docs/index.md`, and kept the maintained docs surface focused on onboarding,
  troubleshooting, project generation, release process, and API stability.
- Added conservative host and device preludes for the first abstraction pass.
- Updated `scripts/verify.sh` so quick/full release gates use
  `--features device-spike` for root Rust-device examples and
  `--manifest-path demo-projects/.../Cargo.toml` for separated demo crates.
- Added `docs/release-profile-template.md` for known-good release machine
  records.

### SDK onboarding and diagnostics (2026-06)

- **Doctor rewrite**: `cargo rocm-oxide doctor` now runs as a non-aborting
  diagnostic collector. Every check prints `[PASS]`, `[WARN]`, or `[FAIL]`.
  Checks cover: `cargo`, `rustc` channel, `amdgcn-amd-amdhsa` target,
  `rust-src`, `/dev/kfd` permissions, ROCm tools (`llc`, `clang`,
  `llvm-readelf`), `rocminfo`, GPU architecture, `rocm_agent_enumerator`, and a
  full `core` build probe. Doctor output ends with a copy-pasteable GitHub
  issue block.
- **Actionable build errors**: `build_tool_command()` in `cargo-rocm-oxide` now
  checks `ROCM_OXIDE_BUILD` env, the source manifest, and `PATH` before
  panicking with an actionable three-option message showing the current
  `RUNTIME_PATH` value.
- **Scaffold README**: Generated projects now include a README that tells users
  to run `cargo rocm-oxide doctor` from the ROCm-Oxide source workspace before
  building.
- **`docs/troubleshooting.md`** (new): Comprehensive error-by-error guide
  covering Rust toolchain failures, `/dev/kfd` issues, ROCm tool discovery,
  `build.rs` panics, scaffold path issues, and bug reporting instructions.
- **`docs/wiki/stability-policy.md`** (new): Explicit experimental-SDK stability
  commitment covering API tiers, generated-bindings stability, crates.io
  status, and breaking-change definitions.
- **`docs/wiki/release_checklist.md`** (new): First experimental tag gate covering
  required examples, required docs, required commands, supported platforms,
  supported ROCm versions, supported GPU architectures, known limitations, and
  the pre-release test matrix.
- **`docs/getting-started.md`** fixes: Corrected `DeviceBuffer::new_zeroed` →
  `new`, `&mut out` → `&out` (generated bindings take `&DeviceBuffer<T>`),
  added `unsafe {}` around generated launch calls, updated doctor description
  to match new `[PASS]`/`[WARN]`/`[FAIL]` format, fixed `verify --quick`
  context (source workspace only), corrected `llc` path to
  `/opt/rocm/lib/llvm/bin/`.
- **`docs/wiki/api_overview.md`** fixes: Corrected generated-binding type mapping
  (`DeviceSliceMut<T>` → `&DeviceBuffer<T>`, not `&mut`), fixed example to use
  `&out` inside `unsafe {}`.
- **`docs/wiki/hello_gpu_rust.md`** fixes: Corrected `llc` path in requirements,
  troubleshooting, and the artifact disassembly command.

### Earlier (pre-2026-06)

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
