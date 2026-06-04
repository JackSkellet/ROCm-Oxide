# Contributing

ROCm-Oxide is an **experimental SDK** moving from prototype toward a first
tagged release. Keep changes focused on durable runtime/compiler behavior, not
demo-only success. See [docs/stability-policy.md](docs/stability-policy.md) for
the API stability and experimental-SDK commitment.

## Before You Contribute

Run the doctor to confirm your environment is valid:

```sh
cargo rocm-oxide doctor
```

Fix all `[FAIL]` items before opening a change. `[WARN]` items for optional
libraries are acceptable.

## Development Setup

- Use the pinned Rust toolchain in `rust-toolchain.toml`.
- Install the `rust-src` component for the pinned nightly.
- Install ROCm tools under `/opt/rocm`, or set `ROCM_PATH`, `HIP_PATH`, or the
  explicit `ROCM_OXIDE_*` tool variables used by `cargo rocm-oxide doctor`.
- Note: `llc`, `llvm-readelf`, and `llvm-objdump` are at
  `/opt/rocm/lib/llvm/bin/`; `clang` and `rocminfo` are at `/opt/rocm/bin/`.

## Verification

Before opening a change:

```sh
scripts/verify.sh --offline
```

For runtime, graph, memory, generated binding, device-code, optional-library, or
example changes, also run:

```sh
scripts/verify.sh --quick
```

Run `scripts/verify.sh --full` before release candidates or when the full
example set is affected. See [docs/release-gates.md](docs/release-gates.md) for
the full CI and promotion policy.

## Change Standards

- Prefer root `rocm_oxide::*` re-exports for stable user-facing API work.
- Keep experimental low-level modules documented and migration-friendly.
- Add a precise `# Safety` contract for every new public unsafe function.
- Add negative tests for invalid ABI, pointer, lifetime, ordering, descriptor,
  cache-key, or capability edges that can be rejected before FFI.
- Record GPU/ROCm capability assumptions in validation artifacts or docs.
- Do not promote a feature based on a single GPU profile when `gfx1100` and
  `gfx1201` report relevant capability differences.
- Update `CHANGELOG.md` (Unreleased section) for changes that affect public
  behavior, docs, generated bindings, or the verification pipeline.

## Reporting Issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`:

- **Bug Report** — runtime failures, panics, wrong output, build failures
- **GPU Compatibility Report** — new GPU architectures, ROCm version results
- **Documentation Issue** — inaccurate docs, broken examples, misleading wording

Always include the output of `cargo rocm-oxide doctor` (it prints a copy-pasteable
GitHub block at the end).

## Generated Artifacts

Generated files under `target/` are verification artifacts, not source. Keep
source changes in `src/`, `crates/`, `tools/`, `examples/`, `docs/`, scripts, or
workflow files.
