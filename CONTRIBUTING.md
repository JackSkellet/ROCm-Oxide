# Contributing

ROCm-Oxide is moving from prototype to production-oriented development. Keep
changes focused on durable runtime/compiler behavior, not demo-only success.

## Development Setup

- Use the pinned Rust toolchain in `rust-toolchain.toml`.
- Install the `rust-src` component for the pinned nightly.
- Install ROCm tools under `/opt/rocm`, or set `ROCM_PATH`, `HIP_PATH`, or the
  explicit `ROCM_OXIDE_*` tool variables used by `cargo rocm-oxide doctor`.
- Run `cargo run --manifest-path tools/rocm-oxide-build/Cargo.toml -- --doctor`
  when toolchain behavior changes.

## Verification

Before opening a change:

```bash
scripts/verify.sh --offline
```

For runtime, graph, memory, generated binding, device-code, optional-library, or
example changes, also run:

```bash
scripts/verify.sh --quick
```

Run `scripts/verify.sh --full` before release candidates or when the full
example set is affected.

## Change Standards

- Prefer root `rocm_oxide::*` re-exports for stable user-facing API work.
- Keep experimental low-level modules documented and migration-friendly.
- Add a precise `# Safety` contract for every new public unsafe function.
- Add negative tests for invalid ABI, pointer, lifetime, ordering, descriptor,
  cache-key, or capability edges that can be rejected before FFI.
- Record GPU/ROCm capability assumptions in validation artifacts or docs.
- Do not promote a feature based on a single GPU profile when `gfx1100` and
  `gfx1201` report relevant capability differences.

## Generated Artifacts

Generated files under `target/` are verification artifacts, not source. Keep
source changes in `src/`, `crates/`, `tools/`, `examples/`, `docs/`, scripts, or
workflow files.
