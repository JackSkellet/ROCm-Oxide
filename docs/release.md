# Release Process

ROCm-Oxide is still an experimental SDK preview. Do not tag a preview release
until the clean root SDK path, separated demo catalogue, docs, diagnostics, and
verification gates all pass.

## Required Gates

Run these from the repository root:

```sh
cargo check
cargo check --examples
cargo check --features device-spike --examples
cargo rocm-oxide doctor
cargo rocm-oxide verify --quick
```

Run every separated demo manifest:

```sh
for manifest in demo-projects/*/Cargo.toml; do
  cargo check --manifest-path "$manifest"
done
```

For GPU release candidates, also run the full verification profile on each
release-gating machine:

```sh
cargo rocm-oxide verify --full
cargo run --example validation_profile -- --json target/validation_profile.json
cargo run --features device-spike --example performance_probe -- --json target/performance_probe.json
```

## Required Records

For each validation machine, keep:

- GPU model and gfx architecture
- ROCm/HIP runtime version
- Rust version
- `cargo rocm-oxide doctor` output
- `validation_profile.json`
- `performance_probe.json`
- failed or skipped gate notes

The current known-good release-gating profiles are `gfx1100` and `gfx1201`.
Other AMD GPUs may work, but they should not block preview tags unless they are
explicitly promoted to release-gating status.

## Wiki Source

The older detailed checklists and promotion notes are preserved in
[`docs/wiki/release_checklist.md`](wiki/release_checklist.md),
[`docs/wiki/release-gates.md`](wiki/release-gates.md), and
[`docs/wiki/production-readiness.md`](wiki/production-readiness.md).
