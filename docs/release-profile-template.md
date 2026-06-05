# Release Profile Template

Create one profile for each release-gating machine before tagging an SDK
preview. Keep the completed profile with the release artifacts.

## Candidate

- Candidate tag:
- Commit:
- Date:
- Operator:
- Machine:
- GPU:
- gfx architecture:
- ROCm version:
- HIP runtime version:
- Kernel / driver:
- Rust version:
- Cargo version:

## Required Commands

Run from the ROCm-Oxide repository root:

```sh
git rev-parse HEAD
cargo rocm-oxide doctor
scripts/first-user-path.sh
scripts/verify.sh --quick
scripts/verify.sh --full
cargo run --example validation_profile -- --json target/production-readiness/validation_profile.json
cargo run --features device-spike --example performance_probe -- --json target/production-readiness/performance_probe.json
```

Run `scripts/verify.sh --host-ci` once per release candidate. Run
`scripts/verify.sh --offline` for each architecture-specific release machine.

## Gate Results

| Gate | Command | Result | Artifact |
|---|---|---|---|
| Source commit | `git rev-parse HEAD` |  |  |
| Doctor | `cargo rocm-oxide doctor` |  |  |
| First-user path | `scripts/first-user-path.sh` |  |  |
| Host CI | `scripts/verify.sh --host-ci` |  | `target/production-readiness/verify-host-ci.log` |
| Offline | `scripts/verify.sh --offline` |  | `target/production-readiness/verify-offline.log` |
| Quick GPU | `scripts/verify.sh --quick` |  | `target/production-readiness/verify-quick.log` |
| Full GPU | `scripts/verify.sh --full` |  | `target/production-readiness/verify-full.log` |

## Retained Artifacts

- `target/production-readiness/validation_profile.json`
- `target/production-readiness/performance_probe.json`
- `target/production-readiness/release_manifest.json`
- `target/production-readiness/spectral_lattice*.png`
- full `cargo rocm-oxide doctor` output

## Skips, Warnings, and Failures

- Doctor warnings:
- Skipped validation tests:
- Optional ROCm libraries unavailable:
- Gate failures:
- Accepted skip reasons:
- Follow-up owner:
