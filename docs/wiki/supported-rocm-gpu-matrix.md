# Supported ROCm and GPU Matrix

This matrix records the profiles that currently gate production-readiness work.
Other ROCm GPUs may work, but they are not release-gating profiles until they
produce the same validation artifacts.

| Status | Architecture | GPU | ROCm/HIP runtime | AMD clang | Gate |
| --- | --- | --- | --- | --- | --- |
| Release-gating | `gfx1100` | AMD Radeon RX 7900 XT | `7.2.53211-364a905` | `22.0.0git` | quick and full GPU profiles |
| Release-gating | `gfx1201` | AMD Radeon RX 9070 XT | `7.2.53211-364a905` | `22.0.0git` | quick and full GPU profiles |

## Shared Expectations

- Rust uses the nightly toolchain selected by `rust-toolchain.toml` with `rust-src`.
- ROCm tools are discovered from explicit `ROCM_OXIDE_*` variables, `ROCM_PATH`,
  `HIP_PATH`, `/opt/rocm`, or `PATH`.
- GPU verification runs with one Rust test thread.
- `validation_profile.json`, `performance_probe.json`, and headless
  `spectral_lattice` artifacts must be retained for promotion decisions.
- Both profiles currently report wavefront size 32, max workgroup size 1024,
  max waves per CU 32, and 64 KB group/LDS segment.

## Known Profile Differences

- `gfx1100` RX 7900 XT reports memory pools, managed memory, concurrent managed
  access, host mapped memory, and host registration. On the local topology it
  does not report host-native PCIe atomics or direct host access to
  device-resident managed memory, so host-visible system-scope atomic tests
  skip where required.
- `gfx1201` RX 9070 XT reports managed memory, concurrent managed access,
  host-native atomics, host mapped memory, host registration, and memory pools.
  It can run mapped host-visible system-scope atomic coverage that the local
  `gfx1100` profile skips.

## Promotion Rule

Do not promote features that depend on topology-sensitive capabilities until
both release-gating profiles either pass the relevant test or record an accepted
skip reason in `validation_profile.json`.
