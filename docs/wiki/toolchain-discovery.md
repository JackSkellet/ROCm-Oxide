# Toolchain Discovery

`rocm-oxide-build --doctor` is the canonical preflight for the Rust-to-HSACO
path. It reports the complete toolchain surface needed by the build:

- host `cargo` and `rustc` versions,
- Rust support for `amdgcn-amd-amdhsa`,
- installed `rust-src`,
- ROCm LLVM tools: `llc`, `clang`, `llvm-readelf`, and `llvm-objdump`,
- ROCm runtime tools: `rocminfo` and `rocm_agent_enumerator`,
- detected GPU architecture,
- whether `core` can be built for `amdgcn-amd-amdhsa` with nightly
  `-Z build-std=core`.

ROCm tool lookup order is explicit:

1. tool-specific overrides such as `ROCM_OXIDE_LLC`,
   `ROCM_OXIDE_CLANG`, `ROCM_OXIDE_LLVM_READELF`,
   `ROCM_OXIDE_LLVM_OBJDUMP`, `ROCMINFO`, and
   `ROCM_AGENT_ENUMERATOR`;
2. `ROCM_PATH`;
3. `HIP_PATH`, including its parent when `HIP_PATH` points at a `hip`
   subdirectory;
4. `/opt/rocm`;
5. `PATH`.

LLVM tools are searched under `lib/llvm/bin`, `llvm/bin`, and `bin` for each
ROCm root. Runtime tools are searched under `bin`.

The normal build path uses the same discovery logic for build-critical tools and
for architecture detection when `--arch` or `ROCM_OXIDE_ARCH` is not supplied.
Doctor validates the wider environment, including runtime enumeration, before a
developer spends time debugging lower-level compiler or HIP failures.
