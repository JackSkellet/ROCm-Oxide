# Compiler Path Notes

## Observed Local Toolchain

- `rustc --print target-list` includes `amdgcn-amd-amdhsa`.
- `/opt/rocm/lib/llvm/bin/llc --version` includes the `amdgcn` target.
- `hipcc --help` supports `--offload-arch=<gfx...>` and `-fgpu-rdc`.
- The installed GPU target is `gfx1201`.

## Stable Rust Device Compilation Probe

This minimal probe shows the stable-toolchain blocker:

```bash
printf '%s\n' \
  '#![no_std]' \
  '#[unsafe(no_mangle)]' \
  'pub unsafe extern "C" fn empty() {}' \
| rustc --target amdgcn-amd-amdhsa --crate-type=lib --emit=llvm-ir -O - -o /tmp/rocm_oxide_empty.ll
```

Result:

```text
can't find crate for `core`
```

The target exists, but stable Rust does not have `core` prebuilt for it in this
environment.

## Working Nightly Spike

The current best path is a rustc-based device compiler wrapper, not a HIP-source
generator.

The spike lives in `device-spike/` and is built by:

```bash
./scripts/compile-device-spike.sh
```

That script delegates to `tools/rocm-oxide-build`, which:

1. uses `cargo +nightly rustc -Z build-std=core`
2. targets `amdgcn-amd-amdhsa`
3. applies `-C target-cpu=$ROCM_OXIDE_ARCH`
4. emits LLVM IR
5. discovers source functions marked with `#[kernel]`
6. rewrites those functions into `amdgpu_kernel` entry points
7. converts generic pointer args and derived `getelementptr` values to global
   address-space pointers
8. adds `amdgpu-flat-work-group-size`
9. lowers with ROCm `llc`
10. links a `.hsaco` with ROCm `clang`
11. validates that each kernel has a `.kd` kernel descriptor

The root `build.rs` runs the build tool before the host crate compiles and
exports:

- `ROCM_OXIDE_DEVICE_HSACO`
- `ROCM_OXIDE_DEVICE_BINDINGS`
- `ROCM_OXIDE_DEVICE_METADATA`

The launch tests are:

```bash
cargo run --example rust_device_add_one
cargo run --example rust_device_generated_bindings
```

This successfully launches two Rust-authored kernels on `gfx1201`:

- `add_one`, using `workitem_id_x`
- `vector_add`, using `workgroup_id_x` and `workitem_id_x`

## Remaining Compiler Work

The current IR post-pass is still a transitional compiler wrapper. It now uses
the explicit source marker rather than a symbol naming convention, but the next
layer should:

- add generated argument validation for buffer lengths and scalar ABI widths
- preserve and validate source-level kernel signatures
- support more pointer-producing IR operations than `getelementptr`
- preserve non-kernel helper functions as ordinary device functions
- generate or validate kernel metadata and workgroup-size attributes
- copy resulting `.hsaco`, metadata, and bindings into a host crate build
  directory

The fallback remains a restricted Rust frontend that emits LLVM IR directly, but
the working rustc path is now strong enough to pursue first.
