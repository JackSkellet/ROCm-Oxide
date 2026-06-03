# ROCm-Oxide Troubleshooting

Run `cargo rocm-oxide doctor` first. It checks every prerequisite, prints
PASS / WARN / FAIL for each item, and suggests a fix for every FAIL. Copy the
block between the dashed markers into a GitHub issue when asking for help.

---

## Doctor output format

```
ROCm-Oxide doctor
=================
system:  Linux my-machine 6.10.0 #1 SMP x86_64 GNU/Linux
cwd:     /home/user/ROCm-Oxide
context: source workspace

[PASS] cargo: cargo 1.82.0-nightly (abc12345 2025-09-01)
[PASS] rustc channel: nightly (rustc 1.82.0-nightly)
[PASS] amdgcn target: amdgcn-amd-amdhsa in rustc target list
[PASS] rust-src component: rust-src installed
[PASS] /dev/kfd: exists and readable
[PASS] tool search order: explicit env > /opt/rocm=/opt/rocm > PATH
[PASS] ROCm llc: /opt/rocm/lib/llvm/bin/llc [/opt/rocm] (AMD clang ...)
...
doctor: all 15 checks passed
```

---

## Rust toolchain issues

### `[FAIL] rustc channel: stable or beta Rust detected`

**Cause:** GPU kernel compilation requires `-Z build-std=core`, which is a
nightly-only flag.

**Fix:** Add a `rust-toolchain.toml` to your project root:

```toml
[toolchain]
channel = "nightly"
components = ["rust-src"]
```

Or switch the active toolchain:

```sh
rustup toolchain install nightly
rustup override set nightly
rustup component add rust-src
```

### `[FAIL] amdgcn target: rustc does not list required target`

**Cause:** The `amdgcn-amd-amdhsa` target is only available on nightly Rust.

**Fix:** Same as above — switch to nightly.

### `[FAIL] rust-src component: missing`

**Cause:** The `rust-src` component is required to compile `core` for the GPU
target from scratch (`-Z build-std=core`).

**Fix:**

```sh
rustup component add rust-src
```

If you have a `rust-toolchain.toml`, add `"rust-src"` to the `components` list.

### `error: the option 'Z' is only accepted on the nightly compiler`

**Cause:** `cargo rustc -Z build-std=core` was invoked with a stable toolchain.

**Fix:** See "rustc channel" above.

---

## GPU / driver issues

### `[FAIL] /dev/kfd: device node does not exist`

**Cause:** The `amdgpu` kernel module is not loaded, or no AMD GPU is present.

**Fix:**

```sh
sudo modprobe amdgpu
# verify it loaded:
ls -la /dev/kfd
```

If the module loads but the node is missing, check `dmesg | grep amdgpu` for
firmware errors. See [docs/supported-rocm-gpu-matrix.md](supported-rocm-gpu-matrix.md).

### `[FAIL] /dev/kfd: permission denied`

**Cause:** Your user is not in the `render` (and optionally `video`) groups.

**Fix:**

```sh
sudo usermod -aG render,video $USER
# then log out and back in (or start a new login shell):
su - $USER
```

Verify with `groups` — you should see `render` in the output.

### `[FAIL] GPU architecture: no AMD GPU detected`

**Cause:** `rocminfo` found no GPU, even though `/dev/kfd` exists. This usually
means `/dev/kfd` is not readable by your user (see above).

**Fix:** Fix `/dev/kfd` permissions first, then re-run doctor.

Alternatively, if you know your GPU architecture, force it:

```sh
export ROCM_OXIDE_ARCH=gfx1100   # RX 7900 XT / 7900 XTX
export ROCM_OXIDE_ARCH=gfx1201   # RX 9070 XT
```

Valid gfx targets for supported GPUs are listed in
[docs/supported-rocm-gpu-matrix.md](supported-rocm-gpu-matrix.md).

---

## ROCm tools issues

### `[FAIL] ROCm llc: not found`

**Cause:** `llc` from ROCm is not on `PATH` and was not found under `/opt/rocm`.

**Fix (option A):** Add the ROCm LLVM bin directory to `PATH`:

```sh
export PATH=/opt/rocm/lib/llvm/bin:$PATH
```

**Fix (option B):** Point directly to the binary:

```sh
export ROCM_OXIDE_LLC=/opt/rocm/lib/llvm/bin/llc
```

**Fix (option C):** Set `ROCM_PATH` to the ROCm installation root:

```sh
export ROCM_PATH=/opt/rocm
```

### `[FAIL] llc amdgcn backend: system llc without amdgcn`

**Cause:** A system-installed `llc` (e.g. from `llvm` package) was found on
`PATH` before the ROCm llc, but it was compiled without the `amdgcn` backend.

**Fix:** Explicitly point to the ROCm llc:

```sh
export ROCM_OXIDE_LLC=/opt/rocm/lib/llvm/bin/llc
```

### `[FAIL] ROCm clang: not found`

**Fix:** Same pattern as `llc` above — set `ROCM_OXIDE_CLANG` or add ROCm to `PATH`.

### `[WARN] ROCm llvm-objdump: not found`

This is optional. It is only needed for LDS-block and scoped-atomics
verification in the device-spike test suite. Normal builds work without it.

### `could not find ROCm tool`

The full error lists every path that was checked. If your ROCm installation is
in a non-standard location, set:

```sh
export ROCM_PATH=/path/to/rocm
```

---

## Build failures

### `rocm-oxide-build not found`

**Cause (scaffold):** If building from a generated scaffold, `build.rs` uses a
relative path (`RUNTIME_PATH`) to find the ROCm-Oxide workspace. The workspace
is not at that relative location.

**Fix options:**

1. Keep the ROCm-Oxide workspace at the expected relative path. Run
   `cat build.rs | grep RUNTIME_PATH` to see what is expected.
2. Set `ROCM_OXIDE_BUILD` to point to a pre-built binary:
   ```sh
   export ROCM_OXIDE_BUILD=/path/to/rocm-oxide-build
   ```
3. Re-generate the scaffold from the correct workspace location:
   ```sh
   cargo rocm-oxide new my-project
   ```

### `rocm-oxide-build failed` / build.rs panic

The panic output includes the full stdout and stderr from `rocm-oxide-build`.
Look for:
- `no #[kernel] functions found` → check `device-spike/src/lib.rs` for `#[kernel]`
- `failed to detect ROCm GPU architecture` → see GPU detection above
- `could not find llc` → see ROCm tools above
- `rustc does not list required target` → see Rust toolchain above

### `could not find Cargo.toml in ...` (wrong working directory)

`cargo rocm-oxide build|run|pipeline|debug|profile` must be run from the
consumer project root (the directory containing the host-side `Cargo.toml`).
Make sure you `cd` to the project directory first.

### `[FAIL] core build probe`

**Cause:** `cargo rustc -Z build-std=core --target amdgcn-amd-amdhsa` failed.

Common sub-causes:
- stable Rust: see "rustc channel" above
- missing `rust-src`: see "rust-src component" above
- the nightly build itself is broken: try `rustup update nightly`

---

## Workspace / scaffold issues

### `verify is only supported from the ROCm-Oxide source workspace`

`cargo rocm-oxide verify` runs the workspace test suite. It must be run from
the ROCm-Oxide source repository, not from a generated scaffold. Change into the
ROCm-Oxide directory and run it there.

### `cargo rocm-oxide doctor` says `context: unknown`

Doctor is being run from somewhere that is neither a source workspace nor a
generated scaffold. For the most useful output, run doctor from either:

- The ROCm-Oxide source workspace root, or
- The root of a project generated by `cargo rocm-oxide new`.

---

## Reporting a bug

1. Run `cargo rocm-oxide doctor` from your ROCm-Oxide workspace.
2. Copy everything between the `--- paste into GitHub issues ---` and
   `--- end doctor report ---` lines.
3. Paste it into your issue along with the error message you saw.

The block includes system info, environment variables, and the status of every
prerequisite check, which is all we need to reproduce your environment.
