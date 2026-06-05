# Project Generation — `cargo rocm-oxide new`

This document describes what `cargo rocm-oxide new <path>` generates, what
its current limitations are, and the roadmap to standalone project support.

---

## Current mode: local scaffold

`cargo rocm-oxide new` creates a **local scaffold**: a project that builds
against the ROCm-Oxide source workspace via relative `path` dependencies in
`Cargo.toml` and `build.rs`.

The generated project is not standalone. It is designed for developing GPU code
alongside the ROCm-Oxide workspace, not for shipping or sharing independently.

### What is generated

Running `cargo rocm-oxide new my-app` from inside the ROCm-Oxide workspace, or
`cargo rocm-oxide new my-app --local /path/to/ROCm-Oxide` from another
directory, produces this layout:

```
my-app/
  Cargo.toml              — host crate; depends on rocm-oxide via relative path
  build.rs                — invokes rocm-oxide-build; RUNTIME_PATH is relative
  rust-toolchain.toml     — selects nightly + rust-src (required for device build)
  README.md               — scaffold-specific usage and portability notes
  src/
    main.rs               — sample host program using generated DeviceKernels
  device-spike/
    Cargo.toml            — device crate; depends on rocm-oxide-{device,kernel}
    src/
      lib.rs              — sample #[kernel] function
```

### Relative path computation

All dependency paths are computed relative at generation time:

| File | Path relative to... | Example |
|------|---------------------|---------|
| `Cargo.toml` `rocm-oxide = { path = ... }` | Project root | `../ROCm-Oxide` |
| `build.rs` `RUNTIME_PATH` | Project root (CWD when build.rs runs) | `../ROCm-Oxide` |
| `device-spike/Cargo.toml` `rocm-oxide-device = { path = ... }` | `device-spike/` | `../../ROCm-Oxide/crates/rocm-oxide-device` |
| `device-spike/Cargo.toml` `rocm-oxide-kernel = { path = ... }` | `device-spike/` | `../../ROCm-Oxide/crates/rocm-oxide-kernel` |

No absolute paths are written.

### Explicit local workspace

Use `--local` when you want to generate a project from a directory that is not
inside the ROCm-Oxide checkout:

```sh
cargo rocm-oxide new my-app --local /path/to/ROCm-Oxide
```

`--path` is an alias for `--local`:

```sh
cargo rocm-oxide new my-app --path ../ROCm-Oxide
```

Both forms still generate relative `path` dependencies. The option only makes
the source workspace explicit; it does not make the scaffold standalone.

`--standalone` is reserved and currently exits with an explanatory error. It
will become available only after the runtime, device API, proc macro, and build
tool can be consumed through crates.io or release artifacts.

---

## Portability matrix

| Scenario | Works? | Reason |
|----------|--------|--------|
| Build in place | ✓ | Relative paths are resolved from project root |
| Move project + workspace together | ✓ | Relative distance preserved |
| Move only the workspace | ✗ | Relative paths resolve to wrong location |
| Move only the project | ✗ | Relative paths resolve to wrong location |
| Clone project on another machine (same relative layout) | ✓ | Provided ROCm-Oxide is cloned at the same relative path |
| Clone project on another machine (arbitrary layout) | ✗ | ROCm-Oxide not at expected relative path |
| `cargo publish` | ✗ | `path` dependencies are rejected by crates.io |

### Safe way to move both

Move the project and workspace as siblings — keep the relative distance between
them identical. For example, if the project was generated inside the workspace:

```
/repos/ROCm-Oxide/          ← workspace
/repos/ROCm-Oxide/my-app/   ← project (runtime_path = "..")
```

Moving both to `/work/` keeps the relative path `..` valid:

```
/work/ROCm-Oxide/
/work/ROCm-Oxide/my-app/
```

If the project is a sibling:

```
/repos/ROCm-Oxide/     ← workspace
/repos/my-app/         ← project (runtime_path = "../ROCm-Oxide")
```

Both must be moved together to preserve `../ROCm-Oxide`.

---

## `cargo rocm-oxide verify` is source-workspace only

`cargo rocm-oxide verify [--quick|--full|...]` is a repository-level gate.
It runs `scripts/verify.sh` from the ROCm-Oxide source root and tests the
entire SDK. It does not apply to generated projects.

To verify a generated project builds correctly:

```sh
cd my-app
cargo build
```

---

## Escape hatches

### Pre-built `rocm-oxide-build` binary

The `build.rs` template checks `ROCM_OXIDE_BUILD` before looking for the source
workspace. If you have a pre-compiled `rocm-oxide-build` binary, you can point
it there:

```sh
ROCM_OXIDE_BUILD=/path/to/rocm-oxide-build cargo build
```

This removes the source-workspace requirement for the build tool. The
`rocm-oxide` runtime dependency in `Cargo.toml` would still require either the
workspace or a crates.io version.

### Manual `path` → `git` dependency

While crates.io publication is not yet available, you can use a git dependency
as a short-term workaround for sharing with other developers who have cloned
ROCm-Oxide:

```toml
# Cargo.toml
[dependencies]
rocm-oxide = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }
```

This still requires the developer to have the correct ROCm toolchain installed
and does not solve the `build.rs` `RUNTIME_PATH` issue (which would also need to
point to the git-cloned source). A full solution requires publishing both the
runtime crate and the build tool.

---

## Blockers to standalone / crates.io support

| Blocker | Detail |
|---------|--------|
| `rocm-oxide` not on crates.io | Required before consumers can use a version dep |
| `rocm-oxide-device` not on crates.io | Device API crate |
| `rocm-oxide-kernel` not on crates.io | Proc-macro crate |
| `rocm-oxide-build` not distributed | Build tool has no release binary or crates.io presence |
| Generated `build.rs` uses source-workspace to find the build tool | Needs a well-known binary or crates.io `build-dependencies` entry |
| `device-spike` name is not configurable yet | Generated projects use the same internal crate name |

### Minimum path to crates.io publication

1. Publish `rocm-oxide-kernel` (proc-macro, no native deps) → crates.io
2. Publish `rocm-oxide-device` (no_std, no native deps) → crates.io
3. Publish `rocm-oxide` (links native ROCm/HIP) → crates.io with build-metadata
4. Publish `rocm-oxide-build` as a binary crate or distribute via release artifacts
5. Update generated `Cargo.toml` to use version deps + `build-dependencies` for the build tool
6. Update generated `build.rs` to use `build-dependencies` instead of `RUNTIME_PATH`

---

## Future: standalone mode

When the above blockers are resolved, `cargo rocm-oxide new --standalone` will
generate a fully self-contained project using crates.io version dependencies:

```toml
[dependencies]
rocm-oxide = "0.1"

[build-dependencies]
rocm-oxide-build = "0.1"
```

Until then, the local scaffold mode described in this document is the only
supported workflow.
