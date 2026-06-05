# Project Generation — `cargo rocm-oxide new`

This document describes what `cargo rocm-oxide new <path>` generates, what
its current limitations are, and the roadmap to standalone project support.

---

## Current modes

`cargo rocm-oxide new` supports two preview scaffold modes:

- **Local scaffold**: depends on a ROCm-Oxide source workspace via relative
  `path` dependencies in `Cargo.toml` and `build.rs`.
- **Git scaffold**: depends on ROCm-Oxide crates from a git repository and
  expects `rocm-oxide-build` to be installed through `ROCM_OXIDE_BUILD` or
  `PATH`.

Neither mode is fully standalone. Local mode is best for developing alongside a
source checkout. Git mode is the current bridge for sharing a generated project
without preserving a fixed local workspace path.

### Local scaffold

Running `cargo rocm-oxide new my-app` from inside the ROCm-Oxide workspace, or
`cargo rocm-oxide new my-app --local /path/to/ROCm-Oxide` from another
directory, produces this layout:

```
my-app/
  Cargo.toml              — host crate; depends on rocm-oxide via relative path
  build.rs                — invokes rocm-oxide-build; RUNTIME_PATH is relative
  rust-toolchain.toml     — selects nightly + rust-src (required for device build)
  README.md               — scaffold-specific usage and portability notes
  .vscode/
    settings.json         — rust-analyzer linked host + device projects
    tasks.json            — scaffold check, doctor, build, and run tasks
    extensions.json       — rust-analyzer recommendation
    rocm-oxide.code-snippets
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

### Editor files

Generated projects include VS Code defaults because the host crate and
`device-spike/` are intentionally separate Cargo projects. The settings link
both manifests for rust-analyzer completion and disable automatic check-on-save
so the editor does not run a host-target `cargo check` against AMDGPU-only
device intrinsics.

Use the generated tasks for real validation:

| Task | Command |
|------|---------|
| `ROCm-Oxide: check scaffold` | `cargo rocm-oxide check-consumer` |
| `ROCm-Oxide: doctor` | local scaffold: run doctor from the source workspace; git scaffold: `cargo rocm-oxide doctor` |
| `ROCm-Oxide: build host + device` | `cargo build` |
| `ROCm-Oxide: run app` | `cargo run` |

Rust snippets are also generated for the common device-kernel forms:
`rocm-kernel-1d` and `rocm-vector-add`.

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

### Git scaffold

Use `--git` when you want generated `Cargo.toml` files to reference a git
repository instead of local paths:

```sh
cargo install --git https://github.com/JackSkellet/ROCm-Oxide rocm-oxide-build
cargo rocm-oxide new my-app --git https://github.com/JackSkellet/ROCm-Oxide --rev <commit>
cd my-app
cargo rocm-oxide check-consumer
cargo run
```

The optional git reference flags are mutually exclusive:

```sh
cargo rocm-oxide new my-app --git https://github.com/JackSkellet/ROCm-Oxide --branch main
cargo rocm-oxide new my-app --git https://github.com/JackSkellet/ROCm-Oxide --tag v0.1.0-sdk-preview.1
cargo rocm-oxide new my-app --git https://github.com/JackSkellet/ROCm-Oxide --rev abc1234
```

Generated host dependency:

```toml
rocm-oxide = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }
```

Generated device dependencies:

```toml
rocm-oxide-device = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }
rocm-oxide-kernel = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }
```

Git mode deliberately does **not** embed a source workspace path in `build.rs`.
The generated build script checks:

1. `ROCM_OXIDE_BUILD=/path/to/rocm-oxide-build`
2. `rocm-oxide-build` on `PATH`

If neither exists, the build fails with the exact install command for the same
git source and revision.

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
| Git scaffold cloned elsewhere | ✓ | Crate deps come from git, but `rocm-oxide-build` must be installed |
| Git scaffold `cargo publish` | ✗ | Git deps and external build-tool requirement are not crates.io-compatible |

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

### Git dependencies

While crates.io publication is not yet available, use the supported `--git`
mode instead of hand-editing scaffold manifests:

```sh
cargo rocm-oxide new my-app --git https://github.com/JackSkellet/ROCm-Oxide --rev abc1234
```

This still requires the correct ROCm toolchain and an installed
`rocm-oxide-build` binary. A full standalone solution requires publishing both
the runtime crates and the build tool.

---

## Blockers to standalone / crates.io support

| Blocker | Detail |
|---------|--------|
| `rocm-oxide` not on crates.io | Required before consumers can use a version dep |
| `rocm-oxide-device` not on crates.io | Device API crate |
| `rocm-oxide-kernel` not on crates.io | Proc-macro crate |
| `rocm-oxide-build` not distributed | Build tool has no release binary or crates.io presence |
| Generated local `build.rs` uses source-workspace to find the build tool | Needs a well-known binary or crates.io `build-dependencies` entry |
| Git scaffolds still require `rocm-oxide-build` on PATH or `ROCM_OXIDE_BUILD` | Git deps remove path dependencies, not the build-tool distribution problem |
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
fully source-workspace-backed workflow; git scaffold mode is the supported
bridge for projects that should not preserve local path dependencies.
