# Onboarding Improvement Notes

These notes document friction points discovered when a consumer project was set up
**manually** instead of via `cargo rocm-oxide new`. All five issues below stem from
that single root cause, but each represents an independent gap that tooling or
documentation could close independently.

---

## 1. `cargo rocm-oxide new` is the only safe entry point — but that is not stated prominently

### What happened
A consumer project was hand-authored: `Cargo.toml`, `device-spike/Cargo.toml`,
`src/main.rs`, and `device-spike/src/lib.rs` were written from scratch. `build.rs`
and `rust-toolchain.toml` were omitted entirely. Every path dependency was wrong.

### Why it is easy to get wrong
`cargo rocm-oxide new` is listed in `print_help()` and briefly noted as a "local
scaffold" command, but the README's **Start here** section only covers
`cargo run --example hello_gpu`. A user who reads the README and decides to
start their own project has no obvious path to `cargo rocm-oxide new`.

### Suggested improvement
Add a **"Creating a consumer project"** section to `README.md` immediately after
**Start here**, showing the exact three-command sequence:

```sh
# 1. Clone (or cd into) the ROCm-Oxide workspace
git clone https://github.com/JackSkellet/ROCm-Oxide.git
cd ROCm-Oxide

# 2. Generate the scaffold — paths are computed automatically
cargo rocm-oxide new ../my-project

# 3. Build and run
cd ../my-project && cargo run
```

The section should explicitly state that **manual project creation is not
supported** and will produce incorrect path dependencies.

---

## 2. The `source_workspace_root()` error message does not include a recovery path

### What happened
`source_workspace_root()` walks up from CWD looking for
`tools/rocm-oxide-build/Cargo.toml`. When the command is run outside the clone the
error is:

```
cargo rocm-oxide new must be run from within (or adjacent to) the ROCm-Oxide
source workspace.
hint: cd into the cloned ROCm-Oxide repository, then re-run this command.
```

The hint mentions "the cloned repository" but does not say where to get it or what
the directory should be named.

### Suggested improvement
Append the clone URL and a concrete example path to the hint line:

```
hint: clone the workspace first, then cd into it and re-run:
  git clone https://github.com/JackSkellet/ROCm-Oxide.git
  cd ROCm-Oxide
  cargo rocm-oxide new <path>
```

This turns a dead-end error into a self-contained recovery procedure.

---

## 3. No command to validate an existing consumer project

### What happened
Once files were hand-authored with wrong paths, there was no tool to diagnose what
was broken. `cargo rocm-oxide doctor` checks the *source workspace*, not a consumer
project; `cargo build` reports a Cargo resolver error that points at a missing
`Cargo.toml`, not at the wrong path in the consumer's manifest.

### Suggested improvement
Add a `cargo rocm-oxide check-consumer` (or extend `doctor`) subcommand that, when
run from a consumer project directory, validates:

1. Each `path = "…"` in `Cargo.toml` and `device-spike/Cargo.toml` resolves to an
   actual `Cargo.toml` that exports the expected crate name.
2. `build.rs` exists and contains a `ROCM_OXIDE_DEVICE_BINDINGS` `rustc-env` emit.
3. `rust-toolchain.toml` exists and requests the `rust-src` component.

Output could mirror `doctor`'s pass/fail style:

```
[pass] rocm-oxide path dependency resolves
[pass] rocm-oxide-device path dependency resolves
[fail] build.rs not found — run `cargo rocm-oxide new` to regenerate scaffold
[fail] rust-toolchain.toml not found
```

---

## 4. `rust-toolchain.toml` and `build.rs` are silently absent without `cargo rocm-oxide new`

### What happened
Both files are critical:
- Without `rust-toolchain.toml` (pinning nightly + `rust-src`), `cargo build`
  uses stable Rust and fails on `-Z build-std=core` during device kernel
  compilation.
- Without `build.rs`, the `.hsaco` is never compiled and the
  `ROCM_OXIDE_DEVICE_BINDINGS` env var is never set, so the `include!()` in
  `main.rs` fails at compile time with an opaque "environment variable not found"
  error.

Neither omission produces an error message that points back to the scaffold.

### Suggested improvement
- Document both files and their purpose in a new **"Required scaffold files"**
  table in `docs/` (or inline in the README consumer section).
- In `build.rs` (the source-workspace one), add a `compile_error!` fallback so
  that if a consumer forgets their own `build.rs` the error says something like:
  `ROCM_OXIDE_DEVICE_BINDINGS not set — ensure build.rs from the scaffold is present`.
  (This is already partially handled by `env!()` panicking, but the message is
  opaque.)

---

## 5. The `cargo rocm-oxide new` help note is buried in the usage footer

### What happened
The note in `print_help()`:

```
new       Creates a LOCAL SCAFFOLD tied to this ROCm-Oxide workspace via
          relative paths. The project is not standalone and cannot be
          published to crates.io. Run from within the ROCm-Oxide workspace.
```

This is the most important constraint for new users, but it appears at the bottom
of a long usage block and only if the user runs `cargo rocm-oxide help`.

### Suggested improvement
Surface the constraint at the *top* of the `new` subcommand's output, before path
computation begins — even on success:

```
note: this project is a local scaffold tied to the ROCm-Oxide workspace at
      /path/to/ROCm-Oxide via relative path dependencies.
      Moving only the generated project will break the build.
      See docs/scaffold-portability.md for options.
```

Printing it on success (not just on error) ensures every user who runs `new`
reads it at least once.

---

## Summary

| # | Root cause | Fix category |
|---|-----------|-------------|
| 1 | README has no consumer project quickstart | Documentation |
| 2 | Error message has no recovery steps | CLI UX |
| 3 | No validator for existing consumer projects | New subcommand |
| 4 | Missing scaffold files produce opaque errors | Documentation + error message |
| 5 | Portability constraint buried in help footer | CLI UX |

All five issues are independent and can be addressed in any order.
