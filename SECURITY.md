# Security Policy

ROCm-Oxide is a `0.x` project with experimental low-level GPU and FFI surfaces.
Security-sensitive fixes should prioritize memory safety, host/device ABI
validation, and deterministic diagnostics.

## Supported Versions

Only the current `main` branch is supported for security fixes before the first
tagged production release.

## Reporting

If this repository is hosted on a platform with private security advisories,
use that advisory channel. Otherwise, contact the maintainer out of band before
publishing details. Avoid posting exploit payloads, secrets, or machine-specific
credentials in public issues.

Useful reports include:

- affected ROCm version, GPU architecture, and `validation_profile.json`;
- exact command and environment variables;
- whether the issue involves host memory, device memory, generated bindings,
  HIPRTC/COMGR compilation, optional libraries, or dynamic loading;
- a minimized reproducer that does not include private data.

## Scope

In scope:

- host memory unsafety caused by safe APIs;
- generated binding ABI mismatches;
- graph/VMM/stream-order lifetime bugs;
- dynamic-library loading or missing-symbol confusion that can call the wrong
  ABI;
- cache-key collisions that reuse incompatible code objects.

Out of scope:

- unsupported CUDA binary compatibility expectations;
- issues requiring unsupported ROCm versions or GPU architectures unless they
  also affect the supported matrix;
- denial-of-service from intentionally malformed local source code beyond
  expected compiler diagnostics.
