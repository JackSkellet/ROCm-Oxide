# ROCm-Oxide Docs

This is the maintained documentation set for the SDK preview. The README should
stay short and link here when a topic needs more detail.

## Start Here

- [Getting started](getting-started.md): first-user path, generated projects,
  Rust device kernels, and common commands.
- [Troubleshooting](troubleshooting.md): error messages, ROCm environment checks,
  and build/runtime fixes.
- [API stability](api-stability.md): stable, experimental, and internal surfaces
  during the `0.x` SDK preview.
- [Project generation](project-generation.md): `cargo rocm-oxide new` scaffold
  layout, path dependencies, and standalone roadmap.
- [Release process](release.md): repeatable gates for preview tags.
- [Visual demos](visual-demos.md): separated visual/capture/artifact demo
  launch table.

## Demo Catalogue

The root `examples/` directory is limited to SDK examples and diagnostics. Large
visual, capture, artifact, and benchmark demos live in
[`demo-projects/`](../demo-projects/README.md).

## Wiki Source

Long-form design notes, historical checklists, CUDA/ROCm parity research, safety
audits, and deep architecture notes are under [`docs/wiki/`](wiki/README.md).
They are kept in the repository for now so they can be edited and later copied
to GitHub Wiki pages without crowding the main docs surface.
