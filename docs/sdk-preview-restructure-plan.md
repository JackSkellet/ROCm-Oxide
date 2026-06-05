# SDK Preview Restructure Plan

This is the working plan for turning ROCm-Oxide from a powerful SDK preview
workspace into a cleaner project that is easier for new users to enter, browse,
and copy from.

The guiding strategy is:

> Make ROCm-Oxide the most approachable Rust-first way to write and run AMD GPU
> code.

This is not a CUDA emulator, a full ML framework, or a graphics engine. Those
may become useful later, but the SDK path is the product: Rust-authored kernels,
HSACO generation, metadata, typed bindings, HIPRTC, ROCm runtime APIs,
diagnostics, validation, and examples.

## Current Status

| Area | Status | Evidence |
| --- | --- | --- |
| Clean root SDK | Done | Root examples are limited to SDK, diagnostics, and release probes; heavy demos live in `demo-projects/`. |
| Separate demo catalogue | Done | `demo-projects/README.md` and per-demo manifests own visual, benchmark, capture, and experiment demos. |
| Shrink root dependencies | Done | Root `Cargo.toml` no longer carries visual/demo-only dependency sets. |
| Docs during move | Done | `README.md`, `docs/index.md`, `examples/README.md`, `docs/visual-demos.md`, and project-generation docs point at the new layout. |
| Condense documentation | Done for repo docs | Maintained docs are short and linked from `docs/index.md`; long-form notes are retained under `docs/wiki/` as wiki source. |
| Improve abstractions | Second pass started | Device preludes, `#[kernel_contract]`, generated launch builders, `rocm_oxide::gpu`, and `gpu_test!` are available. The device API now also has autocomplete-friendly `element_index()`, `for_each_element(...)`, bounded slice `read/set` helpers, and generated starter kernels that use them. |
| Release readiness | gfx1201 full gate passed; gfx1100 still required before tag | Doctor JSON/GitHub modes, first-user-path gate, full local `gfx1201` gate, artifact audit, and known-good release profile docs are in place. The release checklist still requires the `gfx1100` quick/full gate before tagging. |

## Phase 1: Clean The Root

Goal: make the root crate feel like the SDK, not a demo dumping ground.

Keep the root focused on:

- `src/`: host runtime/library API
- `crates/`: device/runtime support crates
- `tools/`: build tool and cargo wrapper
- the reference Rust device crate used by source-workspace examples
- a small set of canonical examples only

Move heavy visual, capture, stress, and experiment work out of the root example
surface. The root should stay lean enough that a new user can understand the SDK
without compiling or reading unrelated GUI/demo infrastructure.

## Phase 2: Split Demos Into Separate Projects

Goal: turn demos into a browseable catalogue instead of a large pile of files.

Large demos should become separate example project folders, each with its own:

- `Cargo.toml`
- `README.md`
- run command
- expected output or screenshot
- hardware/display notes
- focused source tree

Candidate shape:

```text
examples/
  hello-hiprtc/
  rust-device-vector-add/
  generated-bindings/
  validation-profile/
  performance-probe/

demo-projects/
  spectral-lattice/
  matrix-lens/
  path-reconstruction/
  stress-gui/
  compiler-feature-lab/
```

The exact names can change during implementation, but the ownership boundary
should not: each demo should own its dependencies and docs instead of inflating
the root SDK crate.

## Phase 3: Shrink Root Dependencies

Goal: make normal root builds compile only what the SDK actually needs.

Move windowing, capture, presentation, image, Vulkan-adjacent, SDL, PipeWire, and
other demo-only dependencies into demo crates or optional features. A basic
`cargo check` of the root SDK should not require unrelated demo infrastructure.

## Phase 4: Update Docs During The Move

Goal: prevent documentation drift while paths are changing.

Every file move should update the docs that mention it in the same change. The
docs that must stay accurate are:

- `README.md`
- `docs/getting-started.md`
- `docs/project-generation.md`
- `docs/troubleshooting.md`
- `docs/release.md`
- the examples/demo catalogue

The README should present one canonical first-user path. Demo docs should point
to the separated project folders instead of root example files once the split is
complete.

## Phase 5: Condense Documentation

Goal: keep versioned repo docs useful without making new users read a manual
wall.

After the root/demo split, reduce the docs to a smaller maintained set:

- `README.md`
- `docs/index.md`
- `docs/getting-started.md`
- `docs/troubleshooting.md`
- `docs/api-stability.md`
- `docs/project-generation.md`
- `docs/release.md`

Long-form design notes, historical checklists, and deep-dive material can move
to a wiki once the repo docs are stable. Do not delete versioned build/setup docs
until their replacement is live and linked.

## Phase 6: Improve Abstractions

Goal: make writing ROCm-Oxide code feel easier after the project shape is clean.

Do this after the root is lean and demos are separated. That order matters:
abstractions should be designed against the clean SDK path, not against the
current demo-heavy workspace.

Likely abstraction work:

- a device prelude
- first-class kernel contracts instead of hidden comment syntax
- more fluent generated binding launches
- a GPU test harness
- small algorithm helpers such as fill, map, reduce, scan, and sort
- second-pass device ergonomics: `element_index()`, `for_each_element(...)`,
  and bounded `DeviceSlice::read` / `DeviceSliceMut::set` helpers

## Release Ordering

Do not tag the preview release before the root split.

A preview release should show the shape users are expected to copy:

1. clean root SDK
2. separate demo catalogue
3. accurate docs
4. repeatable diagnostics and release gates

Only after that should the project move toward an SDK preview tag.
