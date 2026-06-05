# Kernel Contracts

`rocm-oxide-build` can attach source-level host validation contracts to a
Rust-authored GPU kernel. Prefer `#[kernel_contract(...)]` immediately above a
`#[kernel]` function:

```rust
use rocm_oxide_kernel::{kernel, kernel_contract};

#[kernel_contract(len(frame)=pixel_count)]
#[kernel_contract(len(color)=pixel_count/4)]
#[kernel_contract(len(depth)=pixel_count/4)]
#[kernel_contract(disjoint(color,depth))]
#[kernel]
pub unsafe extern "C" fn depth_aware_upscale(
    frame: *mut u32,
    color: *const u32,
    depth: *const f32,
    pixel_count: usize,
    mode: u32,
) {
    /* device code */
}
```

The build tool parses these contracts before it emits the generated Rust host
bindings and metadata JSON. The older line-comment form still works:

```rust
// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: disjoint(color,depth)
#[kernel]
pub unsafe extern "C" fn legacy_contract_form(
    frame: *mut u32,
    color: *const u32,
    depth: *const f32,
    pixel_count: usize,
) {
    /* device code */
}
```

## Supported Contracts

### Length

```text
#[kernel_contract(len(<buffer_arg>)=<usize expression>)]
```

The left side must name a pointer argument from the kernel signature. The right
side may use scalar kernel arguments, integer literals, and the operators `+`,
`-`, `*`, and `/`.

Examples:

```text
#[kernel_contract(len(frame)=pixel_count)]
#[kernel_contract(len(color)=pixel_count/4)]
#[kernel_contract(len(motion_reactive)=pixel_count/4*3)]
#[kernel_contract(len(camera)=13)]
```

Multiple contracts can also be grouped into one attribute:

```rust
#[kernel_contract(len(frame)=pixel_count, len(color)=pixel_count/4)]
```

### Disjointness

```text
#[kernel_contract(disjoint(<buffer_arg>,<buffer_arg>))]
```

Both names must be pointer or device-slice kernel arguments. The generated host
binding emits a disjointness check for that pair before launch. This is useful
when a kernel relies on two read-only inputs being separate allocations for
address-based indexing, staging, or future optimizer assumptions; the default
generated check only rejects pairs where at least one argument is mutable.

Examples:

```text
#[kernel_contract(disjoint(color,depth))]
#[kernel_contract(disjoint(prev_frame,next_frame))]
```

## Generated Binding Behavior

For a kernel contract like:

```text
#[kernel_contract(len(color)=pixel_count/4)]
```

the generated host binding emits:

```rust
rocm_oxide::validate_buffer_len("color", color.len(), pixel_count/4)?;
```

That means short buffers are rejected on the CPU side with a named error before
`hipModuleLaunchKernel` runs.

For a disjointness contract like:

```text
#[kernel_contract(disjoint(color,depth))]
```

the generated host binding emits:

```rust
rocm_oxide::validate_device_buffers_disjoint("color", color, "depth", depth)?;
```

That means overlapping device allocations are rejected before launch even if
both buffers are read-only. Mutable buffer pairs are still checked by default,
with or without an explicit `disjoint(...)` contract.

## Validation Rules

The build tool rejects:

- contracts whose buffer name is not a pointer argument
- duplicate contracts for the same buffer
- duplicate `disjoint(...)` pairs, including reversed pairs
- `disjoint(...)` contracts that name non-buffer arguments
- expressions that reference non-scalar arguments
- malformed expressions or unsupported characters

For older simple kernels with a scalar argument named `n`, generated bindings
still validate every pointer argument against `n` when no explicit contracts are
present. Explicit length contracts take precedence, so mixed-resolution kernels
can state exact per-buffer requirements instead of pretending all buffers have
the same length. Disjointness contracts do not disable length fallbacks.

## Current Limitations

Length contracts only describe minimum buffer lengths. Generated bindings now
reject obvious aliasing where at least one `DeviceBuffer<T>` argument is mutable,
and `disjoint(...)` can now opt const/const pairs into the same generated
validation. Contracts do not yet express allowed aliasing, element alignment
beyond the `DeviceBuffer<T>` type, scalar ABI width metadata, multidimensional
shapes, or dynamic packed scene layouts. Those should be added as separate
contract kinds rather than hidden in comments or examples.

The generated metadata now also records code-object facts parsed from the AMDGPU
note section: argument offsets, ABI sizes, pointer address spaces, value kinds,
kernarg segment size/alignment, max workgroup size, static LDS bytes,
dynamic-LDS usage, private segment bytes, SGPR/VGPR counts, spill counts,
wavefront size, and dynamic-stack usage. Those are artifact facts rather than
source contracts. Generated bindings expose them as `rocm_oxide::KernelResource`
entries for host-side planning and benchmark reporting.
