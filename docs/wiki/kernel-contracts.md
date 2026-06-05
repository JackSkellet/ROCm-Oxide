# Kernel Contracts

`rocm-oxide-build` can attach source-level host validation contracts to a
Rust-authored GPU kernel. Contracts are written as line comments immediately
above a `#[kernel]` function:

```rust
// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(depth)=pixel_count/4
// rocm-oxide: disjoint(color,depth)
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
bindings and metadata JSON.

## Supported Contracts

### Length

```text
// rocm-oxide: len(<buffer_arg>)=<usize expression>
```

The left side must name a pointer argument from the kernel signature. The right
side may use scalar kernel arguments, integer literals, and the operators `+`,
`-`, `*`, and `/`.

Examples:

```text
// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(motion_reactive)=pixel_count/4*3
// rocm-oxide: len(camera)=13
```

### Disjointness

```text
// rocm-oxide: disjoint(<buffer_arg>,<buffer_arg>)
```

Both names must be pointer or device-slice kernel arguments. The generated host
binding emits a disjointness check for that pair before launch. This is useful
when a kernel relies on two read-only inputs being separate allocations for
address-based indexing, staging, or future optimizer assumptions; the default
generated check only rejects pairs where at least one argument is mutable.

Examples:

```text
// rocm-oxide: disjoint(color,depth)
// rocm-oxide: disjoint(prev_frame,next_frame)
```

## Generated Binding Behavior

For a kernel contract like:

```text
// rocm-oxide: len(color)=pixel_count/4
```

the generated host binding emits:

```rust
rocm_oxide::validate_buffer_len("color", color.len(), pixel_count/4)?;
```

That means short buffers are rejected on the CPU side with a named error before
`hipModuleLaunchKernel` runs.

For a disjointness contract like:

```text
// rocm-oxide: disjoint(color,depth)
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
