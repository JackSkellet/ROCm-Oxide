# Kernel Contracts

`rocm-oxide-build` can attach source-level host validation contracts to a
Rust-authored GPU kernel. Contracts are written as line comments immediately
above a `#[kernel]` function:

```rust
// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(depth)=pixel_count/4
#[kernel]
pub unsafe extern "C" fn depth_aware_upscale(
    frame: *mut u32,
    color: *const u32,
    depth: *const f32,
    pixel_count: usize,
    mode: u32,
    block_x: u32,
) {
    /* device code */
}
```

The build tool parses these contracts before it emits the generated Rust host
bindings and metadata JSON.

## Supported Contract

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

## Validation Rules

The build tool rejects:

- contracts whose buffer name is not a pointer argument
- duplicate contracts for the same buffer
- expressions that reference non-scalar arguments
- malformed expressions or unsupported characters

For older simple kernels with a scalar argument named `n`, generated bindings
still validate every pointer argument against `n` when no explicit contracts are
present. Explicit contracts take precedence, so mixed-resolution kernels can
state exact per-buffer requirements instead of pretending all buffers have the
same length.

## Current Limitations

Contracts only describe minimum buffer lengths. They do not yet express aliasing
rules, element alignment beyond the `DeviceBuffer<T>` type, scalar ABI width
metadata, multidimensional shapes, or dynamic packed scene layouts. Those should
be added as separate contract kinds rather than hidden in comments or examples.

The generated metadata now also records code-object facts parsed from the AMDGPU
note section: argument offsets, ABI sizes, pointer address spaces, value kinds,
kernarg segment size/alignment, max workgroup size, static LDS bytes, private
segment bytes, SGPR/VGPR counts, spill counts, wavefront size, and dynamic-stack
usage. Those are artifact facts rather than source contracts.
