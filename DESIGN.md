# Design: Rust Kernels for AMD ROCm

## Goal

Build a CUDA Oxide-like stack for AMD GPUs:

1. Write host and device code in Rust.
2. Compile device functions into AMDGPU code objects.
3. Launch kernels through a safe Rust ROCm runtime facade.
4. Keep escape hatches for HIP, HSA, rocBLAS, rocRAND, and hand-written LLVM IR.

## Why This Is Plausible

ROCm already uses LLVM's AMDGPU backend. HIP-Clang compiles HIP device code
through that backend into AMDGPU code objects. HIPRTC can compile kernels at
runtime and the HIP module APIs can load the resulting binary.

That means the hard part is not inventing an AMD GPU backend from scratch. The
hard part is connecting Rust device code to the existing AMDGPU backend with the
right ABI, address spaces, intrinsics, metadata, and code-object packaging.

## Architecture

```text
Rust host crate
  |
  |-- #[kernel] Rust device functions
  |
rocm-oxide-macros
  |
  |-- provides #[kernel], exports stable symbols, records launch metadata
  |
rocm-oxide-compiler
  |
  |-- rustc driver / MIR extractor / restricted frontend
  |-- lowers device subset to LLVM IR for amdgcn-amd-amdhsa
  |-- injects AMDGPU intrinsics, address spaces, and kernel metadata
  |-- emits .bc, .ll, or .hsaco
  |
rocm-oxide-runtime
  |
  |-- HIP/HSA FFI
  |-- module loading
  |-- typed buffers
  |-- streams/events
  |-- kernel launch
```

## MVP Scope

The first compiler MVP should be intentionally small:

- `no_std` device functions
- `#[kernel]` entry points
- raw pointers and slices lowered to explicit pointer + length pairs
- `u32`, `i32`, `usize`, `f32`, `f64`
- arithmetic, comparisons, branches, simple counted loops
- thread/block/grid intrinsics
- global memory loads/stores
- one-dimensional launches

Do not start with traits, panics, allocation, formatting, atomics, dynamic
dispatch, async, or generics beyond monomorphized functions.

## Runtime API Sketch

```rust
let device = rocm_oxide::Device::first()?;
let stream = device.create_stream()?;

let a = device.copy_from(&host_a)?;
let b = device.copy_from(&host_b)?;
let mut out = device.alloc::<f32>(host_a.len())?;

unsafe {
    vector_add
        .with_grid((blocks, 1, 1))
        .with_block((256, 1, 1))
        .launch(&stream, (&mut out, &a, &b, host_a.len()))?;
}

let result = out.copy_to_vec(&stream)?;
```

## Compiler Strategy Options

### Option A: rustc LLVM Backend Path

Use rustc's existing LLVM path and teach a driver/build tool to compile selected
device crates for `amdgcn-amd-amdhsa`.

Status: chosen for the next phase. A nightly spike already emits AMDGPU LLVM IR
from a `#![no_std]` Rust device crate, rewrites the exported function into a
launchable `amdgpu_kernel`, links a `.hsaco`, and launches it through HIP.

Pros:

- reuses Rust type checking and monomorphization
- potentially closest to real Rust semantics
- better long-term parity with CUDA Oxide-style workflows

Cons:

- requires unstable/internal rustc work
- AMDGPU address spaces and kernel ABI need careful control
- harder to keep compatible across nightly Rust versions

### Option B: Restricted Rust Frontend to LLVM IR

Parse a restricted Rust kernel subset with `syn`, type-check minimally, and emit
LLVM IR or MLIR-like IR directly.

Pros:

- smaller MVP
- predictable kernel subset
- easier to inspect generated IR

Cons:

- not true Rust semantics
- grows into a compiler maintenance burden
- less satisfying for real Rust libraries

### Option C: Rust DSL to HIP C++

Generate HIP C++ from a Rust macro DSL, then compile with HIPRTC/HIP-Clang.

Pros:

- easiest to ship quickly
- uses mature HIP compiler path
- good for experiments and kernels like elementwise ops

Cons:

- not actually Rust device code
- weak debugging and type fidelity
- limited value over writing HIP

The recommended path is A for the real project. C remains useful only as a
runtime validation harness, and B is now a fallback if rustc internals become the
limiting factor.

## Main Risks

- Rust's GPU target support is not a stable product surface.
- The AMDGPU kernel ABI requires exact metadata and address-space handling.
- ROCm consumer GPU support varies by ROCm version and OS.
- A useful project needs both compiler correctness tests and runtime GPU tests.
- Performance work requires ISA inspection, occupancy checks, and memory-layout
  tuning; correctness alone is not enough.
