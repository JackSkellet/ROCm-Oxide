# ROCm Atomic Scope Notes

ROCm-Oxide exposes workgroup, device, and system intent for `u32` atomics in
device Rust. The current AMDGPU lowering is:

| ROCm-Oxide scope | LLVM IR | gfx1201 ISA observed locally |
| --- | --- | --- |
| workgroup | `syncscope("workgroup")` | `scope:SCOPE_SE` |
| device | `syncscope("agent")` | `scope:SCOPE_DEV` |
| system | backend default | `scope:SCOPE_SYS` |

The system-scope path intentionally leaves LLVM IR on the backend default
instead of inserting `syncscope("system")`; this local LLVM build rejects that
explicit non-inclusive syncscope, while the backend still lowers the global
atomic to `SCOPE_SYS`.

## Negative Contract

`SystemAtomicU32` does not by itself make every allocation safe for concurrent
host polling. The memory kind still controls host visibility:

- Default/coarse device memory is device-only for direct host access. Host code
  observes results after kernel completion plus an explicit copy or a future
  managed-memory synchronization path.
- Fine-grained device allocations improve device-side memory behavior, but they
  are still not direct host-visible pointers in this runtime contract.
- Mapped coherent pinned host memory is host-visible and is the current runtime
  path for host-concurrent system-scope atomic experiments.
- Managed coarse-grain memory is modeled as host-visible after synchronization,
  not during kernel execution.

The runtime exposes this distinction through `AtomicMemoryKind` and
`SystemScopeAtomicVisibility`; tests assert that coarse/default memory does not
get promoted to host-concurrent visibility just because the atomic scope is
system-wide.

The feature showcase currently verifies scoped atomics on default device memory,
fine-grained device memory, and mapped coherent host-visible memory.
