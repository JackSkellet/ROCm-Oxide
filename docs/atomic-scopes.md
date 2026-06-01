# ROCm Atomic Scope Notes

ROCm-Oxide exposes workgroup, device, and system intent for `u32` atomics in
device Rust. The current AMDGPU lowering is:

| ROCm-Oxide scope | LLVM IR | ISA validation |
| --- | --- | --- |
| workgroup | `syncscope("workgroup")` | global atomic instruction; `scope:SCOPE_SE` when objdump prints scope labels |
| device | `syncscope("agent")` | global atomic instruction; `scope:SCOPE_DEV` when objdump prints scope labels |
| system | backend default | global atomic instruction; `scope:SCOPE_SYS` when objdump prints scope labels |

The system-scope path intentionally leaves LLVM IR on the backend default
instead of inserting `syncscope("system")`; this local LLVM build rejects that
explicit non-inclusive syncscope, while the backend still lowers it to a global
atomic. The `gfx1201` RX 9070 XT probe printed the expected `scope:SCOPE_*`
annotations. ROCm LLVM 22 on this `gfx1100` RX 7900 XT machine emits the
expected `global_atomic_add_u32` instructions but `llvm-objdump` does not print
those scope annotations. The build therefore treats transformed LLVM IR as the
canonical scope evidence, requires the expected atomic instructions in ISA, and
validates printed scope labels whenever the disassembler provides them.

## Negative Contract

`SystemAtomicU32` does not by itself make every allocation safe for concurrent
host polling. The memory kind still controls host visibility:

- Default/coarse device memory is device-only for direct host access. Host code
  observes results after kernel completion plus an explicit copy or a future
  managed-memory synchronization path.
- Fine-grained device allocations improve device-side memory behavior, but they
  are still not direct host-visible pointers in this runtime contract.
- Mapped coherent pinned host memory is host-visible only when the device also
  reports host-native atomic support. Without host-native PCIe atomics, it is
  not used for host-concurrent system-scope atomic experiments.
- Managed coarse-grain memory is modeled as host-visible after synchronization,
  not during kernel execution.

The runtime exposes this distinction through `AtomicMemoryKind` and
`SystemScopeAtomicVisibility`; tests assert that coarse/default memory does not
get promoted to host-concurrent visibility just because the atomic scope is
system-wide.

The feature showcase currently verifies scoped atomics on default device memory
and fine-grained device memory. Mapped and managed host-visible atomic smoke
tests are gated on host-native PCIe atomics so they can run on the `gfx1201`
profile and skip on this local `gfx1100` PCIe switch topology.
