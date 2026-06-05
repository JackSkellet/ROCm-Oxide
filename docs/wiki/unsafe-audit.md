# Unsafe and FFI Audit

This audit records the production-readiness boundary for public `unsafe` APIs,
raw HIP/ROCm FFI wrappers, and invalid-order tests.

## Scope

The stable user-facing path is the root `rocm_oxide::*` API. Low-level modules
remain experimental, but their exposed unsafe operations still need explicit
contracts before downstream users depend on them.

Covered surfaces:

- `src/hip.rs`: HIP streams, events, buffers, modules, globals, graphs, memory
  pools, VMM, raw module launches, cooperative multi-device launch FFI, and
  async stream-ordered operations;
- `src/runtime.rs`: checked raw kernel launch, checked graph-node insertion,
  checked stream launches, and cooperative launch wrappers;
- `src/libraries.rs`: optional rocBLAS, rocFFT, hipBLASLt, COMGR, and
  rocPRIM/hipCUB dynamic-library handles and descriptors;
- `src/hiprtc.rs`: runtime compile backends and specialization cache keys;
- `crates/rocm-oxide-device`: GPU-only raw pointer, atomic, scratch, and
  device-slice helpers used by generated kernels.

## Contracts Added

- Public unsafe host APIs now name the caller-owned lifetime or ABI obligation:
  raw kernel argument lists, stream-enqueued host/output buffers, async frees,
  graph node parameters, graph-managed memory frees, raw device pointers, and
  pinned-host aliases.
- Opaque HIP and ROCm handles have Send/Sync rationale at the unsafe impl site:
  wrapper code owns creation/destruction, exposes handles but not Rust
  references into foreign state, and pairs Drop with the owning wrapper.
- Module-owned `Function` and `Global<T>` handles retain the loaded module
  owner, so `hipModuleUnload` cannot run while function or global handles still
  exist.
- Raw module/function handle accessors are unsafe and non-owning: callers must
  keep the owning `Module`/`Kernel` alive, must not unload or destroy the raw
  handles through foreign APIs, and must make the reported `device_ordinal()`
  current before passing the handle to HIP interop calls.
- Low-level cooperative multi-device launch is unsafe and caller-validated:
  each launch entry must already have device support, resident launch shape,
  stream/device ownership, and raw ABI parameter lifetimes checked before HIP.
- Graph nodes and graph memory allocations carry graph membership tokens so
  safe graph builders reject cross-graph dependencies before entering HIP.
  Supported graph node classes are empty/dependency, 1D memcpy, typed H2D/D2H/D2D
  memcpy helpers, memset, kernel, memory allocation, and memory free. Event and
  host-callback graph nodes remain unsupported.
- Async-created `DeviceBuffer` values use a blocking destructor: Drop enqueues
  `hipFreeAsync` on the retained allocation stream and synchronizes that stream
  so error paths do not silently switch to unordered `hipFree`.
- `StreamPool` construction is capped at 64 streams. The pool limits stream
  fanout, but async operation futures still need caller-side back-pressure
  because each `async_on` operation owns a host worker thread until it
  synchronizes.
- Device-side raw helpers are explicitly documented as GPU-only escape hatches:
  callers must uphold pointer address-space, alignment, aliasing, lifetime,
  memory-scope, and per-lane scratch participation rules.

## Invalid-Input Coverage

The production gate includes negative tests for these unsafe/FFI edges:

- graph cross-graph dependencies and cross-graph allocation free tokens;
- zero-byte graph allocations and null graph free pointers;
- VMM zero-size reservations and invalid HIP memory-access flag decoding;
- rocPRIM/hipCUB undersized temporary storage before launch;
- COMGR and HIPRTC specialization cache separation by backend key;
- hipBLASLt invalid SGEMM leading dimensions, nonpositive or excessive
  heuristic counts, and excessive automatic workspace caps.

HIP VMM map/unmap ordering is intentionally RAII-only in the public wrapper:
`DeviceVirtualMemory::new_for_device` reserves, creates, maps, and sets access
as one constructor, while Drop unmaps, releases, and frees the reservation in
reverse order. There is no public safe map/unmap toggle that can double-map or
double-unmap the reservation.

## Release Rule

Before promoting an experimental unsafe API to stable:

- add a per-function `# Safety` contract or document the generated function
  family at the macro/device crate boundary;
- add a negative test for the invalid pointer, lifetime, order, descriptor, or
  cache-key edge if it can be rejected before FFI;
- include any remaining ROCm capability assumption in `validation_profile.json`
  or the release notes.
