# Stream-Ordered Allocation

ROCm-Oxide exposes HIP's stream-ordered allocation path through
`DeviceBuffer::new_async`, `DeviceBuffer::new_from_pool_async`, and
`DeviceBuffer::free_async`.

Local `gfx1100` note: on the RX 7900 XT behind this machine's chipset PCIe
switch, `hipMallocFromPoolAsync` can return a pointer that faults during an
immediate async device-to-host copy even though the allocation call succeeds.
The showcase therefore exercises current-pool controls and trimming, but avoids
the hard-faulting explicit-pool allocation path until that HIP/runtime behavior
is revalidated. Prefer `DeviceBuffer::new_async` for smoke coverage on this
machine.

Rules for using it safely:

- Allocate, use, copy, and free a stream-ordered buffer on the same stream unless
  an explicit event or graph dependency orders the streams.
- Keep the `DeviceBuffer` alive until the stream reaches the last queued use of
  its pointer. Generated `DeviceOperation` bindings retain their module and
  `Arc<DeviceBuffer<_>>` arguments in `KernelLaunchCompletion` for this reason.
- Prefer `free_async` for buffers allocated with `new_async` or
  `new_from_pool_async` when explicit ordering is useful. Async-created
  `DeviceBuffer` values retain their allocation stream; dropping one enqueues a
  matching `hipFreeAsync` on that stream and waits for the cleanup so error paths
  after async enqueues do not fall back to unordered `hipFree`.
- Host slices passed to async copies must stay alive and unmodified until the
  stream reaches the copy. Pinned host buffers make this ordering explicit and
  are preferred for sustained async transfer paths.
- Memory-pool attributes affect future stream-ordered allocations from that
  pool. `MemPool::set_release_threshold`, reuse toggles, statistics, and
  `trim_to` wrap the installed HIP runtime's default/current pool controls.

`Device::default_mem_pool` returns the device default pool. `Device::set_mem_pool`
sets the current pool used by `hipMallocAsync`; `DeviceBuffer::new_from_pool_async`
uses a specific pool directly.

`Device::create_mem_pool` creates an owned HIP memory pool that destroys itself
on drop. `MemPool::set_access` and `MemPool::access` expose HIP pool access
descriptors for peer/multi-device policy work. The local root test verifies
custom-pool release-threshold and access-policy round trips without using the
known hard-faulting explicit-pool allocation path on this machine.

Explicit HIP graph allocation nodes are available through
`Graph::add_mem_alloc_node`, which returns a `GraphMemoryAllocation` carrying the
allocation node, byte size, and graph-managed device pointer. Add a free node
with `GraphMemoryAllocation::add_free_node` after the last graph node that uses
the pointer. This gives generated operation pipelines a concrete allocation-plan
object without depending on the explicit-pool allocation path that is currently
fragile on the local `gfx1100` PCIe topology.

`Device::reserve_virtual_memory` exposes a first HIP VMM path for device-local
virtual memory: reserve address space, create an allocation handle, map it, set
device read/write access, and clean up in reverse order on drop. It is intended
as a low-level building block for future multi-device allocation-plan policies.
