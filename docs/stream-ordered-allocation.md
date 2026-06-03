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
  `DeviceBuffer` values retain their allocation stream. `Drop` is intentionally
  blocking for these allocations: it enqueues a matching `hipFreeAsync` on that
  stream and waits for cleanup so error paths after async enqueues do not fall
  back to unordered `hipFree`. Do not rely on destructor cleanup in
  latency-sensitive loops; use explicit `free_async` and stream ordering there.
- Host slices passed to async copies must stay alive and unmodified until the
  stream reaches the copy. Pinned host buffers make this ordering explicit and
  are preferred for sustained async transfer paths.
- Memory-pool attributes affect future stream-ordered allocations from that
  pool. `MemPool::set_release_threshold`, reuse toggles, statistics, and
  `trim_to` wrap the installed HIP runtime's default/current pool controls.

The `operation` module also exposes owned lazy jobs for host-to-device copies,
device-to-device copies, byte-pattern memset, and tile-plan device transfers.
Those jobs retain their `Arc<DeviceBuffer<_>>` handles and owned host inputs in
the returned completion token, so safe `DeviceOperation` execution does not
borrow memory past the stream enqueue boundary. Explicit graph memcpy helpers
remain `unsafe`: the graph does not own host slices or device buffers, so callers
must keep every pointer valid until all graph launches that can reach the node
have completed.

`StreamPool::new` creates at least one stream and rejects requests above 64
streams. The cap limits how many HIP streams ROCm-Oxide creates for pooled
operation scheduling; it is not an unbounded work queue. `DeviceOperation::async_on`
uses one host worker thread per operation, and `async_in` only chooses the next
pooled stream, so higher-level callers should bound outstanding `DeviceFuture`
counts when they need real back-pressure.

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

The explicit graph surface currently includes empty/dependency nodes, 1D memcpy,
typed H2D/D2H/D2D memcpy helpers, memset, kernel nodes, memory allocation/free
nodes, instantiate/replay, node retargeting, and graph exec update. Event nodes
and host callback nodes are not implemented.

`Device::reserve_virtual_memory` exposes a first HIP VMM path for device-local
virtual memory: reserve address space, create an allocation handle, map it, set
device read/write access, and clean up in reverse order on drop. It is intended
as a low-level building block for future multi-device allocation-plan policies.
