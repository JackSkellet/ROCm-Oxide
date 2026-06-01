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
  `new_from_pool_async`. Dropping a `DeviceBuffer` uses a synchronous `hipFree`,
  so callers should synchronize first when outstanding stream work may still use
  the pointer.
- Host slices passed to async copies must stay alive and unmodified until the
  stream reaches the copy. Pinned host buffers make this ordering explicit and
  are preferred for sustained async transfer paths.
- Memory-pool attributes affect future stream-ordered allocations from that
  pool. `MemPool::set_release_threshold`, reuse toggles, statistics, and
  `trim_to` wrap the installed HIP runtime's default/current pool controls.

`Device::default_mem_pool` returns the device default pool. `Device::set_mem_pool`
sets the current pool used by `hipMallocAsync`; `DeviceBuffer::new_from_pool_async`
uses a specific pool directly.
