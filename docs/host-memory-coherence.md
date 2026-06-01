# Host Memory Coherence

ROCm-Oxide models host-visible memory by separating allocation type from
visibility guarantees.

Runtime pieces:

- `Device::properties()` returns the HIP attributes that affect launch planning
  and host visibility: managed-memory support, concurrent managed access,
  direct managed host access, host mapping, host-native atomics, pageable memory
  access, stream memory pools, register support, async engine count, CU count,
  and wavefront size.
- `Device::all()`, `Device::at(ordinal)`, `Device::can_access_peer`,
  `enable_peer_access`, and `disable_peer_access` expose multi-device and peer
  probes without assuming more than one GPU is present.
- `ManagedBuffer<T>` wraps `hipMallocManaged`. `ManagedBuffer::new_zeroed`
  requests the default fine-grain model. `new_zeroed_coarse_grained` applies
  `hipMemAdviseSetCoarseGrain` to the currently selected HIP device.
- `PinnedHostBuffer::new_zeroed_mapped_coherent` remains the explicit mapped
  coherent host path for zero-copy GPU access.
- `DeviceProperties::mapped_host_reference_capture_kind()` and
  `managed_host_reference_capture_kind()` expose the same classification for
  kernels that capture a host-side pointer/reference in a closure environment.
  Use `AtomicMemoryKind::host_reference_capture_visibility()` to decide whether
  that captured reference is device-only, visible after synchronization, or
  visible while the kernel is running.

Coherence rules:

- Device memory and fine-grained device allocations are device-visible only for
  host concurrency purposes. A system-scope atomic does not make ordinary device
  memory host-pollable while a kernel is running.
- Mapped coherent host memory is modeled as host-visible during a kernel only
  when `DeviceProperties::host_native_atomic_supported` is also true. On PCIe
  paths without native atomics, system-scope atomic smoke tests skip this memory
  kind instead of advertising host-concurrent visibility.
- Fine-grain managed memory is modeled as host-visible during a kernel only when
  both `DeviceProperties::concurrent_managed_access` and
  `DeviceProperties::host_native_atomic_supported` are true. Otherwise it is
  downgraded to synchronization-boundary visibility.
- Coarse-grain managed memory is host-visible after synchronization.
- Host-reference closure captures are valid only when the captured pointer comes
  from a memory kind classified by these helpers. Mapped coherent host memory and
  fine-grain managed memory can be host-visible during a kernel when the device
  reports the required native atomic/concurrent-access attributes; coarse-grain
  managed memory is limited to synchronization boundaries.
- Peer access is explicit. Query it first with `Device::can_access_peer`; enable
  it for the current source device only when HIP reports support.

Generated bindings still validate buffer length, aliasing, block shape, and LDS
contracts. Host-memory kind selection controls visibility expectations around
those validated launches; it does not bypass synchronization requirements.

Reference points:

- AMD HIP coherence control documents fine/coarse-grain allocation behavior,
  including mapped host memory and `hipMemAdviseSetCoarseGrain`.
- AMD HIP unified-memory documentation identifies the device attributes used to
  detect managed-memory and concurrent-managed-access support.
- AMD HIP peer-to-peer memory access documentation requires explicit peer
  capability probes before enabling direct access.
