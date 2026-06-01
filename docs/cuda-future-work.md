# CUDA Feature Research And ROCm-Oxide Follow-Up Plan

ROCm-Oxide should continue to target source-level CUDA/cuda-oxide ergonomics on
AMD GPUs. It should not promise CUDA binary compatibility, PTX compatibility, or
NVIDIA driver/runtime ABI compatibility.

Primary reference set:

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [HIP cooperative groups](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/cooperative_groups_reference.html)
- [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [HIP stream ordered allocator](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html)
- [rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/)
- [hipCUB](https://rocm.docs.amd.com/projects/hipCUB/en/latest/index.html)

## CUDA Features To Track

| CUDA feature area | ROCm-Oxide direction |
| --- | --- |
| Cooperative Groups | Provide Rust device-side group handles for thread blocks, wavefronts, and static tiles. Keep host cooperative module launch as the grid-wide capability path. |
| CUDA Graphs | Move from stream capture only to a typed graph builder with kernel, memcpy, memset, event, host callback, instantiate, update, and replay support. |
| Stream-ordered memory allocator | Expand current async allocation wrappers into pool-owned allocation plans, pool trimming policy, access descriptors, and graph-capturable allocation lifetimes. |
| Unified memory and system memory | Keep explicit coarse/fine-grained managed memory kinds and document when host-concurrent visibility requires host-native atomics. |
| Virtual Memory Management | Add a low-level HIP VMM facade after pool semantics are stable: reserve/map/unmap, access flags, exportable handles, and peer mapping. |
| Cooperative launch and clusters | Treat CUDA clusters and DSMEM as a source rewrite target. Use HIP cooperative launch where supported, or explicit multi-kernel stream/graph tiling with global-memory rendezvous. |
| Asynchronous data copies, pipelines, and TMA | Model TMA-like ports as stream-ordered copies plus explicit LDS tile staging. Add pipeline/token APIs only when the backend can validate real synchronization semantics. |
| WGMMA and tensor operations | Do not emulate WGMMA by name. Route matrix/tensor work through rocWMMA, rocBLAS/hipBLASLt, Composable Kernel, or tiled Rust kernels. |
| CUB/Thrust-style algorithms | Wrap rocPRIM, hipCUB, and rocThrust operations for reduce, scan, sort, select, transform, and prefix operations over `DeviceBuffer`. |
| Dynamic parallelism | Track as a CUDA-only capability. Prefer host graph replay, persistent kernels, work queues, or cooperative multi-kernel scheduling on ROCm. |
| Runtime compilation | Use HIPRTC and/or COMGR for source-level specialization and code-object caching instead of NVRTC, PTX, or nvJitLink. |
| Cache control and synchronization domains | Track as tuning features. Expose only device-reported ROCm capabilities and keep conservative defaults when AMD has no direct analogue. |
| Graphics and external memory interop | Promote the `spectral_lattice` HIP/OpenGL PBO path into reusable graphics interop wrappers once lifecycle and synchronization rules are verified. |

## Implementation Order

1. Cooperative group device API:
   - add `ThreadBlock`, `Wavefront`, and static tile handles in
     `rocm-oxide-device`;
   - expose rank/size/index/sync helpers without hiding AMD wavefront semantics;
   - add a generated-kernel smoke test.
2. Explicit graph builder:
   - keep stream capture, but add builder nodes for kernels, memcopies, memset,
     dependencies, instantiate, replay, and graph update;
   - make generated operations optionally produce graph nodes.
   - Current status: the runtime has explicit graph creation, empty node,
     dependency, memcpy, memset, kernel-node, node-retargeting, instantiate,
     replay, and exec-update wrappers. Generated bindings now expose
     `*_graph_node` helpers for validated kernel-node insertion into explicit
     HIP graphs while `DeviceOperation::capture_graph` remains available for
     stream-capture replay.
3. Memory-pool/VMM maturity:
   - add owned async allocations tied to pools and stream/event dependencies;
   - then add HIP VMM primitives for reserve/map/unmap/access policy.
   - Current status: the runtime can create/destroy owned HIP memory pools,
     set/query pool access policy, reserve/map/access HIP VMM-backed device
     memory with RAII cleanup, and add HIP graph allocation/free nodes through a
     graph-managed allocation-plan object. Generated operation pipelines can now
     build on that object for explicit graph allocation lifetimes.
4. Device algorithm library layer:
   - wrap rocPRIM/hipCUB reductions and scans first because they are broadly
     useful and validate temporary-storage patterns;
   - then add sort/select/transform where API stability is clear.
   - Current status: ROCm-Oxide builds a small rocPRIM/hipCUB C++ shim and
     exposes `RocPrim` wrappers for `u32` sum reduction plus inclusive/exclusive
     sum scans over `DeviceBuffer`, with explicit temporary-storage objects for
     stream-ordered use.
5. Matrix/tensor layer:
   - extend the current rocBLAS path with hipBLASLt/rocWMMA or Composable
     Kernel where installed;
   - keep tiled Rust kernels as the portable fallback.
   - Current status: `MatrixIntegrationReport` identifies hipBLASLt,
     Composable Kernel, and rocWMMA availability, and `HipBlasLt` can load the
     installed library, create a handle, and query its version. On the local
     ROCm install hipBLASLt and Composable Kernel are available, while rocWMMA
     headers are not installed.
6. Runtime specialization cache:
   - cache HIPRTC/COMGR outputs by architecture, source hash, feature flags,
     and launch metadata;
   - expose this as a high-level specialization path, not a PTX/NVVM promise.
   - Current status: `Device::compile_hip_source` now uses a process-wide
     HIPRTC code-object cache keyed by compiler backend, architecture, source
     hash, compile options, and launch metadata. `Device::compile_hip_source_specialized`
     exposes option and launch-metadata keyed specialization directly. COMGR is
     probed through the optional library interop layer and kept as a distinct
     cache backend key for a future COMGR compilation path.

## Non-Goals

- Running CUDA binaries.
- Loading PTX or cubin artifacts.
- Exposing NVIDIA-specific names such as WGMMA, TMA, or DSMEM clusters as if
  they were ABI-compatible on AMD hardware.
- Claiming system-scope host visibility unless the probed ROCm device and memory
  kind support it.
