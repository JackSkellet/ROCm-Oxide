# ROCm Feature Parity Replacements

ROCm-Oxide does not try to copy CUDA-only execution features by name. The
runtime now exposes a small AMD-specific parity layer in `src/parity.rs` and
uses HIP capabilities where they exist.

## CUDA Concept Mapping

| CUDA concept | ROCm-Oxide replacement |
| --- | --- |
| Thread block clusters | HIP cooperative grid launch when the device reports `hipDeviceAttributeCooperativeLaunch`; otherwise stream/graph-scheduled workgroup tiles with an explicit global-memory rendezvous. |
| Tensor Memory Accelerator | Stream-ordered HIP copies into device buffers, then explicit LDS/shared-memory tile staging sized through `LaunchConfig::shared_mem_bytes`. |
| WGMMA | Checked hipBLASLt SGEMM execution for the host-orchestrated GEMM path, Composable Kernel and rocWMMA candidate/probe reporting only, and tiled Rust kernels as the portable fallback. |
| NVVM/LTOIR | AMDGPU LLVM IR, LLVM bitcode, or HIP source that is retargeted before code-object emission. |
| nvJitLink | COMGR or ROCm `clang` links relocatable AMDGPU objects into executable HSACO code objects; loading uses HIP module/library APIs. |

## Runtime Surface

- `DeviceProperties` now includes `cooperative_launch` and
  `cooperative_multi_device_launch`.
- `Device::supports_cooperative_launch()` and
  `Device::supports_cooperative_multi_device_launch()` expose direct probes.
- `Kernel::launch_cooperative_raw_on_stream()` wraps
  `hipModuleLaunchCooperativeKernel` for module-loaded kernels.
- `Kernel::launch_cooperative_multi_device_raw()` validates each entry's launch
  shape, cooperative multi-device support, and occupancy-derived resident grid
  capacity before delegating to HIP.
- `hip::launch_cooperative_multi_device()` wraps
  `hipModuleLaunchCooperativeKernelMultiDevice` for callers that have already
  validated every device, stream, launch shape, and raw ABI parameter list.
- `validate_cooperative_launch_config()` keeps HIP's per-dimension
  `grid * block < 2^32` cooperative-launch limit explicit before launch.
- `validate_cooperative_launch_for_device()` checks cooperative-launch support
  and the occupancy-derived resident block capacity. `Kernel` cooperative
  launches call this path before reaching HIP.
- `DeviceLimits` includes HIP max-grid dimensions, so checked launch paths
  reject oversized grids before the module launch call.
- Explicit HIP graph support currently covers empty/dependency nodes, 1D
  memcpy, typed host-to-device/device-to-host/device-to-device memcpy helpers,
  memset, kernel nodes, graph memory allocation/free nodes, instantiate/replay,
  node retargeting, and graph exec update. Event nodes and host-callback nodes
  are not implemented runtime surfaces yet.
- `rocm_feature_parity_for_device()` turns a probed device into a
  `RocmFeatureSet` for code generators and examples.
- `rocm_advanced_hardware_rewrite_plan()` makes CUDA thread-block clusters,
  DSMEM, TMA, and WGMMA explicit source-level rewrite targets with
  `abi_compatible=false`.
- `rocm_code_object_interop_plan()` defines the AMD artifact replacement for
  NVIDIA NVVM/LTOIR and nvJitLink flows: AMDGPU IR, COMGR/clang code-object
  linking, HIP module loading, optional ROCm library FFI, and cache keys over
  backend, architecture, source/object inputs, options, and launch metadata.

The important boundary is still explicit: CUDA DSMEM clusters, Hopper TMA, and
NVIDIA WGMMA are not promised as ABI-compatible concepts. NVVM, LTOIR, PTX,
cubin, and nvJitLink artifacts are not accepted as ROCm binary contracts.
Composable Kernel and rocWMMA are reported as availability candidates until a
real execution wrapper lands; hipBLASLt SGEMM is the current checked matrix
library execution path. Candidate reporting must not be described as kernel
execution or feature completion.
Ports should use the replacement plan as a source-level rewrite target, then
use ROCm-Oxide runtime and generated-binding checks for launch shape,
cooperative-launch support, resident cooperative grids, rendezvous buffers,
async-copy/lifetime contracts, LDS sizing, matrix-layout/workspace requirements,
and artifact metadata before the HIP or ROCm-library call.

Primary references:
[HIP device attributes](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html),
[HIP module/cooperative launch API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html),
[HIP stream ordered allocator](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html),
[rocWMMA](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/),
[rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/).
