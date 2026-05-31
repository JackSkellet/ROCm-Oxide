# ROCm Feature Parity Replacements

ROCm-Oxide does not try to copy CUDA-only execution features by name. The
runtime now exposes a small AMD-specific parity layer in `src/parity.rs` and
uses HIP capabilities where they exist.

## CUDA Concept Mapping

| CUDA concept | ROCm-Oxide replacement |
| --- | --- |
| Thread block clusters | HIP cooperative grid launch when the device reports `hipDeviceAttributeCooperativeLaunch`; otherwise stream/graph-scheduled workgroup tiles with an explicit global-memory rendezvous. |
| Tensor Memory Accelerator | Stream-ordered HIP copies into device buffers, then explicit LDS/shared-memory tile staging sized through `LaunchConfig::shared_mem_bytes`. |
| WGMMA | rocWMMA-style wavefront fragments where that stack is installed, rocBLAS/hipBLAS library calls for host-orchestrated GEMM, and tiled Rust kernels as the portable fallback. |

## Runtime Surface

- `DeviceProperties` now includes `cooperative_launch` and
  `cooperative_multi_device_launch`.
- `Device::supports_cooperative_launch()` and
  `Device::supports_cooperative_multi_device_launch()` expose direct probes.
- `Kernel::launch_cooperative_raw_on_stream()` wraps
  `hipModuleLaunchCooperativeKernel` for module-loaded kernels.
- `validate_cooperative_launch_config()` keeps HIP's per-dimension
  `grid * block < 2^32` cooperative-launch limit explicit before launch.
- `rocm_feature_parity_for_device()` turns a probed device into a
  `RocmFeatureSet` for code generators and examples.

The important boundary is still explicit: CUDA DSMEM clusters, Hopper TMA, and
NVIDIA WGMMA are not promised as ABI-compatible concepts. Ports should use the
replacement plan as a source-level rewrite target, then let generated bindings
validate buffer ownership, LDS sizing, and launch shape before the HIP call.

Primary references:
[HIP device attributes](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html),
[HIP module/cooperative launch API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html),
[HIP stream ordered allocator](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html),
[rocWMMA](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/),
[rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/).
