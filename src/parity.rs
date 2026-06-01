use crate::{DeviceProperties, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaPortingConcept {
    ThreadBlockCluster,
    TensorMemoryAccelerator,
    WarpGroupMma,
    NvvmLtoIr,
    NvJitLink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMathBackend {
    RocBlasLibrary,
    RocWmmaDeviceFragments,
    TiledRustKernel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmFeaturePlan {
    pub concept: CudaPortingConcept,
    pub replacement: &'static str,
    pub requires_runtime_capability: bool,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmFeatureSet {
    pub cluster_launch: RocmFeaturePlan,
    pub tile_memory_transfer: RocmFeaturePlan,
    pub matrix_math: RocmFeaturePlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmWorkgroupClusterPlan {
    pub cooperative_launch: bool,
    pub multiprocessor_count: u32,
    pub workgroups_per_multiprocessor: u32,
    pub max_resident_workgroups: u32,
    pub block_threads: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmTileTransferPlan {
    pub tile_bytes: usize,
    pub stage_count: u32,
    pub staged_lds_bytes: usize,
    pub stream_ordered_copy: bool,
    pub host_mapped_staging: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmMatrixMathPlan {
    pub backend: MatrixMathBackend,
    pub wavefront_size: u32,
    pub uses_matrix_cores: bool,
    pub requires_external_library: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmCodeObjectInteropPlan {
    pub lto_ir: RocmFeaturePlan,
    pub jit_link: RocmFeaturePlan,
    pub source_ir: &'static str,
    pub compile_link_backend: &'static str,
    pub load_backend: &'static str,
    pub library_backend: &'static str,
    pub cache_key: &'static str,
    pub cuda_binary_compatible: bool,
}

pub fn rocm_feature_parity_for_device(properties: DeviceProperties) -> RocmFeatureSet {
    RocmFeatureSet {
        cluster_launch: RocmFeaturePlan {
            concept: CudaPortingConcept::ThreadBlockCluster,
            replacement: if properties.cooperative_launch {
                "HIP cooperative grid launch plus explicit global-memory rendezvous"
            } else {
                "stream/graph-scheduled workgroup tiles with explicit global-memory rendezvous"
            },
            requires_runtime_capability: properties.cooperative_launch,
            notes: "ROCm does not expose CUDA DSMEM clusters directly; keep the cross-workgroup contract explicit.",
        },
        tile_memory_transfer: RocmFeaturePlan {
            concept: CudaPortingConcept::TensorMemoryAccelerator,
            replacement: "stream-ordered HIP copies into device buffers plus LDS/shared-memory tile staging",
            requires_runtime_capability: properties.async_engine_count > 0,
            notes: "Model TMA ports as staged transfers with visible buffer ownership and launch-time LDS sizing.",
        },
        matrix_math: RocmFeaturePlan {
            concept: CudaPortingConcept::WarpGroupMma,
            replacement: "rocWMMA or rocBLAS-backed matrix math, with tiled Rust kernels as the portable fallback",
            requires_runtime_capability: properties.warp_size == 32 || properties.warp_size == 64,
            notes: "WGMMA is NVIDIA-specific; the AMD path is wavefront/block fragments or a ROCm library call.",
        },
    }
}

pub const fn rocm_code_object_interop_plan() -> RocmCodeObjectInteropPlan {
    RocmCodeObjectInteropPlan {
        lto_ir: RocmFeaturePlan {
            concept: CudaPortingConcept::NvvmLtoIr,
            replacement: "AMDGPU LLVM IR, LLVM bitcode, or HIP source passed through COMGR",
            requires_runtime_capability: true,
            notes: "NVVM and LTOIR are NVIDIA formats; keep interchange at the source/IR layer and retarget to AMDGPU before code-object emission.",
        },
        jit_link: RocmFeaturePlan {
            concept: CudaPortingConcept::NvJitLink,
            replacement: "COMGR or ROCm clang linking relocatable AMDGPU objects into HSACO code objects",
            requires_runtime_capability: true,
            notes: "ROCm loads executable code objects through HIP modules or HIP library APIs; do not promise PTX, cubin, or nvJitLink ABI compatibility.",
        },
        source_ir: "Rust-authored AMDGPU LLVM IR, LLVM bitcode, or HIP source",
        compile_link_backend: "COMGR compile/link backend with ROCm clang as the offline generated-artifact fallback",
        load_backend: "hipModuleLoadData/hipModuleGetFunction with generated metadata validation",
        library_backend: "optional ROCm library FFI such as rocBLAS, rocFFT, rocPRIM/hipCUB, hipBLASLt, and Composable Kernel",
        cache_key: "backend + architecture + source/object inputs + options + launch metadata",
        cuda_binary_compatible: false,
    }
}

impl RocmWorkgroupClusterPlan {
    pub fn for_device(
        properties: DeviceProperties,
        workgroups_per_multiprocessor: u32,
        block_threads: u32,
    ) -> Result<Self> {
        if workgroups_per_multiprocessor == 0 {
            return Err(Error::InvalidLaunch(
                "workgroups per multiprocessor must be nonzero".to_string(),
            ));
        }
        if block_threads == 0 {
            return Err(Error::InvalidLaunch(
                "cluster replacement block size must be nonzero".to_string(),
            ));
        }
        let max_resident_workgroups = properties
            .multiprocessor_count
            .checked_mul(workgroups_per_multiprocessor)
            .ok_or_else(|| {
                Error::InvalidLaunch(
                    "cluster replacement resident-workgroup count overflows u32".to_string(),
                )
            })?;
        Ok(Self {
            cooperative_launch: properties.cooperative_launch,
            multiprocessor_count: properties.multiprocessor_count,
            workgroups_per_multiprocessor,
            max_resident_workgroups,
            block_threads,
        })
    }
}

impl RocmTileTransferPlan {
    pub fn for_2d_tile(
        properties: DeviceProperties,
        element_size_bytes: usize,
        tile_width: usize,
        tile_height: usize,
        stage_count: u32,
    ) -> Result<Self> {
        if element_size_bytes == 0 || tile_width == 0 || tile_height == 0 || stage_count == 0 {
            return Err(Error::InvalidLaunch(
                "tile transfer element size, dimensions, and stage count must be nonzero"
                    .to_string(),
            ));
        }
        let tile_bytes = element_size_bytes
            .checked_mul(tile_width)
            .and_then(|value| value.checked_mul(tile_height))
            .ok_or_else(|| {
                Error::InvalidLaunch("tile transfer byte size overflows usize".to_string())
            })?;
        let staged_lds_bytes = tile_bytes
            .checked_mul(stage_count as usize)
            .ok_or_else(|| {
                Error::InvalidLaunch("staged tile byte size overflows usize".to_string())
            })?;
        Ok(Self {
            tile_bytes,
            stage_count,
            staged_lds_bytes,
            stream_ordered_copy: properties.async_engine_count > 0,
            host_mapped_staging: properties.can_map_host_memory,
        })
    }
}

impl RocmMatrixMathPlan {
    pub const fn library_backed(properties: DeviceProperties) -> Self {
        Self {
            backend: MatrixMathBackend::RocBlasLibrary,
            wavefront_size: properties.warp_size,
            uses_matrix_cores: true,
            requires_external_library: true,
        }
    }

    pub const fn tiled_kernel(properties: DeviceProperties) -> Self {
        Self {
            backend: MatrixMathBackend::TiledRustKernel,
            wavefront_size: properties.warp_size,
            uses_matrix_cores: false,
            requires_external_library: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props() -> DeviceProperties {
        DeviceProperties {
            ordinal: 0,
            managed_memory: true,
            concurrent_managed_access: true,
            cooperative_launch: true,
            cooperative_multi_device_launch: false,
            direct_managed_mem_access_from_host: false,
            can_map_host_memory: true,
            can_use_host_pointer_for_registered_mem: false,
            host_native_atomic_supported: true,
            pageable_memory_access: false,
            pageable_memory_access_uses_host_page_tables: false,
            memory_pools_supported: true,
            unified_addressing: true,
            host_register_supported: true,
            async_engine_count: 2,
            multiprocessor_count: 32,
            warp_size: 32,
            clock_instruction_rate_khz: 100_000,
            wall_clock_rate_khz: 100_000,
        }
    }

    #[test]
    fn maps_cuda_specific_concepts_to_rocm_replacements() {
        let features = rocm_feature_parity_for_device(props());
        assert_eq!(
            features.cluster_launch.concept,
            CudaPortingConcept::ThreadBlockCluster
        );
        assert!(features.cluster_launch.replacement.contains("cooperative"));
        assert_eq!(
            features.tile_memory_transfer.concept,
            CudaPortingConcept::TensorMemoryAccelerator
        );
        assert_eq!(
            features.matrix_math.concept,
            CudaPortingConcept::WarpGroupMma
        );
    }

    #[test]
    fn plans_resident_workgroups_from_cu_count() {
        let plan = RocmWorkgroupClusterPlan::for_device(props(), 2, 256).unwrap();
        assert!(plan.cooperative_launch);
        assert_eq!(plan.max_resident_workgroups, 64);
    }

    #[test]
    fn computes_staged_tile_bytes() {
        let plan = RocmTileTransferPlan::for_2d_tile(props(), 4, 32, 16, 2).unwrap();
        assert_eq!(plan.tile_bytes, 2048);
        assert_eq!(plan.staged_lds_bytes, 4096);
        assert!(plan.stream_ordered_copy);
        assert!(plan.host_mapped_staging);
    }

    #[test]
    fn maps_nvvm_and_nvjitlink_to_rocm_artifact_model() {
        let plan = rocm_code_object_interop_plan();
        assert_eq!(plan.lto_ir.concept, CudaPortingConcept::NvvmLtoIr);
        assert_eq!(plan.jit_link.concept, CudaPortingConcept::NvJitLink);
        assert!(plan.compile_link_backend.contains("COMGR"));
        assert!(plan.load_backend.contains("hipModuleLoadData"));
        assert!(!plan.cuda_binary_compatible);
    }
}
