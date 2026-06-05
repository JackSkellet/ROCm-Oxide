# Code Object Linking

`rocm-oxide-build` treats every kernel-bearing crate in the discovered device
bundle as a separate code-object input:

1. compile the crate to AMDGPU LLVM IR,
2. rewrite only the marked kernels in that IR,
3. lower the rewritten IR to an object with ROCm `llc`,
4. link all generated objects into one HSACO with ROCm `clang`,
5. validate that the linked HSACO has every expected kernel function and
   descriptor,
6. read the linked code-object metadata and require a resource row for every
   generated kernel before host bindings are emitted.

The generated metadata now records the link graph under `link.objects`, including
the source package, rewritten LLVM IR path, object path, and kernel names
contributed by each input. This keeps the final HSACO, metadata JSON, and typed
host bindings tied to the same artifact set.

The production verification gate now audits that artifact set after GPU
verification runs. It requires the recorded link inputs to exist, requires the
linked HSACO to exist, checks that every `link.objects[*].kernels` entry has a
matching generated `kernels[*].name` metadata row, and checks that performance
probe resource rows match the embedded compiler metadata for sampled kernels.
This closes the easy metadata-drift gap where the runtime could launch current
bytes while the shipped JSON or bindings described a different link graph.

Runtime loading still uses the HIP module path through `hipModuleLoadData` and
`hipModuleGetFunction`, because generated bindings already know the kernel
symbols. AMD's HIP module-management API also exposes library-oriented entry
points such as `hipLibraryLoadData`, `hipLibraryGetKernel`, and kernel-count
queries. Those are useful future hooks for runtime artifact inspection, but the
current launch path should keep using the explicit generated metadata until a
runtime enumeration layer can add value without weakening validation.

## CUDA Artifact Interop Mapping

ROCm-Oxide does not treat NVIDIA NVVM IR, LTOIR, PTX, cubin, or nvJitLink as
binary-compatible inputs. The supported replacement model is:

1. keep source-level interchange in Rust-authored AMDGPU LLVM IR, LLVM bitcode,
   or HIP source,
2. compile with the generated LLVM/`llc`/ROCm `clang` path or the runtime COMGR
   HIP source backend,
3. link relocatable AMDGPU objects into executable HSACO code objects,
4. cache code objects by backend, architecture, source/object inputs, options,
   and launch metadata,
5. load executable bytes through HIP module APIs, and
6. call optional ROCm libraries through typed FFI wrappers when a library path
   is a better fit than generated kernels.

`rocm_code_object_interop_plan()` exposes this mapping to code generators and
examples so ports can reason about the artifact boundary without promising CUDA
ABI compatibility.
