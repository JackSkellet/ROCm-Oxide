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

Runtime loading still uses the HIP module path through `hipModuleLoadData` and
`hipModuleGetFunction`, because generated bindings already know the kernel
symbols. AMD's HIP module-management API also exposes library-oriented entry
points such as `hipLibraryLoadData`, `hipLibraryGetKernel`, and kernel-count
queries. Those are useful future hooks for runtime artifact inspection, but the
current launch path should keep using the explicit generated metadata until a
runtime enumeration layer can add value without weakening validation.
