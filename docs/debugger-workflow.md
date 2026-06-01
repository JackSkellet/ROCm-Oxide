# ROCm Debugger Workflow

ROCm-Oxide emits optimized release HSACO files by default. For ROCgdb or
ROCm-native debugger sessions, enable device debug info explicitly:

```bash
ROCM_OXIDE_DEVICE_DEBUG=1 cargo run --example rust_device_generated_bindings
```

With this flag set, `tools/rocm-oxide-build` adds `-C debuginfo=2` to the final
device-crate `cargo rustc -- ...` invocation, lowers the debug-metadata-bearing
LLVM IR with ROCm `llc`, and passes `-g` to ROCm `clang` while producing the
same `.hsaco`, metadata JSON, and generated binding paths. Build-std
dependencies still use the normal target-CPU flags so AMDGPU
`core`/`compiler_builtins` compilation stays on the stable path.
The IR post-pass also strips rustc's `dwarfAddressSpace` debug metadata field,
which this ROCm `llc` rejects, without removing the surrounding debug records.

Useful artifacts are written under:

```text
device-spike/target/amdgcn-amd-amdhsa/release/
```

The important files are:

- `rocm_oxide_device_spike*.ll` for transformed AMDGPU LLVM IR
- `rocm_oxide_device_spike*.o` for lowered ROCm object files
- `rocm_oxide_device_spike.hsaco` for the linked code object
- `rocm_oxide_device_spike.metadata.json` for kernel resource and contract facts

Recommended workflow:

1. Build once with `ROCM_OXIDE_DEVICE_DEBUG=1`.
2. Keep the generated `.ll`, `.o`, `.hsaco`, and metadata files together.
3. Launch the host example under ROCgdb or a ROCm debugger using the same ROCm
   installation that supplied `llc` and `clang`.
4. Use the metadata JSON to map generated kernel names, launch contracts, and
   resource facts back to the source-level Rust kernel declarations.

The flag does not change the default release path. Leave it unset for normal
benchmarking and demo runs.
