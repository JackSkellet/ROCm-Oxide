#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/rocm-oxide-consumer-smoke.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

mkdir -p "$WORKDIR/src"

cat > "$WORKDIR/Cargo.toml" <<EOF
[package]
name = "rocm-oxide-consumer-smoke"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
rocm-oxide = { path = "$ROOT" }
EOF

cat > "$WORKDIR/src/main.rs" <<'EOF'
use rocm_oxide::{
    AtomicMemoryKind, DeviceLimits, DeviceSlice, DeviceSliceMut, Dim3, KernelMetadata,
    LaunchConfig, SystemScopeAtomicVisibility, validate_block_x, validate_buffer_len,
    validate_cooperative_launch_config, validate_launch_config_for_limits,
};

fn main() -> rocm_oxide::Result<()> {
    let config = LaunchConfig::for_num_elems_with_block_size(1024, 128)
        .try_with_dynamic_shared_mem::<u32>(32)?;
    validate_launch_config_for_limits(config, DeviceLimits::prototype(), KernelMetadata::default())?;
    validate_block_x(config, 128)?;
    validate_buffer_len("external_consumer", 1024, 1024)?;
    validate_cooperative_launch_config(config)?;

    let dim = Dim3::new(2, 3, 4);
    assert_eq!(dim.as_tuple(), (2, 3, 4));
    assert_eq!(
        AtomicMemoryKind::ManagedCoarseGrain.system_scope_visibility(),
        SystemScopeAtomicVisibility::HostVisibleAfterSynchronization
    );

    let mut values = [1u32, 2, 3, 4];
    let mutable = DeviceSliceMut {
        ptr: values.as_mut_ptr(),
        len: values.len(),
    };
    let immutable: DeviceSlice<u32> = mutable.as_const();
    assert_eq!(immutable.len, values.len());

    Ok(())
}
EOF

cargo check --manifest-path "$WORKDIR/Cargo.toml" --target-dir "$ROOT/target/consumer-smoke"
