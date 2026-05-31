use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hiprtc");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");

    println!("cargo:rerun-if-changed=device-spike/Cargo.toml");
    println!("cargo:rerun-if-changed=device-spike/src");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-device/Cargo.toml");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-device/src");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-kernel/Cargo.toml");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-kernel/src");
    println!("cargo:rerun-if-changed=tools/rocm-oxide-build/Cargo.toml");
    println!("cargo:rerun-if-changed=tools/rocm-oxide-build/src");
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_ARCH");

    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--manifest-path",
            "tools/rocm-oxide-build/Cargo.toml",
            "--",
        ])
        .output()
        .expect("failed to run rocm-oxide-build");

    if !output.status.success() {
        panic!(
            "rocm-oxide-build failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let hsaco = stdout
        .lines()
        .last()
        .expect("rocm-oxide-build did not print a hsaco path");
    let stem = hsaco
        .strip_suffix(".hsaco")
        .expect("rocm-oxide-build output should end in .hsaco");
    let bindings = format!("{stem}.bindings.rs");
    let metadata = format!("{stem}.metadata.json");
    assert!(
        Path::new(&bindings).is_file(),
        "missing generated bindings: {bindings}"
    );
    assert!(
        Path::new(&metadata).is_file(),
        "missing generated metadata: {metadata}"
    );

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let hsaco_out = out_dir.join("rocm_oxide_device_spike.hsaco");
    let bindings_out = out_dir.join("rocm_oxide_device_spike.bindings.rs");
    let metadata_out = out_dir.join("rocm_oxide_device_spike.metadata.json");

    fs::copy(hsaco, &hsaco_out).expect("failed to copy hsaco into OUT_DIR");
    fs::copy(&bindings, &bindings_out).expect("failed to copy bindings into OUT_DIR");
    fs::copy(&metadata, &metadata_out).expect("failed to copy metadata into OUT_DIR");

    println!(
        "cargo:rustc-env=ROCM_OXIDE_DEVICE_HSACO={}",
        hsaco_out.display()
    );
    println!(
        "cargo:rustc-env=ROCM_OXIDE_DEVICE_BINDINGS={}",
        bindings_out.display()
    );
    println!(
        "cargo:rustc-env=ROCM_OXIDE_DEVICE_METADATA={}",
        metadata_out.display()
    );
}
