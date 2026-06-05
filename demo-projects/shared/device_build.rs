use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("demo project should live under demo-projects/<name>")
        .to_path_buf();

    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("device-spike/Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("device-spike/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root
            .join("crates/rocm-oxide-device/Cargo.toml")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("crates/rocm-oxide-device/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root
            .join("crates/rocm-oxide-kernel/Cargo.toml")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("crates/rocm-oxide-kernel/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("tools/rocm-oxide-build/Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("tools/rocm-oxide-build/src").display()
    );
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_DEVICE_DEBUG");

    let output = Command::new("cargo")
        .current_dir(&repo_root)
        .args([
            "run",
            "--quiet",
            "--manifest-path",
            "tools/rocm-oxide-build/Cargo.toml",
            "--",
            "--crate",
            "device-spike",
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
    let hsaco = PathBuf::from(hsaco);
    let stem = hsaco
        .with_extension("")
        .to_string_lossy()
        .into_owned();
    let bindings = PathBuf::from(format!("{stem}.bindings.rs"));
    let metadata = PathBuf::from(format!("{stem}.metadata.json"));
    let manifest = PathBuf::from(format!("{stem}.manifest.json"));

    for artifact in [&hsaco, &bindings, &metadata, &manifest] {
        assert!(
            artifact.is_file(),
            "missing generated artifact: {}",
            artifact.display()
        );
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let hsaco_out = out_dir.join("rocm_oxide_device_spike.hsaco");
    let bindings_out = out_dir.join("rocm_oxide_device_spike.bindings.rs");
    let metadata_out = out_dir.join("rocm_oxide_device_spike.metadata.json");
    let manifest_out = out_dir.join("rocm_oxide_device_spike.manifest.json");

    fs::copy(&hsaco, &hsaco_out).expect("failed to copy hsaco into OUT_DIR");
    fs::copy(&bindings, &bindings_out).expect("failed to copy bindings into OUT_DIR");
    fs::copy(&metadata, &metadata_out).expect("failed to copy metadata into OUT_DIR");
    fs::copy(&manifest, &manifest_out).expect("failed to copy release manifest into OUT_DIR");

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
    println!(
        "cargo:rustc-env=ROCM_OXIDE_DEVICE_MANIFEST={}",
        manifest_out.display()
    );
}
