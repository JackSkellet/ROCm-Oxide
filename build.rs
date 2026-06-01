use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let rocm_path = rocm_path();
    let rocm_lib = rocm_path.join("lib");

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCXX");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rustc-link-search=native={}", rocm_lib.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hiprtc");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rocm_lib.display());

    compile_rocprim_shim(&rocm_path);

    println!("cargo:rerun-if-changed=device-spike/Cargo.toml");
    println!("cargo:rerun-if-changed=device-spike/src");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-device/Cargo.toml");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-device/src");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-kernel/Cargo.toml");
    println!("cargo:rerun-if-changed=crates/rocm-oxide-kernel/src");
    println!("cargo:rerun-if-changed=tools/rocm-oxide-build/Cargo.toml");
    println!("cargo:rerun-if-changed=tools/rocm-oxide-build/src");
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_DEVICE_DEBUG");

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

fn rocm_path() -> PathBuf {
    env::var_os("ROCM_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
}

fn compile_rocprim_shim(rocm_path: &Path) {
    let source = Path::new("src/rocprim_shim.cpp");
    println!("cargo:rerun-if-changed={}", source.display());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let object = out_dir.join("rocm_oxide_rocprim_shim.o");
    let archive = out_dir.join("librocm_oxide_rocprim_shim.a");
    let compiler = cxx_compiler(rocm_path);

    let mut command = Command::new(&compiler);
    command
        .arg("-std=c++17")
        .arg("-O2")
        .arg("-fPIC")
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(&object)
        .arg(format!("-I{}", rocm_path.join("include").display()));
    if compiler
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("hipcc"))
        && let Some(arch) = rocm_arch(rocm_path)
    {
        command.arg(format!("--offload-arch={arch}"));
    }

    let output = command.output().unwrap_or_else(|err| {
        panic!(
            "failed to run rocPRIM shim compiler `{}`: {err}",
            compiler.display()
        )
    });
    if !output.status.success() {
        panic!(
            "failed to compile rocPRIM shim with `{}`\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            compiler.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let ar = env::var_os("AR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ar"));
    let output = Command::new(&ar)
        .args(["crs"])
        .arg(&archive)
        .arg(&object)
        .output()
        .unwrap_or_else(|err| panic!("failed to run `{}` for rocPRIM shim: {err}", ar.display()));
    if !output.status.success() {
        panic!(
            "failed to archive rocPRIM shim with `{}`\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            ar.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=rocm_oxide_rocprim_shim");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn cxx_compiler(rocm_path: &Path) -> PathBuf {
    for var in ["HIPCXX", "CXX"] {
        if let Some(path) = env::var_os(var).filter(|value| !value.is_empty()) {
            return PathBuf::from(path);
        }
    }
    let rocm_hipcc = rocm_path.join("bin").join("hipcc");
    if rocm_hipcc.is_file() {
        return rocm_hipcc;
    }
    let rocm_clang = rocm_path.join("llvm").join("bin").join("clang++");
    if rocm_clang.is_file() {
        return rocm_clang;
    }
    PathBuf::from("hipcc")
}

fn rocm_arch(rocm_path: &Path) -> Option<String> {
    if let Some(arch) = env::var("ROCM_OXIDE_ARCH")
        .ok()
        .filter(|value| !value.is_empty())
    {
        return Some(arch);
    }

    let enumerator = rocm_path.join("bin").join("rocm_agent_enumerator");
    let command = if enumerator.is_file() {
        enumerator
    } else {
        PathBuf::from("rocm_agent_enumerator")
    };
    let output = Command::new(command).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("gfx") && *line != "gfx000")
        .map(ToOwned::to_owned)
}
