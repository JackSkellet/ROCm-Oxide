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

    if !device_spike_feature_enabled() {
        return;
    }

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
    let manifest = format!("{stem}.manifest.json");
    let hsaco = Path::new(hsaco);
    let bindings = Path::new(&bindings);
    let metadata = Path::new(&metadata);
    let manifest = Path::new(&manifest);
    validate_generated_artifacts(hsaco, bindings, metadata, manifest);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let hsaco_out = out_dir.join("rocm_oxide_device_spike.hsaco");
    let bindings_out = out_dir.join("rocm_oxide_device_spike.bindings.rs");
    let metadata_out = out_dir.join("rocm_oxide_device_spike.metadata.json");
    let manifest_out = out_dir.join("rocm_oxide_device_spike.manifest.json");

    fs::copy(hsaco, &hsaco_out).expect("failed to copy hsaco into OUT_DIR");
    fs::copy(bindings, &bindings_out).expect("failed to copy bindings into OUT_DIR");
    fs::copy(metadata, &metadata_out).expect("failed to copy metadata into OUT_DIR");
    fs::copy(manifest, &manifest_out).expect("failed to copy release manifest into OUT_DIR");

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

fn device_spike_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_DEVICE_SPIKE").is_some()
}

fn validate_generated_artifacts(hsaco: &Path, bindings: &Path, metadata: &Path, manifest: &Path) {
    assert!(
        hsaco.is_file(),
        "missing generated hsaco: {}",
        hsaco.display()
    );
    assert!(
        bindings.is_file(),
        "missing generated bindings: {}",
        bindings.display()
    );
    assert!(
        metadata.is_file(),
        "missing generated metadata: {}",
        metadata.display()
    );
    assert!(
        manifest.is_file(),
        "missing generated release manifest: {}",
        manifest.display()
    );

    let metadata_text = fs::read_to_string(metadata).unwrap_or_else(|err| {
        panic!(
            "failed to read generated metadata {}: {err}",
            metadata.display()
        )
    });
    let target = find_json_string(&metadata_text, "target").unwrap_or_else(|| {
        panic!(
            "generated metadata {} is missing `target`",
            metadata.display()
        )
    });
    assert_eq!(
        target,
        "amdgcn-amd-amdhsa",
        "generated metadata {} targets `{target}`, expected amdgcn-amd-amdhsa",
        metadata.display()
    );

    let arch = find_json_string(&metadata_text, "arch").unwrap_or_else(|| {
        panic!(
            "generated metadata {} is missing `arch`",
            metadata.display()
        )
    });
    assert!(
        arch.starts_with("gfx"),
        "generated metadata {} has invalid ROCm arch `{arch}`",
        metadata.display()
    );
    if let Some(expected_arch) = env::var("ROCM_OXIDE_ARCH")
        .ok()
        .filter(|value| !value.is_empty())
    {
        assert_eq!(
            arch,
            expected_arch,
            "generated metadata {} arch `{arch}` does not match ROCM_OXIDE_ARCH `{expected_arch}`",
            metadata.display()
        );
    }

    let metadata_hsaco = find_json_string(&metadata_text, "hsaco").unwrap_or_else(|| {
        panic!(
            "generated metadata {} is missing `hsaco`",
            metadata.display()
        )
    });
    let metadata_hsaco = PathBuf::from(metadata_hsaco);
    let expected_hsaco = fs::canonicalize(hsaco).unwrap_or_else(|err| {
        panic!(
            "failed to canonicalize generated hsaco {}: {err}",
            hsaco.display()
        )
    });
    let metadata_hsaco = fs::canonicalize(&metadata_hsaco).unwrap_or_else(|err| {
        panic!(
            "generated metadata {} points at missing hsaco {}: {err}",
            metadata.display(),
            metadata_hsaco.display()
        )
    });
    assert_eq!(
        metadata_hsaco,
        expected_hsaco,
        "generated metadata {} points at {}, but rocm-oxide-build returned {}",
        metadata.display(),
        metadata_hsaco.display(),
        expected_hsaco.display()
    );

    assert!(
        metadata_text.contains("\"link\"") && metadata_text.contains("\"objects\""),
        "generated metadata {} is missing link-object provenance",
        metadata.display()
    );
    assert!(
        metadata_text.contains("\"kernels\"") && metadata_text.contains("\"name\""),
        "generated metadata {} does not list generated kernels",
        metadata.display()
    );

    let manifest_text = fs::read_to_string(manifest).unwrap_or_else(|err| {
        panic!(
            "failed to read generated release manifest {}: {err}",
            manifest.display()
        )
    });
    let manifest_format = find_json_string(&manifest_text, "format").unwrap_or_else(|| {
        panic!(
            "generated release manifest {} is missing `format`",
            manifest.display()
        )
    });
    assert_eq!(
        manifest_format,
        "rocm-oxide-release-manifest-v1",
        "generated release manifest {} has unexpected format `{manifest_format}`",
        manifest.display()
    );
    let manifest_target = find_json_string(&manifest_text, "target").unwrap_or_else(|| {
        panic!(
            "generated release manifest {} is missing `target`",
            manifest.display()
        )
    });
    assert_eq!(
        manifest_target,
        target,
        "generated release manifest {} target `{manifest_target}` does not match metadata target `{target}`",
        manifest.display()
    );
    let manifest_arch = find_json_string(&manifest_text, "arch").unwrap_or_else(|| {
        panic!(
            "generated release manifest {} is missing `arch`",
            manifest.display()
        )
    });
    assert_eq!(
        manifest_arch,
        arch,
        "generated release manifest {} arch `{manifest_arch}` does not match metadata arch `{arch}`",
        manifest.display()
    );
    for artifact in [hsaco, bindings, metadata] {
        let artifact_path = artifact.display().to_string();
        assert!(
            manifest_text.contains(&artifact_path),
            "generated release manifest {} does not include artifact {}",
            manifest.display(),
            artifact.display()
        );
    }
    assert_same_stem(hsaco, bindings, "bindings");
    assert_same_stem(hsaco, metadata, "metadata");
    assert_same_stem(hsaco, manifest, "release manifest");
}

fn assert_same_stem(hsaco: &Path, artifact: &Path, label: &str) {
    let hsaco_stem = hsaco
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("generated hsaco filename must be valid UTF-8");
    let artifact_name = artifact
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("generated {label} filename must be valid UTF-8"));
    assert!(
        artifact_name.starts_with(hsaco_stem),
        "generated {label} {} does not match hsaco stem `{hsaco_stem}`",
        artifact.display()
    );
}

fn find_json_string(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let index = text.find(&needle)?;
    let rest = text[index + needle.len()..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let mut value = String::new();
    let mut escaped = false;
    for ch in rest.chars() {
        if escaped {
            value.push(match ch {
                '"' => '"',
                '\\' => '\\',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            });
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(value);
        } else {
            value.push(ch);
        }
    }
    None
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
