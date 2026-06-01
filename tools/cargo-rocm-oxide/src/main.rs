use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "rocm-oxide") {
        args.remove(0);
    }

    let Some(command) = args.first().and_then(|arg| arg.to_str()).map(str::to_owned) else {
        print_help();
        return Ok(());
    };
    args.remove(0);

    match command.as_str() {
        "doctor" => run_build_tool(["--doctor"], &[]),
        "build" => run_build_tool(std::iter::empty::<&str>(), &args),
        "run" => cargo_run(&args),
        "inspect" => inspect(&args),
        "pipeline" => pipeline(&args),
        "new" => new_project(&args),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown cargo rocm-oxide command `{other}`")),
    }
}

fn print_help() {
    println!(
        "Usage:
    cargo rocm-oxide doctor
    cargo rocm-oxide build [-- --arch <gfx arch>]
    cargo rocm-oxide run [cargo-run-args]
    cargo rocm-oxide inspect [metadata.json]
    cargo rocm-oxide pipeline [--build]
    cargo rocm-oxide new <path>"
    );
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn workspace_root() -> Result<PathBuf, String> {
    let mut current =
        env::current_dir().map_err(|err| format!("failed to read current directory: {err}"))?;
    loop {
        if current.join("tools/rocm-oxide-build/Cargo.toml").is_file() {
            return Ok(current);
        }
        if !current.pop() {
            return Err("could not find a rocm-oxide workspace root".to_string());
        }
    }
}

fn run_build_tool<I>(fixed_args: I, passthrough: &[OsString]) -> Result<(), String>
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let root = workspace_root()?;
    let mut command = Command::new(cargo());
    command
        .arg("run")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(root.join("tools/rocm-oxide-build/Cargo.toml"))
        .arg("--")
        .current_dir(&root);
    for arg in fixed_args {
        command.arg(arg.as_ref());
    }
    for arg in passthrough.iter().filter(|arg| arg.as_os_str() != "--") {
        command.arg(arg);
    }
    run_status(command, "run rocm-oxide-build")
}

fn cargo_run(args: &[OsString]) -> Result<(), String> {
    let root = workspace_root()?;
    let mut command = Command::new(cargo());
    command.arg("run").current_dir(&root);
    for arg in args {
        command.arg(arg);
    }
    run_status(command, "run host crate")
}

fn pipeline(args: &[OsString]) -> Result<(), String> {
    let root = workspace_root()?;
    if args.iter().any(|arg| arg == "--build") {
        run_build_tool(std::iter::empty::<&str>(), &[])?;
    }

    println!("ROCm-Oxide pipeline");
    println!(
        "1. discover #[kernel] functions in device-spike/src and kernel-bearing path dependencies"
    );
    println!("2. cargo rustc -Z build-std=core --target amdgcn-amd-amdhsa");
    println!("3. rewrite marked Rust functions into AMDGPU/HSA kernels");
    println!("4. lower LLVM IR with ROCm llc");
    println!("5. link HSACO with ROCm clang");
    println!("6. validate kernel symbols and .kd descriptors with llvm-readelf");
    println!("7. emit metadata, mirrored repr(C) device structs, and typed host bindings");
    println!("8. root build.rs copies artifacts into OUT_DIR for host embedding");

    let metadata = find_latest_metadata(&root);
    if let Some(metadata) = metadata {
        println!();
        run_build_tool(["--inspect-metadata"], &[metadata.into_os_string()])?;
    }
    Ok(())
}

fn inspect(args: &[OsString]) -> Result<(), String> {
    let root = workspace_root()?;
    let metadata = if let Some(path) = args.first() {
        PathBuf::from(path)
    } else {
        find_latest_metadata(&root)
            .ok_or_else(|| "no generated metadata found; run `cargo rocm-oxide build` first".to_string())?
    };
    run_build_tool(["--inspect-metadata"], &[metadata.into_os_string()])
}

fn find_latest_metadata(root: &Path) -> Option<PathBuf> {
    let path = root.join(
        "device-spike/target/amdgcn-amd-amdhsa/release/rocm_oxide_device_spike.metadata.json",
    );
    path.is_file().then_some(path)
}

fn new_project(args: &[OsString]) -> Result<(), String> {
    let Some(path) = args.first() else {
        return Err("cargo rocm-oxide new requires a path".to_string());
    };
    let path = PathBuf::from(path);
    if path.exists() {
        return Err(format!("target already exists: {}", path.display()));
    }

    fs::create_dir_all(path.join("src"))
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    fs::write(
        path.join("Cargo.toml"),
        r#"[package]
name = "rocm-oxide-app"
version = "0.1.0"
edition = "2024"

[dependencies]
rocm-oxide = { path = ".." }
"#,
    )
    .map_err(|err| format!("failed to write Cargo.toml: {err}"))?;
    fs::write(
        path.join("src/main.rs"),
        r#"fn main() {
    println!("ROCm-Oxide project scaffold created. Point this dependency at a packaged rocm-oxide runtime before publishing.");
}
"#,
    )
    .map_err(|err| format!("failed to write src/main.rs: {err}"))?;

    println!("created {}", path.display());
    Ok(())
}

fn run_status(mut command: Command, label: &str) -> Result<(), String> {
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|err| format!("failed to {label}: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{label} failed with status {status}"))
    }
}
