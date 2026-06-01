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
        "profile" => profile(&args),
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
    cargo rocm-oxide profile [--trace] [--name NAME] [--pmc COUNTER[,COUNTER...]] [--output-directory DIR] [-- <program> ...]
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
    println!("7. emit metadata, layout-proven device structs, and typed host bindings");
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileMode {
    Compute,
    Trace,
}

fn profile(args: &[OsString]) -> Result<(), String> {
    let root = workspace_root()?;
    let mut profile = ProfileArgs::parse(args, &root)?;
    let mut command_args = std::mem::take(&mut profile.command);
    let profiler =
        find_profiler(profile.mode, &root).ok_or_else(|| profile.missing_profiler_error())?;

    if command_args.is_empty() {
        build_performance_probe(&root)?;
        command_args = vec![
            root.join("target/debug/examples/performance_probe")
                .into_os_string(),
            OsString::from("--json"),
            root.join("target/performance_probe.profiled.json")
                .into_os_string(),
        ];
    }

    fs::create_dir_all(&profile.output_directory).map_err(|err| {
        format!(
            "failed to create profile output directory {}: {err}",
            profile.output_directory.display()
        )
    })?;

    let mut command = Command::new(&profiler.path);
    match (profile.mode, profiler.kind) {
        (ProfileMode::Compute, ProfilerKind::RocprofCompute) => {
            command
                .arg("profile")
                .arg("-n")
                .arg(&profile.name)
                .arg("--output-directory")
                .arg(&profile.output_directory)
                .arg("--");
        }
        (ProfileMode::Compute, ProfilerKind::Rocprofv3) => {
            command.arg("--pmc").args(&profile.pmc_counters);
            command
                .arg("--output-directory")
                .arg(&profile.output_directory)
                .arg("--output-file")
                .arg(&profile.name)
                .arg("--output-format")
                .arg("csv")
                .arg("--");
        }
        (ProfileMode::Trace, ProfilerKind::Rocprofv3) => {
            command
                .arg("--stats")
                .arg("--sys-trace")
                .arg("--output-directory")
                .arg(&profile.output_directory)
                .arg("--output-file")
                .arg(&profile.name)
                .arg("--output-format")
                .arg("csv")
                .arg("--");
        }
        (ProfileMode::Trace, ProfilerKind::RocprofCompute) => {
            return Err("trace mode requires rocprofv3; set ROCM_OXIDE_PROFILER to rocprofv3 or install ROCprofiler-SDK".to_string());
        }
    }
    command.args(&command_args).current_dir(&root);
    println!(
        "profiling {} with {} into {}",
        display_command(&command_args),
        profiler.label(),
        profile.output_directory.display()
    );
    run_status(command, "profile ROCm-Oxide workload")
}

struct ProfileArgs {
    mode: ProfileMode,
    name: OsString,
    pmc_counters: Vec<OsString>,
    output_directory: PathBuf,
    command: Vec<OsString>,
}

impl ProfileArgs {
    fn parse(args: &[OsString], root: &Path) -> Result<Self, String> {
        let mut mode = ProfileMode::Compute;
        let mut name = OsString::from("rocm_oxide_performance_probe");
        let mut pmc_counters = Vec::new();
        let mut has_explicit_pmc = false;
        let mut output_directory = root.join("target/rocm-oxide-profile");
        let mut command = Vec::new();
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if arg == "--" {
                command.extend(args[index + 1..].iter().cloned());
                break;
            }
            match arg.to_str() {
                Some("--compute") => mode = ProfileMode::Compute,
                Some("--trace") => mode = ProfileMode::Trace,
                Some("--name") => {
                    index += 1;
                    name = args
                        .get(index)
                            .cloned()
                            .ok_or_else(|| "--name requires a profile name".to_string())?;
                }
                Some("--pmc") => {
                    index += 1;
                    if !has_explicit_pmc {
                        pmc_counters.clear();
                        has_explicit_pmc = true;
                    }
                    let counters = args
                        .get(index)
                        .and_then(|arg| arg.to_str())
                        .ok_or_else(|| "--pmc requires a comma-separated counter list".to_string())?;
                    let mut parsed = counters
                        .split(',')
                        .map(str::trim)
                        .filter(|counter| !counter.is_empty())
                        .map(OsString::from)
                        .collect::<Vec<_>>();
                    if parsed.is_empty() {
                        return Err("--pmc requires at least one counter".to_string());
                    }
                    pmc_counters.append(&mut parsed);
                }
                Some("--output-directory") => {
                    index += 1;
                    output_directory = PathBuf::from(
                        args.get(index)
                            .cloned()
                            .ok_or_else(|| "--output-directory requires a path".to_string())?,
                    );
                }
                Some("--help") | Some("-h") => {
                    print_profile_help();
                    std::process::exit(0);
                }
                Some(other) => {
                    return Err(format!(
                        "unknown profile option `{other}`; pass workload commands after `--`"
                    ));
                }
                None => return Err("profile arguments must be valid UTF-8 before `--`".to_string()),
            }
            index += 1;
        }
        Ok(Self {
            mode,
            name,
            pmc_counters: if has_explicit_pmc {
                pmc_counters
            } else {
                vec![OsString::from("Wavefronts")]
            },
            output_directory,
            command,
        })
    }

    fn missing_profiler_error(&self) -> String {
        match self.mode {
            ProfileMode::Compute => {
                "could not find `rocprof-compute`, `rocprofiler-compute`, or `rocprofv3`; install ROCm Compute Profiler or ROCprofiler-SDK, set ROCM_OXIDE_PROFILER to its path, or extract ROCm packages under target/rocm-packages/root".to_string()
            }
            ProfileMode::Trace => {
                "could not find `rocprofv3`; install ROCprofiler-SDK, set ROCM_OXIDE_PROFILER to its path, or extract ROCm packages under target/rocm-packages/root".to_string()
            }
        }
    }
}

fn print_profile_help() {
    println!(
        "Usage:
    cargo rocm-oxide profile [--compute|--trace] [--name NAME] [--pmc COUNTER[,COUNTER...]] [--output-directory DIR]
    cargo rocm-oxide profile [--compute|--trace] [options] -- <program> [args]

Without an explicit program, this builds and profiles the performance_probe example.
When ROCm Compute Profiler is unavailable, compute mode falls back to rocprofv3 counter collection.
The --pmc option customizes that rocprofv3 fallback and defaults to Wavefronts."
    );
}

fn build_performance_probe(root: &Path) -> Result<(), String> {
    let mut command = Command::new(cargo());
    command
        .arg("build")
        .arg("--example")
        .arg("performance_probe")
        .current_dir(root);
    run_status(command, "build performance_probe example")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfilerKind {
    RocprofCompute,
    Rocprofv3,
}

struct ProfilerTool {
    path: PathBuf,
    kind: ProfilerKind,
}

impl ProfilerTool {
    fn label(&self) -> &'static str {
        match self.kind {
            ProfilerKind::RocprofCompute => "ROCm Compute Profiler",
            ProfilerKind::Rocprofv3 => "rocprofv3",
        }
    }
}

fn find_profiler(mode: ProfileMode, root: &Path) -> Option<ProfilerTool> {
    if let Some(path) = env::var_os("ROCM_OXIDE_PROFILER")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        return Some(ProfilerTool {
            kind: infer_profiler_kind(mode, &path),
            path,
        });
    }

    match mode {
        ProfileMode::Compute => find_program(root, &["rocprof-compute", "rocprofiler-compute"])
            .map(|path| ProfilerTool {
                path,
                kind: ProfilerKind::RocprofCompute,
            })
            .or_else(|| {
                find_program(root, &["rocprofv3"]).map(|path| ProfilerTool {
                    path,
                    kind: ProfilerKind::Rocprofv3,
                })
            }),
        ProfileMode::Trace => find_program(root, &["rocprofv3"]).map(|path| ProfilerTool {
            path,
            kind: ProfilerKind::Rocprofv3,
        }),
    }
}

fn infer_profiler_kind(mode: ProfileMode, path: &Path) -> ProfilerKind {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("rocprofv3"))
    {
        return ProfilerKind::Rocprofv3;
    }
    match mode {
        ProfileMode::Compute => ProfilerKind::RocprofCompute,
        ProfileMode::Trace => ProfilerKind::Rocprofv3,
    }
}

fn find_program(root: &Path, candidates: &[&str]) -> Option<PathBuf> {
    let mut search_dirs = Vec::new();
    if let Some(path) = env::var_os("PATH") {
        search_dirs.extend(env::split_paths(&path));
    }
    search_dirs.push(root.join("target/rocm-packages/root/opt/rocm/bin"));
    search_dirs.push(PathBuf::from("/opt/rocm/bin"));

    for candidate in candidates {
        let candidate_path = Path::new(candidate);
        if candidate_path.components().count() > 1 && is_executable(candidate_path) {
            return Some(candidate_path.to_path_buf());
        }
        for dir in &search_dirs {
            let path = dir.join(candidate);
            if is_executable(&path) {
                return Some(path);
            }
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn display_command(args: &[OsString]) -> String {
    args.iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ")
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
