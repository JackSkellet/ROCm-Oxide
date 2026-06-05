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
        "doctor" => run_build_tool(["--doctor"], &args),
        "build" => run_build_tool(std::iter::empty::<&str>(), &args),
        "run" => cargo_run(&args),
        "debug" => cargo_debug(&args),
        "inspect" => inspect(&args),
        "pipeline" => pipeline(&args),
        "profile" => profile(&args),
        "verify" => verify(&args),
        "new" => new_project(&args),
        "check-consumer" => check_consumer(),
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
    cargo rocm-oxide doctor [--json|--github]
    cargo rocm-oxide build [-- --arch <gfx arch>]
    cargo rocm-oxide run [cargo-run-args]
    cargo rocm-oxide debug [cargo-run-args]
    cargo rocm-oxide inspect [metadata.json]
    cargo rocm-oxide pipeline [--build] [--crate PATH] [--output-stem NAME]
    cargo rocm-oxide profile [--trace] [--name NAME] [--pmc COUNTER[,COUNTER...]] [--output-directory DIR] [-- <program> ...]
    cargo rocm-oxide verify [--host-ci|--offline|--quick|--full]
    cargo rocm-oxide new <path>
    cargo rocm-oxide new <path> --local ROCM_OXIDE_WORKSPACE
    cargo rocm-oxide new <path> --path ROCM_OXIDE_WORKSPACE
    cargo rocm-oxide new <path> --git REPOSITORY [--branch NAME|--tag NAME|--rev SHA]
    cargo rocm-oxide new <path> --standalone
    cargo rocm-oxide check-consumer

Notes:
    new              Creates a LOCAL SCAFFOLD tied to a ROCm-Oxide source workspace
                     via relative paths. By default, run it from within the workspace.
                     Use --local ROCM_OXIDE_WORKSPACE, or alias --path, when
                     generating from another directory. Use --git for a scaffold
                     whose Rust dependencies come from a git repository; it still
                     requires rocm-oxide-build through ROCM_OXIDE_BUILD or PATH.
                     --standalone is reserved
                     until crates.io/release artifacts exist.
    check-consumer   Validates a generated scaffold project. Run from inside the
                     consumer project directory. Checks path dependencies, build.rs,
                     and rust-toolchain.toml.
    verify           Source-workspace gate only. Run from the ROCm-Oxide repo root,
                     not from generated projects. Use `cargo build` in generated
                     projects to verify the build instead."
    );
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn project_root() -> Result<PathBuf, String> {
    let mut current =
        env::current_dir().map_err(|err| format!("failed to read current directory: {err}"))?;
    loop {
        if current.join("Cargo.toml").is_file() {
            return Ok(current);
        }
        if !current.pop() {
            return Err("could not find a Cargo.toml project root".to_string());
        }
    }
}

fn source_workspace_root() -> Result<PathBuf, String> {
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
    let root = project_root()?;
    run_build_tool_in(&root, fixed_args, passthrough)
}

fn run_build_tool_in<I>(root: &Path, fixed_args: I, passthrough: &[OsString]) -> Result<(), String>
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut command = build_tool_command(root)?;
    for arg in fixed_args {
        command.arg(arg.as_ref());
    }
    for arg in passthrough.iter().filter(|arg| arg.as_os_str() != "--") {
        command.arg(arg);
    }
    run_status(command, "run rocm-oxide-build")
}

fn build_tool_command(root: &Path) -> Result<Command, String> {
    if let Some(path) = env::var_os("ROCM_OXIDE_BUILD").filter(|value| !value.is_empty()) {
        let mut command = Command::new(path);
        command.current_dir(root);
        return Ok(command);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join("rocm-oxide-build");
        if is_executable(&sibling) {
            let mut command = Command::new(sibling);
            command.current_dir(root);
            return Ok(command);
        }
    }

    if let Some(path) = find_program_on_path("rocm-oxide-build") {
        let mut command = Command::new(path);
        command.current_dir(root);
        return Ok(command);
    }

    let source_root = source_workspace_root()?;
    let mut command = Command::new(cargo());
    command
        .arg("run")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(source_root.join("tools/rocm-oxide-build/Cargo.toml"))
        .arg("--")
        .current_dir(root);
    Ok(command)
}

fn find_program_on_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|path| is_executable(path))
}

fn cargo_run(args: &[OsString]) -> Result<(), String> {
    let root = project_root()?;
    let mut command = Command::new(cargo());
    command.arg("run").current_dir(&root);
    for arg in args {
        command.arg(arg);
    }
    run_status(command, "run host crate")
}

fn cargo_debug(args: &[OsString]) -> Result<(), String> {
    let root = project_root()?;
    let mut command = Command::new(cargo());
    command
        .arg("run")
        .args(args)
        .env("ROCM_OXIDE_DEVICE_DEBUG", "1")
        .current_dir(&root);
    run_status(command, "run host crate with ROCm-Oxide device debug")
}

fn pipeline(args: &[OsString]) -> Result<(), String> {
    let root = project_root()?;
    let args = PipelineArgs::parse(args)?;
    if args.build {
        run_build_tool_in(&root, std::iter::empty::<&str>(), &args.build_tool_args())?;
    }

    println!("ROCm-Oxide pipeline");
    println!(
        "1. discover #[kernel] functions in {} and kernel-bearing path dependencies",
        args.device_crate.display()
    );
    println!("2. cargo rustc -Z build-std=core --target amdgcn-amd-amdhsa");
    println!("3. rewrite marked Rust functions into AMDGPU/HSA kernels");
    println!("4. lower LLVM IR with ROCm llc");
    println!("5. link HSACO with ROCm clang");
    println!("6. validate kernel symbols and .kd descriptors with llvm-readelf");
    println!("7. emit metadata, layout-proven device structs, and typed host bindings");
    println!("8. root build.rs copies artifacts into OUT_DIR for host embedding");

    let metadata = find_latest_metadata(&root, &args.device_crate, &args.output_stem);
    if let Some(metadata) = metadata {
        println!();
        run_build_tool_in(&root, ["--inspect-metadata"], &[metadata.into_os_string()])?;
    }
    Ok(())
}

fn inspect(args: &[OsString]) -> Result<(), String> {
    let root = project_root()?;
    let metadata = if let Some(path) = args.first() {
        PathBuf::from(path)
    } else {
        find_latest_metadata(
            &root,
            Path::new("device-spike"),
            std::ffi::OsStr::new("rocm_oxide_device_spike"),
        )
        .ok_or_else(|| {
            "no generated metadata found; run `cargo rocm-oxide build` first".to_string()
        })?
    };
    run_build_tool_in(&root, ["--inspect-metadata"], &[metadata.into_os_string()])
}

struct PipelineArgs {
    build: bool,
    device_crate: PathBuf,
    output_stem: OsString,
}

impl PipelineArgs {
    fn parse(args: &[OsString]) -> Result<Self, String> {
        let mut build = false;
        let mut device_crate = PathBuf::from("device-spike");
        let mut output_stem = OsString::from("rocm_oxide_device_spike");
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            match arg.to_str() {
                Some("--build") => build = true,
                Some("--crate") => {
                    index += 1;
                    device_crate = PathBuf::from(
                        args.get(index)
                            .cloned()
                            .ok_or_else(|| "--crate requires a path".to_string())?,
                    );
                }
                Some("--output-stem") => {
                    index += 1;
                    output_stem = args
                        .get(index)
                        .cloned()
                        .ok_or_else(|| "--output-stem requires a filename stem".to_string())?;
                }
                Some("--help") | Some("-h") => {
                    print_help();
                    std::process::exit(0);
                }
                Some(other) => return Err(format!("unknown pipeline option `{other}`")),
                None => return Err("pipeline arguments must be valid UTF-8".to_string()),
            }
            index += 1;
        }
        Ok(Self {
            build,
            device_crate,
            output_stem,
        })
    }

    fn build_tool_args(&self) -> Vec<OsString> {
        vec![
            OsString::from("--crate"),
            self.device_crate.clone().into_os_string(),
            OsString::from("--output-stem"),
            self.output_stem.clone(),
        ]
    }
}

fn verify(args: &[OsString]) -> Result<(), String> {
    let root = source_workspace_root().map_err(|_| {
        "`cargo rocm-oxide verify` only runs from within the ROCm-Oxide source workspace.\n\
         It is a repository-level gate, not a per-project command.\n\
         hint: cd into your cloned ROCm-Oxide directory and run `cargo rocm-oxide verify --quick`.\n\
         hint: to check that your generated project builds, use `cargo build` inside it instead."
            .to_string()
    })?;
    let mut command = Command::new(root.join("scripts/verify.sh"));
    command.args(args).current_dir(&root);
    run_status(command, "run ROCm-Oxide verification gate")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileMode {
    Compute,
    Trace,
}

fn profile(args: &[OsString]) -> Result<(), String> {
    let root = source_workspace_root()?;
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
                    let counters =
                        args.get(index)
                            .and_then(|arg| arg.to_str())
                            .ok_or_else(|| {
                                "--pmc requires a comma-separated counter list".to_string()
                            })?;
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

fn find_latest_metadata(
    root: &Path,
    device_crate: &Path,
    output_stem: &std::ffi::OsStr,
) -> Option<PathBuf> {
    let mut file_name = output_stem.to_os_string();
    file_name.push(".metadata.json");
    let path = root
        .join(device_crate)
        .join("target/amdgcn-amd-amdhsa/release")
        .join(file_name);
    path.is_file().then_some(path)
}

fn check_consumer() -> Result<(), String> {
    let root = project_root()?;
    let manifest = root.join("Cargo.toml");
    let device_manifest = root.join("device-spike/Cargo.toml");
    let build_rs = root.join("build.rs");
    let toolchain = root.join("rust-toolchain.toml");

    let mut all_pass = true;

    // Check root Cargo.toml path dependencies resolve
    let manifest_content =
        fs::read_to_string(&manifest).map_err(|err| format!("failed to read Cargo.toml: {err}"))?;
    let root_path_deps = extract_path_deps(&manifest_content, &root);
    if root_path_deps.is_empty() && contains_git_dependency(&manifest_content, "rocm-oxide") {
        println!("[pass] rocm-oxide git dependency configured");
        if env::var_os("ROCM_OXIDE_BUILD").is_some()
            || find_program_on_path("rocm-oxide-build").is_some()
        {
            println!("[pass] rocm-oxide-build available through ROCM_OXIDE_BUILD or PATH");
        } else {
            println!(
                "[warn] git scaffold needs rocm-oxide-build through ROCM_OXIDE_BUILD or PATH before cargo build"
            );
        }
    }
    for (dep, resolved) in root_path_deps {
        let dep_manifest = resolved.join("Cargo.toml");
        if dep_manifest.is_file() {
            println!("[pass] {dep} path dependency resolves");
        } else {
            println!(
                "[fail] {dep} path dependency does not resolve — {}",
                dep_manifest.display()
            );
            all_pass = false;
        }
    }

    // Check device-spike/Cargo.toml if present
    if device_manifest.is_file() {
        let device_content = fs::read_to_string(&device_manifest)
            .map_err(|err| format!("failed to read device-spike/Cargo.toml: {err}"))?;
        let device_path_deps = extract_path_deps(&device_content, &root.join("device-spike"));
        if device_path_deps.is_empty()
            && contains_git_dependency(&device_content, "rocm-oxide-device")
            && contains_git_dependency(&device_content, "rocm-oxide-kernel")
        {
            println!("[pass] device-spike git dependencies configured");
        }
        for (dep, resolved) in device_path_deps {
            let dep_manifest = resolved.join("Cargo.toml");
            if dep_manifest.is_file() {
                println!("[pass] device-spike: {dep} path dependency resolves");
            } else {
                println!(
                    "[fail] device-spike: {dep} path dependency does not resolve — {}",
                    dep_manifest.display()
                );
                all_pass = false;
            }
        }
    } else {
        println!(
            "[warn] device-spike/Cargo.toml not found — skipping device crate dependency checks"
        );
    }

    // Check build.rs exists and emits ROCM_OXIDE_DEVICE_BINDINGS
    if build_rs.is_file() {
        let build_content = fs::read_to_string(&build_rs)
            .map_err(|err| format!("failed to read build.rs: {err}"))?;
        if build_content.contains("ROCM_OXIDE_DEVICE_BINDINGS") {
            println!("[pass] build.rs present and sets ROCM_OXIDE_DEVICE_BINDINGS");
        } else {
            println!(
                "[warn] build.rs present but does not appear to set ROCM_OXIDE_DEVICE_BINDINGS"
            );
        }
    } else {
        println!("[fail] build.rs not found — run `cargo rocm-oxide new` to regenerate scaffold");
        all_pass = false;
    }

    // Check rust-toolchain.toml exists and requests rust-src
    if toolchain.is_file() {
        let toolchain_content = fs::read_to_string(&toolchain)
            .map_err(|err| format!("failed to read rust-toolchain.toml: {err}"))?;
        if toolchain_content.contains("rust-src") {
            println!("[pass] rust-toolchain.toml present and requests rust-src");
        } else {
            println!("[warn] rust-toolchain.toml present but does not list rust-src component");
        }
    } else {
        println!(
            "[fail] rust-toolchain.toml not found — without it cargo may use stable Rust and fail -Z build-std=core"
        );
        all_pass = false;
    }

    if all_pass {
        println!();
        println!("all checks passed");
    } else {
        println!();
        println!(
            "one or more checks failed — re-run `cargo rocm-oxide new <path>` to regenerate the scaffold"
        );
    }
    Ok(())
}

fn contains_git_dependency(toml: &str, dep_name: &str) -> bool {
    let inline = format!("{dep_name} = {{");
    let section = format!("[dependencies.{dep_name}]");
    let mut in_section = false;

    for line in toml.lines() {
        let trimmed = line.trim();
        if trimmed == section {
            in_section = true;
            continue;
        }
        if trimmed.starts_with('[') {
            in_section = false;
        }
        if trimmed.starts_with(&inline) && trimmed.contains("git =") {
            return true;
        }
        if in_section && trimmed.starts_with("git =") {
            return true;
        }
    }

    false
}

/// Extract `path = "..."` values from a Cargo.toml string and resolve them
/// relative to `base`. Returns (dependency-name, resolved-path) pairs.
fn extract_path_deps(toml: &str, base: &Path) -> Vec<(String, PathBuf)> {
    let mut deps = Vec::new();
    let mut current_dep: Option<String> = None;
    for line in toml.lines() {
        let trimmed = line.trim();
        // Detect [dependencies.foo] or foo = { ... } style headers
        if let Some(rest) = trimmed.strip_prefix('[') {
            if let Some(name) = rest
                .strip_prefix("dependencies.")
                .and_then(|s| s.strip_suffix(']'))
            {
                current_dep = Some(name.trim().to_owned());
            } else {
                current_dep = None;
            }
        } else if let Some((lhs, rhs)) = trimmed.split_once('=') {
            let dep_name = lhs.trim().to_owned();
            let rhs = rhs.trim();
            // Inline: foo = { path = "..." }
            if let Some(path_val) = extract_path_value(rhs) {
                deps.push((dep_name.clone(), normalize_path(&base.join(path_val))));
            }
            // path = "..." inside a [dependencies.foo] section
            if dep_name == "path" {
                if let Some(name) = &current_dep {
                    if let Some(path_val) = extract_path_value(rhs) {
                        deps.push((name.clone(), normalize_path(&base.join(path_val))));
                    }
                }
            }
        }
    }
    deps
}

fn extract_path_value(s: &str) -> Option<&str> {
    // Match: "some/path" or { path = "some/path", ... }
    let search = if s.contains("path") {
        s
    } else {
        return None;
    };
    // Find path = "..." anywhere in the value
    let after_path = search.split("path").nth(1)?;
    let after_eq = after_path.split('=').nth(1)?.trim();
    let inner = after_eq.trim_start_matches('{').trim();
    let start = inner.find('"')? + 1;
    let end = inner[start..].find('"')?;
    Some(&inner[start..start + end])
}

struct NewProjectArgs {
    path: PathBuf,
    source_workspace: Option<PathBuf>,
    git: Option<GitDependency>,
    standalone: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GitDependency {
    url: String,
    reference: Option<GitReference>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GitReference {
    Branch(String),
    Tag(String),
    Rev(String),
}

impl GitDependency {
    fn dependency_value(&self) -> String {
        let mut value = format!("{{ git = {:?}", self.url);
        match &self.reference {
            Some(GitReference::Branch(branch)) => {
                value.push_str(&format!(", branch = {branch:?}"));
            }
            Some(GitReference::Tag(tag)) => {
                value.push_str(&format!(", tag = {tag:?}"));
            }
            Some(GitReference::Rev(rev)) => {
                value.push_str(&format!(", rev = {rev:?}"));
            }
            None => {}
        }
        value.push_str(" }");
        value
    }

    fn install_hint(&self) -> String {
        let mut command = format!("cargo install --git {:?} rocm-oxide-build", self.url);
        match &self.reference {
            Some(GitReference::Branch(branch)) => {
                command.push_str(&format!(" --branch {branch:?}"));
            }
            Some(GitReference::Tag(tag)) => {
                command.push_str(&format!(" --tag {tag:?}"));
            }
            Some(GitReference::Rev(rev)) => {
                command.push_str(&format!(" --rev {rev:?}"));
            }
            None => {}
        }
        command
    }
}

impl NewProjectArgs {
    fn parse(args: &[OsString]) -> Result<Self, String> {
        let mut path = None;
        let mut source_workspace = None;
        let mut git = None;
        let mut git_reference = None;
        let mut standalone = false;
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            match arg.to_str() {
                Some("--help") | Some("-h") => {
                    print_new_project_help();
                    std::process::exit(0);
                }
                Some("--local") | Some("--path") => {
                    if standalone {
                        return Err(
                            "`--local`/`--path` cannot be combined with `--standalone`".to_string()
                        );
                    }
                    if git.is_some() {
                        return Err(
                            "`--local`/`--path` cannot be combined with `--git`".to_string()
                        );
                    }
                    if source_workspace.is_some() {
                        return Err("only one --local/--path workspace may be provided".to_string());
                    }
                    index += 1;
                    let Some(value) = args.get(index) else {
                        return Err("--local requires a ROCm-Oxide workspace path".to_string());
                    };
                    source_workspace = Some(PathBuf::from(value));
                }
                Some(value) if value.starts_with("--local=") || value.starts_with("--path=") => {
                    if standalone {
                        return Err(
                            "`--local`/`--path` cannot be combined with `--standalone`".to_string()
                        );
                    }
                    if git.is_some() {
                        return Err(
                            "`--local`/`--path` cannot be combined with `--git`".to_string()
                        );
                    }
                    if source_workspace.is_some() {
                        return Err("only one --local/--path workspace may be provided".to_string());
                    }
                    let value = value
                        .trim_start_matches("--local=")
                        .trim_start_matches("--path=");
                    if value.is_empty() {
                        return Err("--local requires a ROCm-Oxide workspace path".to_string());
                    }
                    source_workspace = Some(PathBuf::from(value));
                }
                Some("--git") => {
                    if standalone {
                        return Err("`--git` cannot be combined with `--standalone`".to_string());
                    }
                    if source_workspace.is_some() {
                        return Err(
                            "`--git` cannot be combined with `--local`/`--path`".to_string()
                        );
                    }
                    if git.is_some() {
                        return Err("only one --git repository may be provided".to_string());
                    }
                    index += 1;
                    let Some(value) = args.get(index).and_then(|value| value.to_str()) else {
                        return Err("--git requires a repository URL".to_string());
                    };
                    if value.is_empty() {
                        return Err("--git requires a repository URL".to_string());
                    }
                    git = Some(value.to_string());
                }
                Some(value) if value.starts_with("--git=") => {
                    if standalone {
                        return Err("`--git` cannot be combined with `--standalone`".to_string());
                    }
                    if source_workspace.is_some() {
                        return Err(
                            "`--git` cannot be combined with `--local`/`--path`".to_string()
                        );
                    }
                    if git.is_some() {
                        return Err("only one --git repository may be provided".to_string());
                    }
                    let value = value.trim_start_matches("--git=");
                    if value.is_empty() {
                        return Err("--git requires a repository URL".to_string());
                    }
                    git = Some(value.to_string());
                }
                Some("--branch") | Some("--tag") | Some("--rev") => {
                    let kind = arg.to_str().unwrap_or_default();
                    index += 1;
                    let Some(value) = args.get(index).and_then(|value| value.to_str()) else {
                        return Err(format!("{kind} requires a value"));
                    };
                    set_git_reference(&mut git_reference, kind, value)?;
                }
                Some(value)
                    if value.starts_with("--branch=")
                        || value.starts_with("--tag=")
                        || value.starts_with("--rev=") =>
                {
                    let (kind, value) = value
                        .split_once('=')
                        .expect("pattern above ensures an equals sign");
                    set_git_reference(&mut git_reference, kind, value)?;
                }
                Some("--standalone") => {
                    if source_workspace.is_some() {
                        return Err("`--standalone` cannot be combined with `--local`/`--path`; standalone mode will use published packages once supported".to_string());
                    }
                    if git.is_some() {
                        return Err("`--standalone` cannot be combined with `--git`".to_string());
                    }
                    standalone = true;
                }
                Some(other) if other.starts_with('-') => {
                    return Err(format!("unknown new option `{other}`"));
                }
                Some(_) => {
                    if path.is_some() {
                        return Err(
                            "cargo rocm-oxide new accepts exactly one project path".to_string()
                        );
                    }
                    path = Some(PathBuf::from(arg));
                }
                None => return Err("new arguments must be valid UTF-8".to_string()),
            }
            index += 1;
        }
        let Some(path) = path else {
            return Err("cargo rocm-oxide new requires a path".to_string());
        };
        if git_reference.is_some() && git.is_none() {
            return Err("`--branch`, `--tag`, and `--rev` require `--git`".to_string());
        }
        Ok(Self {
            path,
            source_workspace,
            git: git.map(|url| GitDependency {
                url,
                reference: git_reference,
            }),
            standalone,
        })
    }
}

fn set_git_reference(
    git_reference: &mut Option<GitReference>,
    kind: &str,
    value: &str,
) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{kind} requires a non-empty value"));
    }
    if git_reference.is_some() {
        return Err("only one of --branch, --tag, or --rev may be provided".to_string());
    }
    *git_reference = Some(match kind {
        "--branch" => GitReference::Branch(value.to_string()),
        "--tag" => GitReference::Tag(value.to_string()),
        "--rev" => GitReference::Rev(value.to_string()),
        _ => return Err(format!("unknown git reference option `{kind}`")),
    });
    Ok(())
}

fn print_new_project_help() {
    println!(
        "Usage:
    cargo rocm-oxide new <path>
    cargo rocm-oxide new <path> --local ROCM_OXIDE_WORKSPACE
    cargo rocm-oxide new <path> --path ROCM_OXIDE_WORKSPACE
    cargo rocm-oxide new <path> --git REPOSITORY [--branch NAME|--tag NAME|--rev SHA]
    cargo rocm-oxide new <path> --standalone

Options:
    --local PATH      Use PATH as the ROCm-Oxide source workspace when generating
                      from outside the workspace.
    --path PATH       Alias for --local.
    --git URL         Use git dependencies for rocm-oxide, rocm-oxide-device,
                      and rocm-oxide-kernel. The build tool must be installed
                      through ROCM_OXIDE_BUILD or PATH.
    --branch NAME     Git branch to use with --git.
    --tag NAME        Git tag to use with --git.
    --rev SHA         Git revision to use with --git.
    --standalone      Reserved for future crates.io/release-artifact scaffolds.

Default mode uses relative path dependencies. --git removes the local dependency
paths, but is not fully standalone until rocm-oxide-build is distributed as a
crates.io package or release artifact."
    );
}

fn explicit_source_workspace_root(path: &Path) -> Result<PathBuf, String> {
    let cwd =
        env::current_dir().map_err(|err| format!("failed to read current directory: {err}"))?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let normalized = normalize_path(&absolute);
    let root = normalized.canonicalize().unwrap_or(normalized);
    let required = [
        "Cargo.toml",
        "tools/rocm-oxide-build/Cargo.toml",
        "crates/rocm-oxide-device/Cargo.toml",
        "crates/rocm-oxide-kernel/Cargo.toml",
    ];
    for relative in required {
        if !root.join(relative).is_file() {
            return Err(format!(
                "--local workspace {} is missing {relative}",
                root.display()
            ));
        }
    }
    Ok(root)
}

struct ScaffoldConfig {
    mode_label: String,
    runtime_dependency: String,
    device_api_dependency: String,
    kernel_macro_dependency: String,
    build_tool: BuildToolSource,
    summary_lines: Vec<String>,
    readme: String,
    note: String,
}

enum BuildToolSource {
    SourceWorkspace { runtime_path: String },
    Installed { install_hint: String },
}

impl ScaffoldConfig {
    fn local(project_abs: &Path, device_spike_abs: &Path, source_root: &Path) -> Self {
        // All dependency paths in generated files are relative so that the
        // project + workspace can be moved together without breaking.
        let runtime_path = relative_path_from_to(project_abs, source_root);
        let runtime_path_from_device_spike = relative_path_from_to(device_spike_abs, source_root);

        let device_api_path = runtime_path_from_device_spike.join("crates/rocm-oxide-device");
        let kernel_macro_path = runtime_path_from_device_spike.join("crates/rocm-oxide-kernel");

        let runtime_path = runtime_path.display().to_string();
        let device_api_path = device_api_path.display().to_string();
        let kernel_macro_path = kernel_macro_path.display().to_string();

        let note = format!(
            "note: this project is a local scaffold tied to the ROCm-Oxide workspace at\n      \
             {} via relative path dependencies.\n      \
             Moving only the generated project will break the build.\n      \
             See docs/project-generation.md for options.",
            source_root.display()
        );

        Self {
            mode_label: "local (relative paths to ROCm-Oxide workspace)".to_string(),
            runtime_dependency: format!(r#"{{ path = {runtime_path:?} }}"#),
            device_api_dependency: format!(r#"{{ path = {device_api_path:?} }}"#),
            kernel_macro_dependency: format!(r#"{{ path = {kernel_macro_path:?} }}"#),
            build_tool: BuildToolSource::SourceWorkspace {
                runtime_path: runtime_path.clone(),
            },
            summary_lines: vec![format!(
                "ROCm-Oxide workspace: {runtime_path}  (relative from project root)"
            )],
            readme: local_scaffold_readme(&runtime_path),
            note,
        }
    }

    fn git(git: GitDependency) -> Self {
        let dependency = git.dependency_value();
        let install_hint = git.install_hint();
        let note = format!(
            "note: this project uses ROCm-Oxide git dependencies from {}.\n      \
             It is not fully standalone: install rocm-oxide-build and keep it on PATH,\n      \
             or set ROCM_OXIDE_BUILD=/path/to/rocm-oxide-build before building.",
            git.url
        );

        Self {
            mode_label: "git dependencies (installed build tool required)".to_string(),
            runtime_dependency: dependency.clone(),
            device_api_dependency: dependency.clone(),
            kernel_macro_dependency: dependency,
            build_tool: BuildToolSource::Installed {
                install_hint: install_hint.clone(),
            },
            summary_lines: vec![
                format!("Git repository: {}", git.url),
                format!("Build tool install: {install_hint}"),
            ],
            readme: git_scaffold_readme(&git, &install_hint),
            note,
        }
    }

    fn print_note(&self) {
        println!("{}", self.note);
        println!();
    }

    fn readme(&self) -> String {
        self.readme.clone()
    }
}

fn new_project(args: &[OsString]) -> Result<(), String> {
    let args = NewProjectArgs::parse(args)?;
    if args.standalone {
        return Err(
            "`cargo rocm-oxide new --standalone` is not supported yet.\n\
             Standalone scaffolds require published `rocm-oxide`, `rocm-oxide-device`, \
             `rocm-oxide-kernel`, and `rocm-oxide-build` artifacts.\n\
             Use `cargo rocm-oxide new <path> --local <ROCm-Oxide workspace>` for now."
                .to_string(),
        );
    }

    let path = args.path;
    if path.exists() {
        return Err(format!("target already exists: {}", path.display()));
    }

    // Absolute path of the project that is about to be created. We cannot
    // canonicalize it yet (the directory does not exist), so we normalize
    // lexically instead.
    let cwd =
        env::current_dir().map_err(|err| format!("failed to read current directory: {err}"))?;
    let project_abs = normalize_path(&cwd.join(&path));
    let device_spike_abs = project_abs.join("device-spike");

    let scaffold = if let Some(git) = args.git {
        ScaffoldConfig::git(git)
    } else {
        let source_root = if let Some(source_workspace) = args.source_workspace {
            explicit_source_workspace_root(&source_workspace)?
        } else {
            // Require the source workspace to be reachable so we can compute correct
            // relative paths.
            let root = source_workspace_root().map_err(|_| {
                "cargo rocm-oxide new could not find a ROCm-Oxide source workspace.\n\
                 hint: run from inside the workspace, or pass it explicitly:\n\
                   cargo rocm-oxide new <path> --local /path/to/ROCm-Oxide\n\
                 hint: clone the workspace first if needed:\n\
                   git clone https://github.com/JackSkellet/ROCm-Oxide.git"
                    .to_string()
            })?;
            root.canonicalize().unwrap_or(root)
        };

        ScaffoldConfig::local(&project_abs, &device_spike_abs, &source_root)
    };

    scaffold.print_note();

    fs::create_dir_all(path.join("src"))
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    fs::create_dir_all(path.join("device-spike/src")).map_err(|err| {
        format!(
            "failed to create {}: {err}",
            path.join("device-spike").display()
        )
    })?;

    fs::write(
        path.join("Cargo.toml"),
        format!(
            r#"[package]
name = "rocm-oxide-app"
version = "0.1.0"
edition = "2024"

[dependencies]
rocm-oxide = {runtime_dependency}
"#,
            runtime_dependency = scaffold.runtime_dependency,
        ),
    )
    .map_err(|err| format!("failed to write Cargo.toml: {err}"))?;

    fs::write(
        path.join("build.rs"),
        consumer_build_script(&scaffold.build_tool),
    )
    .map_err(|err| format!("failed to write build.rs: {err}"))?;

    fs::write(
        path.join("device-spike/Cargo.toml"),
        format!(
            r#"[package]
name = "rocm-oxide-app-device"
version = "0.1.0"
edition = "2024"
publish = false

[lib]
crate-type = ["rlib"]

[dependencies]
rocm-oxide-device = {device_api_dependency}
rocm-oxide-kernel = {kernel_macro_dependency}

[profile.release]
panic = "abort"
codegen-units = 1
lto = false
"#,
            device_api_dependency = scaffold.device_api_dependency,
            kernel_macro_dependency = scaffold.kernel_macro_dependency,
        ),
    )
    .map_err(|err| format!("failed to write device-spike/Cargo.toml: {err}"))?;

    fs::write(
        path.join("device-spike/src/lib.rs"),
        r#"#![no_std]

use rocm_oxide_device::prelude::*;
use rocm_oxide_kernel::{kernel, kernel_contract};

#[kernel_contract(len(out)=n)]
#[kernel]
pub unsafe extern "C" fn fill_indices(out: DeviceSliceMut<u32>, n: usize) {
    let index = global_id_x();
    if index < n {
        unsafe { out.write_unchecked(index, index as u32) };
    }
}
"#,
    )
    .map_err(|err| format!("failed to write device-spike/src/lib.rs: {err}"))?;

    fs::write(
        path.join("src/main.rs"),
        r#"use rocm_oxide::prelude::*;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

fn main() -> Result<()> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;

    let n = 256usize;
    let out = DeviceBuffer::<u32>::new(n)?;
    unsafe {
        kernels.fill_indices_launcher().grid_for(n).launch(&out, n)?;
    }
    rocm_oxide::hip::synchronize()?;

    let values = out.copy_to_vec()?;
    for (index, value) in values.iter().copied().enumerate() {
        if value != index as u32 {
            return Err(Error::InvalidLaunch(format!("mismatch at {index}: got {value}")));
        }
    }

    println!("Rust-authored AMDGPU kernel passed on {}", device.arch());
    Ok(())
}
"#,
    )
    .map_err(|err| format!("failed to write src/main.rs: {err}"))?;

    // Select the same nightly toolchain the ROCm-Oxide workspace requires.
    // Without this, cargo may use stable Rust and fail on `-Z build-std=core`.
    fs::write(
        path.join("rust-toolchain.toml"),
        r#"[toolchain]
channel = "nightly"
components = ["rust-src", "clippy", "rustfmt"]
"#,
    )
    .map_err(|err| format!("failed to write rust-toolchain.toml: {err}"))?;

    fs::write(path.join("README.md"), scaffold.readme())
        .map_err(|err| format!("failed to write README.md: {err}"))?;

    println!("created {}", path.display());
    println!();
    println!("  Scaffold mode: {}", scaffold.mode_label);
    for line in &scaffold.summary_lines {
        println!("  {line}");
    }
    println!();
    println!("  Build and run:");
    println!("    cd {}", path.display());
    println!("    cargo rocm-oxide check-consumer");
    println!("    cargo run");
    println!();
    println!("  Note: `cargo rocm-oxide verify` must be run from the ROCm-Oxide");
    println!("  source workspace, not from this project.");
    Ok(())
}

fn local_scaffold_readme(runtime_path: &str) -> String {
    format!(
        r#"# rocm-oxide-app - ROCm-Oxide local scaffold

> **Local scaffold only.** This project was generated by `cargo rocm-oxide new`
> and depends on the ROCm-Oxide source workspace via a relative `path` dependency.
> It is **not** a standalone project. See "Portability" below.

## Build and run

```sh
cargo rocm-oxide check-consumer
cargo run
```

This will:
1. Validate the scaffold's relative paths and required build files.
2. Run `rocm-oxide-build` from the ROCm-Oxide workspace to compile the Rust GPU
   kernel in `device-spike/` for `amdgcn-amd-amdhsa`.
3. Produce a `.hsaco` code object and a typed `DeviceKernels` binding.
4. Compile and run `src/main.rs`, which loads the kernel and verifies it on the GPU.

## Write your own kernel

1. Open `device-spike/src/lib.rs`.
2. Add a `#[kernel]` function. Length contracts for generated binding runtime
   validation are expressed with `#[kernel_contract(...)]` immediately before
   the function. See `docs/wiki/kernel-contracts.md` in the ROCm-Oxide workspace.
3. `cargo run` picks up the change automatically.

## Portability

This project was scaffolded with the following relative path to the ROCm-Oxide
workspace:

    {runtime_path}   (relative to this project's root)

**What works:**
- Moving this project and the ROCm-Oxide workspace together, preserving the
  relative path between them.

**What breaks:**
- Moving only the ROCm-Oxide workspace without moving this project.
- Cloning this project on a machine where ROCm-Oxide is not at the same
  relative path.
- `cargo publish` because `path` dependencies are rejected by crates.io.

**Escape hatch - pre-built `rocm-oxide-build`:**
Set `ROCM_OXIDE_BUILD=/path/to/rocm-oxide-build` to point `build.rs` to a
pre-compiled build tool binary instead of using the source workspace. The
`rocm-oxide` runtime dependency in `Cargo.toml` would still need to be updated
to a crates.io version once one is published.

See `docs/project-generation.md` in the ROCm-Oxide workspace for the full
portability roadmap.

## Prerequisites

- AMD GPU (RDNA 2+, RDNA 3+, RDNA 4+, or CDNA 2/3)
- ROCm 7.2 validated at `/opt/rocm` (or set `ROCM_PATH`)
- `/opt/rocm/bin` on `PATH` for ROCm executables such as `clang`
- `/opt/rocm/lib/llvm/bin` on `PATH` for LLVM tools such as `llc` and `llvm-readelf`
- Rust nightly with `rust-src` selected by `rust-toolchain.toml`

Before building, verify all tools are present by running doctor from the
ROCm-Oxide source workspace:

```sh
cd {runtime_path}
cargo rocm-oxide doctor
```

Fix any FAIL items before running `cargo run` in this project. Copy the doctor
output between the dashed markers when filing a bug report.
"#
    )
}

fn git_scaffold_readme(git: &GitDependency, install_hint: &str) -> String {
    let dependency = git.dependency_value();
    format!(
        r#"# rocm-oxide-app - ROCm-Oxide git scaffold

> **Git scaffold.** This project depends on ROCm-Oxide crates from a git
> repository. It no longer needs the ROCm-Oxide source workspace at a fixed
> relative path, but it still needs the `rocm-oxide-build` binary available
> through `ROCM_OXIDE_BUILD` or `PATH`.

## One-time build tool install

Install the matching build tool from the same git source:

```sh
{install_hint}
```

You can also set an explicit binary path:

```sh
export ROCM_OXIDE_BUILD=/path/to/rocm-oxide-build
```

## Build and run

```sh
cargo rocm-oxide check-consumer
cargo run
```

This will:
1. Validate the scaffold files.
2. Run `rocm-oxide-build` from `ROCM_OXIDE_BUILD` or `PATH`.
3. Compile the Rust GPU kernel in `device-spike/` for `amdgcn-amd-amdhsa`.
4. Produce a `.hsaco` code object and a typed `DeviceKernels` binding.
5. Compile and run `src/main.rs`, which loads the kernel and verifies it on the GPU.

## Dependency source

Generated dependency value:

```toml
{dependency}
```

Repository:

```text
{git_url}
```

## Portability

**What works:**
- Cloning this project on another machine without preserving a local
  ROCm-Oxide checkout path.
- Pinning with `--rev` or `--tag` so the runtime, device API, and proc macro
  crates come from the same source revision.

**What still is not standalone:**
- `rocm-oxide-build` is not consumed as a build-dependency yet.
- `cargo publish` is still blocked until ROCm-Oxide crates are published to
  crates.io.

## Write your own kernel

1. Open `device-spike/src/lib.rs`.
2. Add a `#[kernel]` function.
3. Use `#[kernel_contract(...)]` for generated binding runtime validation.
4. `cargo run` picks up the change automatically.

## Prerequisites

- AMD GPU (RDNA 2+, RDNA 3+, RDNA 4+, or CDNA 2/3)
- ROCm 7.2 validated at `/opt/rocm` (or set `ROCM_PATH`)
- `/opt/rocm/bin` on `PATH` for ROCm executables such as `clang`
- `/opt/rocm/lib/llvm/bin` on `PATH` for LLVM tools such as `llc` and `llvm-readelf`
- Rust nightly with `rust-src` selected by `rust-toolchain.toml`
"#,
        git_url = git.url,
    )
}

fn consumer_build_script(build_tool: &BuildToolSource) -> String {
    let (runtime_path, install_hint) = match build_tool {
        BuildToolSource::SourceWorkspace { runtime_path } => (
            format!("Some({runtime_path:?})"),
            "cargo install --path <rocm-oxide-workspace>/tools/rocm-oxide-build".to_string(),
        ),
        BuildToolSource::Installed { install_hint } => ("None".to_string(), install_hint.clone()),
    };
    let install_hint = format!("{install_hint:?}");
    r#"use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEVICE_CRATE: &str = "device-spike";
const OUTPUT_STEM: &str = "rocm_oxide_device_spike";
const RUNTIME_PATH: Option<&str> = __ROCM_OXIDE_RUNTIME_PATH__;
const BUILD_TOOL_INSTALL_HINT: &str = __ROCM_OXIDE_BUILD_TOOL_INSTALL_HINT__;

fn main() {
    println!("cargo:rerun-if-changed=device-spike/Cargo.toml");
    println!("cargo:rerun-if-changed=device-spike/src");
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_OXIDE_BUILD");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    let mut command = build_tool_command();
    command.args(["--crate", DEVICE_CRATE, "--output-stem", OUTPUT_STEM]);
    let output = command.output().expect("failed to run rocm-oxide-build");
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
    let hsaco = Path::new(hsaco);
    let bindings = PathBuf::from(format!("{stem}.bindings.rs"));
    let metadata = PathBuf::from(format!("{stem}.metadata.json"));
    let manifest = PathBuf::from(format!("{stem}.manifest.json"));

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let hsaco_out = out_dir.join(format!("{OUTPUT_STEM}.hsaco"));
    let bindings_out = out_dir.join(format!("{OUTPUT_STEM}.bindings.rs"));
    let metadata_out = out_dir.join(format!("{OUTPUT_STEM}.metadata.json"));
    let manifest_out = out_dir.join(format!("{OUTPUT_STEM}.manifest.json"));

    copy_artifact(hsaco, &hsaco_out, "hsaco");
    copy_artifact(&bindings, &bindings_out, "bindings");
    copy_artifact(&metadata, &metadata_out, "metadata");
    copy_artifact(&manifest, &manifest_out, "manifest");

    println!("cargo:rustc-env=ROCM_OXIDE_DEVICE_HSACO={}", hsaco_out.display());
    println!("cargo:rustc-env=ROCM_OXIDE_DEVICE_BINDINGS={}", bindings_out.display());
    println!("cargo:rustc-env=ROCM_OXIDE_DEVICE_METADATA={}", metadata_out.display());
    println!("cargo:rustc-env=ROCM_OXIDE_DEVICE_MANIFEST={}", manifest_out.display());
}

fn build_tool_command() -> Command {
    if let Some(path) = env::var_os("ROCM_OXIDE_BUILD").filter(|value| !value.is_empty()) {
        return Command::new(path);
    }

    if let Some(runtime_path) = RUNTIME_PATH {
        let source_manifest = Path::new(runtime_path).join("tools/rocm-oxide-build/Cargo.toml");
        if source_manifest.is_file() {
            let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
            let mut command = Command::new(cargo);
            command
                .arg("run")
                .arg("--quiet")
                .arg("--manifest-path")
                .arg(source_manifest)
                .arg("--");
            return command;
        }
    }

    // Try rocm-oxide-build on PATH as a last resort.
    if which_on_path("rocm-oxide-build") {
        return Command::new("rocm-oxide-build");
    }

    // Nothing found: emit an actionable error at build time.
    panic!(
        "\n\
         \n\
         rocm-oxide-build not found.\n\
         \n\
         This scaffold was generated with RUNTIME_PATH = {:?}.\n\
         Local path scaffolds need that path to contain tools/rocm-oxide-build/Cargo.toml.\n\
         Git scaffolds need rocm-oxide-build installed separately.\n\
         \n\
         Fix options:\n\
           1. Set ROCM_OXIDE_BUILD=/path/to/pre-built/rocm-oxide-build in your\n\
              environment or .cargo/config.toml [env] section.\n\
           2. Install rocm-oxide-build onto PATH via:\n\
                {}\n\
           3. For local path scaffolds, keep the ROCm-Oxide workspace at the\n\
              generated relative path.\n\
         \n\
         See README.md in this project for details.\n",
        RUNTIME_PATH,
        BUILD_TOOL_INSTALL_HINT,
    )
}

fn which_on_path(name: &str) -> bool {
    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

fn copy_artifact(from: &Path, to: &Path, label: &str) {
    if !from.is_file() {
        panic!("missing generated {label}: {}", from.display());
    }
    fs::copy(from, to)
        .unwrap_or_else(|err| panic!("failed to copy generated {label} {} to {}: {err}", from.display(), to.display()));
}
"#
    .replace("__ROCM_OXIDE_RUNTIME_PATH__", &runtime_path)
    .replace("__ROCM_OXIDE_BUILD_TOOL_INSTALL_HINT__", &install_hint)
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

/// Return the path to `to` expressed relative to `from`.
///
/// Both paths must be absolute. Neither needs to exist on disk (no
/// canonicalization is performed). The result is equivalent to what
/// `pathdiff::diff_paths(to, from)` would return.
fn relative_path_from_to(from: &Path, to: &Path) -> PathBuf {
    let from_components: Vec<_> = from.components().collect();
    let to_components: Vec<_> = to.components().collect();

    let common_len = from_components
        .iter()
        .zip(to_components.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let up_count = from_components.len() - common_len;
    let mut result = PathBuf::new();
    for _ in 0..up_count {
        result.push("..");
    }
    for component in &to_components[common_len..] {
        result.push(component);
    }
    if result.as_os_str().is_empty() {
        result.push(".");
    }
    result
}

/// Collapse `.` and `..` components in an absolute path without touching the
/// filesystem (i.e. without calling `canonicalize()`). Symlinks are not
/// resolved — this is a purely lexical operation.
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_path_sibling_directories() {
        let from = Path::new("/home/user/my-project");
        let to = Path::new("/home/user/ROCm-Oxide");
        assert_eq!(
            relative_path_from_to(from, to),
            PathBuf::from("../ROCm-Oxide")
        );
    }

    #[test]
    fn relative_path_project_inside_workspace() {
        // project scaffolded as a child of the workspace
        let from = Path::new("/home/user/ROCm-Oxide/my-project");
        let to = Path::new("/home/user/ROCm-Oxide");
        assert_eq!(relative_path_from_to(from, to), PathBuf::from(".."));
    }

    #[test]
    fn relative_path_device_spike_to_sibling_workspace() {
        // device-spike/ is one level deeper than the project root
        let from = Path::new("/home/user/my-project/device-spike");
        let to = Path::new("/home/user/ROCm-Oxide");
        assert_eq!(
            relative_path_from_to(from, to),
            PathBuf::from("../../ROCm-Oxide")
        );
    }

    #[test]
    fn relative_path_deeply_nested() {
        let from = Path::new("/home/user/projects/deep/my-project");
        let to = Path::new("/home/user/ROCm-Oxide");
        assert_eq!(
            relative_path_from_to(from, to),
            PathBuf::from("../../../ROCm-Oxide")
        );
    }

    #[test]
    fn relative_path_same_directory() {
        let from = Path::new("/home/user/dir");
        let to = Path::new("/home/user/dir");
        assert_eq!(relative_path_from_to(from, to), PathBuf::from("."));
    }

    #[test]
    fn normalize_removes_dot() {
        assert_eq!(
            normalize_path(Path::new("/a/./b/./c")),
            PathBuf::from("/a/b/c")
        );
    }

    #[test]
    fn normalize_removes_dotdot() {
        assert_eq!(
            normalize_path(Path::new("/a/b/../c")),
            PathBuf::from("/a/c")
        );
    }

    #[test]
    fn consumer_build_script_uses_relative_path() {
        let script = consumer_build_script(&BuildToolSource::SourceWorkspace {
            runtime_path: "../ROCm-Oxide".to_string(),
        });
        assert!(
            script.contains(r#"Some("../ROCm-Oxide")"#),
            "build.rs should embed the relative runtime path"
        );
    }

    #[test]
    fn consumer_build_script_no_home_paths() {
        // Regression guard: generated build.rs must not bake in absolute home
        // paths when a relative path is passed.
        let script = consumer_build_script(&BuildToolSource::SourceWorkspace {
            runtime_path: "../ROCm-Oxide".to_string(),
        });
        assert!(
            !script.contains("/home/"),
            "generated build.rs must not contain absolute /home/ paths"
        );
        assert!(
            !script.contains("/root/"),
            "generated build.rs must not contain absolute /root/ paths"
        );
    }

    #[test]
    fn git_dependency_formats_optional_revision() {
        let dependency = GitDependency {
            url: "https://github.com/JackSkellet/ROCm-Oxide".to_string(),
            reference: Some(GitReference::Rev("abc1234".to_string())),
        };
        assert_eq!(
            dependency.dependency_value(),
            r#"{ git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }"#
        );
        assert_eq!(
            dependency.install_hint(),
            r#"cargo install --git "https://github.com/JackSkellet/ROCm-Oxide" rocm-oxide-build --rev "abc1234""#
        );
    }

    #[test]
    fn consumer_build_script_git_mode_omits_runtime_path() {
        let script = consumer_build_script(&BuildToolSource::Installed {
            install_hint:
                r#"cargo install --git "https://github.com/JackSkellet/ROCm-Oxide" rocm-oxide-build"#
                    .to_string(),
        });
        assert!(
            script.contains("const RUNTIME_PATH: Option<&str> = None;"),
            "git scaffolds should not embed a local source workspace path"
        );
        assert!(
            script.contains("cargo install --git"),
            "git scaffolds should include the build-tool install hint"
        );
    }
}
