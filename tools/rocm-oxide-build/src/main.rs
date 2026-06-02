use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const TARGET: &str = "amdgcn-amd-amdhsa";
const DEFAULT_ROCM_PATH: &str = "/opt/rocm";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    if args.doctor {
        return doctor();
    }
    if let Some(path) = &args.inspect_metadata {
        return inspect_metadata(path);
    }

    let root = workspace_root()?;
    let device_crate = root.join(&args.device_crate);
    let arch = args.arch.or_else(detect_arch).ok_or_else(|| {
        "failed to detect ROCm GPU architecture; pass --arch gfx... or set ROCM_OXIDE_ARCH"
            .to_string()
    })?;
    validate_gpu_arch(&arch)?;
    let tools = ToolPaths::discover()?;
    let debug_info = device_debug_info_enabled();

    ensure_tool("cargo", &["--version"])?;
    ensure_tool("rustc", &["--version"])?;
    ensure_amdgpu_target()?;
    ensure_rust_src()?;

    let device_crates = discover_device_crate_bundle(&device_crate)?;
    let mut kernels = BTreeMap::new();
    let mut device_structs = BTreeMap::new();
    let mut device_globals = BTreeMap::new();
    let mut kernel_irs = Vec::new();
    let mut objects = Vec::new();
    let mut link_inputs = Vec::new();

    for device_crate in &device_crates {
        let crate_kernels = discover_kernels(device_crate)?;
        for device_global in discover_device_globals(device_crate)?.values() {
            if let Some(previous) =
                device_globals.insert(device_global.name.clone(), device_global.clone())
            {
                return Err(format!(
                    "duplicate marked device global `{}`\n  first: {}\n  again: {}",
                    device_global.name, previous.span, device_global.span
                ));
            }
        }
        let mut crate_device_structs = discover_device_structs(device_crate)?;
        let used_device_structs =
            used_device_struct_names(crate_kernels.values(), crate_device_structs.keys());
        if !used_device_structs.is_empty() {
            let rustc_layouts =
                query_rustc_device_layouts(device_crate, &arch, &used_device_structs)?;
            for name in &used_device_structs {
                let device_struct = crate_device_structs
                    .get_mut(name)
                    .ok_or_else(|| format!("internal error: missing device struct `{name}`"))?;
                let layout = rustc_layouts.get(name).ok_or_else(|| {
                    format!(
                        "{}: rustc did not report AMDGPU layout facts for device struct `{name}`",
                        device_struct.span
                    )
                })?;
                let mut layout = layout.clone();
                validate_rustc_layout(device_struct, &mut layout)?;
                device_struct.layout = layout;
                device_struct.layout_source = DeviceStructLayoutSource::Rustc;
            }
        }
        for device_struct in crate_device_structs.values() {
            if !used_device_structs.contains(&device_struct.name) {
                continue;
            }
            if let Some(previous) =
                device_structs.insert(device_struct.name.clone(), device_struct.clone())
            {
                return Err(format!(
                    "duplicate device struct `{}`\n  first: {}\n  again: {}",
                    device_struct.name, previous.span, device_struct.span
                ));
            }
        }
        if crate_kernels.is_empty() {
            continue;
        }
        for kernel in crate_kernels.values() {
            if let Some(previous) = kernels.insert(kernel.name.clone(), kernel.clone()) {
                return Err(format!(
                    "duplicate #[kernel] symbol `{}`\n  first: {}\n  again: {}",
                    kernel.name, previous.span, kernel.span
                ));
            }
        }
        build_device_crate(device_crate, &arch, debug_info)?;
        let package_name = package_name(device_crate)?;
        let artifact_stem = if device_crate == &device_crates[0] {
            args.output_stem.clone()
        } else {
            format!("{}_{}", args.output_stem, package_name.replace('-', "_"))
        };
        let release_dir = device_crate.join("target").join(TARGET).join("release");
        let deps_dir = release_dir.join("deps");
        let input_ir = newest_llvm_ir(&deps_dir, &package_name)?;
        let kernel_ir = release_dir.join(format!("{artifact_stem}.kernel.ll"));
        let obj = release_dir.join(format!("{artifact_stem}.o"));

        let kernel_names = crate_kernels.keys().cloned().collect::<BTreeSet<_>>();
        let input_kernels = crate_kernels.keys().cloned().collect::<Vec<_>>();
        let source = fs::read_to_string(&input_ir)
            .map_err(|err| format!("failed to read {}: {err}", input_ir.display()))?;
        let transformed = compiler_step("rewrite Rust-emitted LLVM IR", || {
            transform_ir(&source, &kernel_names, &crate_kernels, &device_globals)
        })?;
        let transformed = strip_rocm_llc_unsupported_debug_metadata(&transformed);
        fs::write(&kernel_ir, transformed)
            .map_err(|err| format!("failed to write {}: {err}", kernel_ir.display()))?;
        kernel_irs.push(kernel_ir.clone());

        let mut lower = Command::new(&tools.llc.path);
        lower
            .arg("-mtriple=amdgcn-amd-amdhsa")
            .arg(format!("-mcpu={arch}"))
            .arg("-filetype=obj");
        lower.arg(&kernel_ir).arg("-o").arg(&obj);
        run_command(&mut lower, "lower LLVM IR with ROCm llc")?;
        if crate_kernels.contains_key("lds_block_sum")
            || crate_kernels.contains_key("static_lds_reverse")
        {
            verify_lds_artifacts(&kernel_ir, &obj, &tools.llvm_objdump.path)?;
        }
        if crate_kernels.contains_key("scoped_atomics") {
            verify_scoped_atomic_artifacts(&kernel_ir, &obj, &tools.llvm_objdump.path)?;
        }
        objects.push(obj.clone());
        link_inputs.push(LinkInput {
            package_name,
            llvm_ir: kernel_ir,
            object: obj,
            kernels: input_kernels,
        });
    }

    if kernels.is_empty() {
        return Err("no #[kernel] functions found in device crate bundle".to_string());
    }

    let kernel_names = kernels.keys().cloned().collect::<BTreeSet<_>>();
    let release_dir = device_crate.join("target").join(TARGET).join("release");
    fs::create_dir_all(&release_dir)
        .map_err(|err| format!("failed to create {}: {err}", release_dir.display()))?;
    let hsaco = release_dir.join(format!("{}.hsaco", args.output_stem));
    let metadata = release_dir.join(format!("{}.metadata.json", args.output_stem));
    let bindings = release_dir.join(format!("{}.bindings.rs", args.output_stem));

    let mut link = Command::new(&tools.clang.path);
    link.arg("-target")
        .arg("amdgcn-amd-amdhsa")
        .arg(format!("-mcpu={arch}"));
    if debug_info {
        link.arg("-g");
    }
    for obj in &objects {
        link.arg(obj);
    }
    link.arg("-o").arg(&hsaco);
    run_command(&mut link, "link AMDGPU code object with ROCm clang")?;

    validate_code_object(&hsaco, &kernel_names, &tools.llvm_readelf.path)?;
    let mut code_object_metadata = read_code_object_metadata(&hsaco, &tools.llvm_readelf.path)?;
    for kernel_ir in &kernel_irs {
        annotate_dynamic_shared_mem_from_ir(&mut code_object_metadata, kernel_ir)?;
    }
    validate_code_object_metadata(&code_object_metadata, &kernel_names)?;
    write_metadata(
        &metadata,
        &arch,
        &hsaco,
        &link_inputs,
        &kernels,
        &device_structs,
        &device_globals,
        &code_object_metadata,
    )?;
    write_bindings(
        &bindings,
        &hsaco,
        &kernels,
        &device_structs,
        &device_globals,
        &code_object_metadata,
    )?;
    println!("{}", hsaco.display());
    Ok(())
}

#[derive(Debug)]
struct Args {
    device_crate: PathBuf,
    arch: Option<String>,
    output_stem: String,
    doctor: bool,
    inspect_metadata: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut device_crate = PathBuf::from("device-spike");
        let mut arch = env::var("ROCM_OXIDE_ARCH").ok().filter(|s| !s.is_empty());
        let mut output_stem = "rocm_oxide_device_spike".to_string();
        let mut doctor = false;
        let mut inspect_metadata = None;

        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--doctor" => doctor = true,
                "--inspect-metadata" => {
                    inspect_metadata = Some(iter.next().map(PathBuf::from).ok_or_else(|| {
                        "--inspect-metadata requires a metadata path".to_string()
                    })?);
                }
                "--crate" => {
                    device_crate = iter
                        .next()
                        .map(PathBuf::from)
                        .ok_or_else(|| "--crate requires a path".to_string())?;
                }
                "--arch" => {
                    arch = Some(
                        iter.next()
                            .ok_or_else(|| "--arch requires a gfx target".to_string())?,
                    );
                }
                "--output-stem" => {
                    output_stem = iter
                        .next()
                        .ok_or_else(|| "--output-stem requires a filename stem".to_string())?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        Ok(Self {
            device_crate,
            arch,
            output_stem,
            doctor,
            inspect_metadata,
        })
    }
}

fn print_help() {
    println!(
        "Usage: rocm-oxide-build [--crate device-spike] [--arch <gfx arch>] [--output-stem name]\n       rocm-oxide-build --doctor\n       rocm-oxide-build --inspect-metadata path/to/metadata.json"
    );
}

fn doctor() -> Result<(), String> {
    println!("ROCm-Oxide doctor");
    report_tool("cargo", &["--version"])?;
    report_tool("rustc", &["--version"])?;
    report_amdgpu_target()?;
    report_rust_src()?;
    report_tool_search_order();
    let tools = ToolPaths::discover()?;
    report_rocm_tool("ROCm llc", &tools.llc)?;
    report_llc_amdgcn(&tools.llc)?;
    report_rocm_tool("ROCm clang", &tools.clang)?;
    report_rocm_tool("ROCm llvm-readelf", &tools.llvm_readelf)?;
    report_rocm_tool("ROCm llvm-objdump", &tools.llvm_objdump)?;
    let rocminfo = find_rocm_tool("ROCMINFO", "rocminfo", ToolLayout::Bin, &[])?;
    let rocm_agent_enumerator = find_rocm_tool(
        "ROCM_AGENT_ENUMERATOR",
        "rocm_agent_enumerator",
        ToolLayout::Bin,
        &[],
    )?;

    let rocminfo_summary = report_rocminfo(&rocminfo)?;
    let arch = match rocminfo_summary.arch {
        Some(arch) => {
            validate_gpu_arch(&arch)?;
            println!("ok: selected AMD GPU architecture {arch}");
            arch
        }
        None => {
            return Err(
                "failed to detect AMD GPU architecture; set ROCM_OXIDE_ARCH=gfx...".to_string(),
            );
        }
    };
    report_rocm_agents(&rocm_agent_enumerator, &arch)?;

    report_core_build_probe(&arch)?;
    println!("ok: doctor report complete; build prerequisites are present");
    Ok(())
}

fn report_tool(program: &str, args: &[&str]) -> Result<(), String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run {program}: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {:?} failed:\n{}",
            args,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first = stdout.lines().next().unwrap_or("<no version output>");
    println!("ok: {program} {first}");
    Ok(())
}

fn inspect_metadata(path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let arch = find_json_string(&text, "arch").unwrap_or_else(|| "<unknown>".to_string());
    let target = find_json_string(&text, "target").unwrap_or_else(|| "<unknown>".to_string());
    let resources = parse_kernel_resource_rows(&text);
    let kernel_count = resources.len();
    let contract_count = text.matches("\"required_len\":").count();
    let linked_object_count = text.matches("\"package\":").count();
    let device_struct_count = text.matches("\"layout_source\":").count();
    let max_workgroup = max_json_u32(&text, "max_flat_workgroup_size");
    let max_vgpr = max_json_u32(&text, "vgpr_count");
    let max_sgpr = max_json_u32(&text, "sgpr_count");
    let max_lds = max_json_u32(&text, "group_segment_fixed_size");
    let max_private = max_json_u32(&text, "private_segment_fixed_size");
    let dynamic_lds = text.matches("\"uses_dynamic_shared_mem\": true").count();
    let dynamic_stack = text.matches("\"uses_dynamic_stack\": true").count();
    println!("metadata: {}", path.display());
    println!("target: {target}");
    println!("arch: {arch}");
    println!("kernels: {kernel_count}");
    println!("buffer contracts: {contract_count}");
    println!("linked objects: {linked_object_count}");
    println!("device structs: {device_struct_count}");
    if let Some(value) = max_workgroup {
        println!("max flat workgroup size: {value}");
    }
    if let Some(value) = max_vgpr {
        println!("max VGPR count: {value}");
    }
    if let Some(value) = max_sgpr {
        println!("max SGPR count: {value}");
    }
    if let Some(value) = max_lds {
        println!("max static LDS bytes: {value}");
    }
    if let Some(value) = max_private {
        println!("max private segment bytes: {value}");
    }
    println!("kernels using dynamic LDS: {dynamic_lds}");
    println!("kernels using dynamic stack: {dynamic_stack}");
    if !resources.is_empty() {
        println!();
        println!("per-kernel resources:");
        println!(
            "{:<48} {:>5} {:>5} {:>5} {:>7} {:>6} {:>7} {:>8} {:>5} {:>5}",
            "kernel",
            "vgpr",
            "sgpr",
            "wave",
            "lds",
            "dynlds",
            "private",
            "kernarg",
            "spill",
            "stack"
        );
        for row in resources {
            println!(
                "{:<48} {:>5} {:>5} {:>5} {:>7} {:>6} {:>7} {:>8} {:>5} {:>5}",
                row.name,
                display_opt(row.vgpr_count),
                display_opt(row.sgpr_count),
                display_opt(row.wavefront_size),
                display_opt(row.group_segment_fixed_size),
                display_bool(row.uses_dynamic_shared_mem),
                display_opt(row.private_segment_fixed_size),
                display_opt(row.kernarg_segment_size),
                display_spills(row.sgpr_spill_count, row.vgpr_spill_count),
                display_bool(row.uses_dynamic_stack),
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct KernelResourceRow {
    name: String,
    kernarg_segment_size: Option<u32>,
    max_flat_workgroup_size: Option<u32>,
    group_segment_fixed_size: Option<u32>,
    private_segment_fixed_size: Option<u32>,
    sgpr_count: Option<u32>,
    vgpr_count: Option<u32>,
    sgpr_spill_count: Option<u32>,
    vgpr_spill_count: Option<u32>,
    wavefront_size: Option<u32>,
    uses_dynamic_shared_mem: Option<bool>,
    uses_dynamic_stack: Option<bool>,
}

fn parse_kernel_resource_rows(text: &str) -> Vec<KernelResourceRow> {
    let mut rows = Vec::new();
    let mut current: Option<KernelResourceRow> = None;
    let mut in_code_object = false;

    for line in text.lines() {
        if matches!(line.trim(), "\"structs\": [" | "\"globals\": [") {
            break;
        }
        if line.starts_with("      \"name\":") {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = find_json_string(line, "name").map(|name| KernelResourceRow {
                name,
                ..KernelResourceRow::default()
            });
            in_code_object = false;
            continue;
        }

        let trimmed = line.trim();
        if trimmed == "\"code_object\": {" {
            in_code_object = true;
            continue;
        }
        if !in_code_object {
            continue;
        }
        if trimmed.trim_end_matches(',') == "}" {
            in_code_object = false;
            continue;
        }

        if let Some(row) = current.as_mut() {
            parse_kernel_resource_field(row, trimmed);
        }
    }

    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

fn parse_kernel_resource_field(row: &mut KernelResourceRow, line: &str) {
    if let Some(value) = json_u32_field(line, "kernarg_segment_size") {
        row.kernarg_segment_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "max_flat_workgroup_size") {
        row.max_flat_workgroup_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "group_segment_fixed_size") {
        row.group_segment_fixed_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "private_segment_fixed_size") {
        row.private_segment_fixed_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "sgpr_count") {
        row.sgpr_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "vgpr_count") {
        row.vgpr_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "sgpr_spill_count") {
        row.sgpr_spill_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "vgpr_spill_count") {
        row.vgpr_spill_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "wavefront_size") {
        row.wavefront_size = Some(value);
    } else if let Some(value) = json_bool_field(line, "uses_dynamic_shared_mem") {
        row.uses_dynamic_shared_mem = Some(value);
    } else if let Some(value) = json_bool_field(line, "uses_dynamic_stack") {
        row.uses_dynamic_stack = Some(value);
    }
}

fn json_u32_field(line: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\": ");
    let value = line.trim().strip_prefix(&needle)?;
    value.trim_end_matches(',').parse::<u32>().ok()
}

fn json_bool_field(line: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\": ");
    match line.trim().strip_prefix(&needle)?.trim_end_matches(',') {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn display_opt(value: Option<u32>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn display_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "-",
    }
}

fn display_spills(sgpr: Option<u32>, vgpr: Option<u32>) -> String {
    match (sgpr, vgpr) {
        (Some(sgpr), Some(vgpr)) => format!("{sgpr}/{vgpr}"),
        (Some(sgpr), None) => format!("{sgpr}/-"),
        (None, Some(vgpr)) => format!("-/{vgpr}"),
        (None, None) => "-".to_string(),
    }
}

fn find_json_string(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\": \"");
    let start = text.find(&needle)? + needle.len();
    let rest = &text[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn max_json_u32(text: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\": ");
    text.lines()
        .filter_map(|line| {
            let value = line.trim().strip_prefix(&needle)?;
            value.trim_end_matches(',').parse::<u32>().ok()
        })
        .max()
}

fn workspace_root() -> Result<PathBuf, String> {
    env::current_dir().map_err(|err| format!("failed to get current directory: {err}"))
}

fn detect_arch() -> Option<String> {
    let rocminfo = find_rocm_tool("ROCMINFO", "rocminfo", ToolLayout::Bin, &[]).ok()?;
    inspect_rocminfo(&rocminfo.path).ok()?.arch
}

fn ensure_tool(program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|err| format!("failed to run {program}: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} {:?} failed", args))
    }
}

#[derive(Debug, Clone)]
struct ToolPaths {
    llc: ToolPath,
    clang: ToolPath,
    llvm_readelf: ToolPath,
    llvm_objdump: ToolPath,
}

impl ToolPaths {
    fn discover() -> Result<Self, String> {
        Ok(Self {
            llc: find_rocm_tool("ROCM_OXIDE_LLC", "llc", ToolLayout::Llvm, &["--version"])?,
            clang: find_rocm_tool(
                "ROCM_OXIDE_CLANG",
                "clang",
                ToolLayout::Llvm,
                &["--version"],
            )?,
            llvm_readelf: find_rocm_tool(
                "ROCM_OXIDE_LLVM_READELF",
                "llvm-readelf",
                ToolLayout::Llvm,
                &["--version"],
            )?,
            llvm_objdump: find_rocm_tool(
                "ROCM_OXIDE_LLVM_OBJDUMP",
                "llvm-objdump",
                ToolLayout::Llvm,
                &["--version"],
            )?,
        })
    }
}

#[derive(Debug, Clone)]
struct ToolPath {
    path: PathBuf,
    source: String,
}

#[derive(Debug, Clone, Copy)]
enum ToolLayout {
    Llvm,
    Bin,
}

fn find_rocm_tool(
    env_var: &str,
    name: &str,
    layout: ToolLayout,
    check_args: &[&str],
) -> Result<ToolPath, String> {
    let mut candidates = Vec::<ToolPath>::new();
    if let Some(path) = env::var_os(env_var).filter(|value| !value.is_empty()) {
        push_tool_candidate(&mut candidates, PathBuf::from(path), env_var);
    }
    for (source, root) in rocm_roots() {
        for path in rocm_tool_paths(&root, name, layout) {
            push_tool_candidate(&mut candidates, path, &source);
        }
    }
    push_tool_candidate(&mut candidates, PathBuf::from(name), "PATH");

    for candidate in &candidates {
        if tool_works(&candidate.path, check_args) {
            return Ok(candidate.clone());
        }
    }

    Err(format_missing_rocm_tool(env_var, name, &candidates))
}

fn validate_gpu_arch(arch: &str) -> Result<(), String> {
    let valid = arch.starts_with("gfx")
        && arch.len() > 3
        && arch[3..]
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() || ch == '_');
    if valid {
        Ok(())
    } else {
        Err(format!(
            "unsupported GPU architecture `{arch}`; expected a ROCm gfx target such as gfx1100 or gfx1201. Pass --arch gfx... or set ROCM_OXIDE_ARCH=gfx..."
        ))
    }
}

fn format_missing_rocm_tool(env_var: &str, name: &str, candidates: &[ToolPath]) -> String {
    let checked = candidates
        .iter()
        .map(|candidate| format!("[{}] {}", candidate.source, candidate.path.display()))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "could not find `{name}`; checked candidates: {checked}. Set {env_var}=/path/to/{name}, ROCM_PATH=/path/to/rocm, HIP_PATH=/path/to/rocm, or install ROCm tools under /opt/rocm"
    )
}

fn rocm_roots() -> Vec<(String, PathBuf)> {
    let mut roots = Vec::new();
    push_rocm_root_from_env(&mut roots, "ROCM_PATH");
    push_rocm_root_from_env(&mut roots, "HIP_PATH");
    push_rocm_root(&mut roots, "/opt/rocm", PathBuf::from(DEFAULT_ROCM_PATH));
    roots
}

fn push_rocm_root_from_env(roots: &mut Vec<(String, PathBuf)>, env_var: &str) {
    let Some(path) = env::var_os(env_var).filter(|value| !value.is_empty()) else {
        return;
    };
    let path = PathBuf::from(path);
    push_rocm_root(roots, env_var, path.clone());
    if path.file_name().is_some_and(|name| name == "hip")
        && let Some(parent) = path.parent()
    {
        push_rocm_root(roots, &format!("{env_var} parent"), parent.to_path_buf());
    }
}

fn push_rocm_root(roots: &mut Vec<(String, PathBuf)>, source: &str, path: PathBuf) {
    if !roots.iter().any(|(_, existing)| existing == &path) {
        roots.push((source.to_string(), path));
    }
}

fn rocm_tool_paths(root: &Path, name: &str, layout: ToolLayout) -> Vec<PathBuf> {
    match layout {
        ToolLayout::Llvm => ["lib/llvm/bin", "llvm/bin", "bin"]
            .into_iter()
            .map(|dir| root.join(dir).join(name))
            .collect(),
        ToolLayout::Bin => vec![root.join("bin").join(name), root.join(name)],
    }
}

fn push_tool_candidate(candidates: &mut Vec<ToolPath>, path: PathBuf, source: &str) {
    if !candidates.iter().any(|candidate| candidate.path == path) {
        candidates.push(ToolPath {
            path,
            source: source.to_string(),
        });
    }
}

fn tool_works(program: &Path, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn report_tool_search_order() {
    let roots = rocm_roots()
        .into_iter()
        .map(|(source, path)| format!("{source}={}", path.display()))
        .collect::<Vec<_>>()
        .join(", ");
    println!("ok: ROCm tool search order: explicit tool env, {roots}, PATH");
}

fn report_rocm_tool(label: &str, tool: &ToolPath) -> Result<(), String> {
    let output = Command::new(&tool.path)
        .arg("--version")
        .output()
        .map_err(|err| format!("failed to run {}: {err}", tool.path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version failed:\n{}",
            tool.path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first = stdout.lines().next().unwrap_or("<no version output>");
    println!(
        "ok: {label} {} [{}] ({first})",
        tool.path.display(),
        tool.source
    );
    Ok(())
}

fn report_llc_amdgcn(llc: &ToolPath) -> Result<(), String> {
    let output = Command::new(&llc.path)
        .arg("--version")
        .output()
        .map_err(|err| format!("failed to run {} --version: {err}", llc.path.display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("amdgcn") {
        println!("ok: llc supports the amdgcn backend");
        Ok(())
    } else {
        Err(format!(
            "{} does not report amdgcn backend support; set ROCM_OXIDE_LLC to ROCm's llc",
            llc.path.display()
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RocminfoSummary {
    runtime_version: Option<String>,
    arch: Option<String>,
}

fn inspect_rocminfo(path: &Path) -> Result<RocminfoSummary, String> {
    let output = Command::new(path)
        .output()
        .map_err(|err| format!("failed to run {}: {err}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(RocminfoSummary {
        runtime_version: stdout.lines().find_map(|line| {
            let (_, value) = line.split_once("Runtime Version:")?;
            Some(value.trim().to_string())
        }),
        arch: stdout.lines().find_map(|line| {
            let (_, value) = line.split_once("Name:")?;
            let value = value.trim();
            if value.starts_with("gfx") && !value.contains('-') {
                Some(value.to_string())
            } else {
                None
            }
        }),
    })
}

fn report_rocminfo(tool: &ToolPath) -> Result<RocminfoSummary, String> {
    let summary = inspect_rocminfo(&tool.path)?;
    println!(
        "ok: ROCm rocminfo {} [{}] runtime={} detected_arch={}",
        tool.path.display(),
        tool.source,
        summary.runtime_version.as_deref().unwrap_or("<unknown>"),
        summary.arch.as_deref().unwrap_or("<none>")
    );
    Ok(summary)
}

fn report_rocm_agents(tool: &ToolPath, selected_arch: &str) -> Result<(), String> {
    let output = Command::new(&tool.path)
        .output()
        .map_err(|err| format!("failed to run {}: {err}", tool.path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            tool.path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let agents = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("gfx") && !line.contains('-'))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if agents.is_empty() {
        return Err(format!(
            "{} did not report any gfx agents",
            tool.path.display()
        ));
    }
    if !agents.iter().any(|agent| agent == selected_arch) {
        return Err(format!(
            "{} reported agents [{}], but selected arch is {selected_arch}",
            tool.path.display(),
            agents.join(", ")
        ));
    }
    println!(
        "ok: ROCm rocm_agent_enumerator {} [{}] agents={}",
        tool.path.display(),
        tool.source,
        agents.join(", ")
    );
    Ok(())
}

fn ensure_amdgpu_target() -> Result<(), String> {
    rust_target_list().and_then(|targets| {
        if targets.lines().any(|line| line.trim() == TARGET) {
            Ok(())
        } else {
            Err(format!("rustc does not list required target `{TARGET}`"))
        }
    })
}

fn report_amdgpu_target() -> Result<(), String> {
    ensure_amdgpu_target()?;
    println!("ok: rustc target {TARGET}");
    Ok(())
}

fn rust_target_list() -> Result<String, String> {
    let output = Command::new("rustc")
        .args(["--print", "target-list"])
        .output()
        .map_err(|err| format!("failed to query rustc target list: {err}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(format!(
            "rustc --print target-list failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn ensure_rust_src() -> Result<(), String> {
    let path = rust_src_core_path()?;
    if path.is_file() {
        Ok(())
    } else {
        Err(format!(
            "missing Rust source component at {}; install it with `rustup component add rust-src`",
            path.display()
        ))
    }
}

fn report_rust_src() -> Result<(), String> {
    ensure_rust_src()?;
    println!("ok: rust-src component is installed");
    Ok(())
}

fn rust_src_core_path() -> Result<PathBuf, String> {
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .map_err(|err| format!("failed to query rustc sysroot: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustc --print sysroot failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(PathBuf::from(sysroot).join("lib/rustlib/src/rust/library/core/src/lib.rs"))
}

fn report_core_build_probe(arch: &str) -> Result<(), String> {
    let probe_dir = create_core_probe_crate()?;
    let target_dir = probe_dir.join("target");
    let mut command = cargo_command();
    command
        .arg("rustc")
        .arg("-Z")
        .arg("build-std=core")
        .arg("--target")
        .arg(TARGET)
        .arg("--release")
        .arg("--")
        .arg("--emit=llvm-ir")
        .current_dir(&probe_dir)
        .env("RUSTFLAGS", format!("-C target-cpu={arch}"))
        .env("CARGO_TARGET_DIR", &target_dir);
    sanitize_rust_env(&mut command);

    let result = run_command(&mut command, "build `core` for amdgcn-amd-amdhsa")
        .map_err(with_core_build_hint);
    let _ = fs::remove_dir_all(&probe_dir);
    result?;
    println!("ok: `core` builds for {TARGET} with nightly build-std");
    Ok(())
}

fn create_core_probe_crate() -> Result<PathBuf, String> {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock before Unix epoch: {err}"))?
        .as_nanos();
    let root = env::temp_dir().join(format!(
        "rocm-oxide-core-probe-{}-{suffix}",
        std::process::id()
    ));
    fs::create_dir_all(root.join("src"))
        .map_err(|err| format!("failed to create {}: {err}", root.display()))?;
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"rocm-oxide-core-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[lib]\ncrate-type = [\"rlib\"]\n",
    )
    .map_err(|err| format!("failed to write probe Cargo.toml: {err}"))?;
    fs::write(
        root.join("src/lib.rs"),
        "#![no_std]\n#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn rocm_oxide_core_probe() {}\n",
    )
    .map_err(|err| format!("failed to write probe lib.rs: {err}"))?;
    Ok(root)
}

fn cargo_command() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn device_debug_info_enabled() -> bool {
    env_flag_enabled(env::var_os("ROCM_OXIDE_DEVICE_DEBUG").as_deref())
}

fn env_flag_enabled(value: Option<&OsStr>) -> bool {
    let Some(value) = value.and_then(OsStr::to_str) else {
        return false;
    };
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off")
}

fn device_rustflags(arch: &str) -> String {
    format!("-C target-cpu={arch}")
}

fn device_debug_rustc_args(debug_info: bool) -> &'static [&'static str] {
    if debug_info {
        &["-C", "debuginfo=2"]
    } else {
        &[]
    }
}

fn build_device_crate(device_crate: &Path, arch: &str, debug_info: bool) -> Result<(), String> {
    let mut command = cargo_command();
    command
        .arg("rustc")
        .arg("-Z")
        .arg("build-std=core")
        .arg("--target")
        .arg(TARGET)
        .arg("--release")
        .arg("--")
        .arg("--emit=llvm-ir")
        .args(device_debug_rustc_args(debug_info))
        .current_dir(device_crate)
        .env("RUSTFLAGS", device_rustflags(arch));
    sanitize_rust_env(&mut command);
    run_command(&mut command, "compile Rust device crate to AMDGPU LLVM IR")
        .map_err(with_core_build_hint)
}

fn query_rustc_device_layouts(
    device_crate: &Path,
    arch: &str,
    struct_names: &BTreeSet<String>,
) -> Result<BTreeMap<String, DeviceStructLayout>, String> {
    if struct_names.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut command = cargo_command();
    command
        .arg("rustc")
        .arg("-Z")
        .arg("build-std=core")
        .arg("--target")
        .arg(TARGET)
        .arg("--release")
        .arg("--")
        .arg("-Zprint-type-sizes")
        .arg("--emit=metadata")
        .arg("--cfg")
        .arg(format!(
            "rocm_oxide_layout_query_{}",
            unique_build_suffix()?
        ))
        .current_dir(device_crate)
        .env("RUSTFLAGS", format!("-C target-cpu={arch}"));
    sanitize_rust_env(&mut command);
    let output = command
        .output()
        .map_err(|err| format!("failed to query rustc AMDGPU struct layouts: {err}"))?;
    if !output.status.success() {
        return Err(with_core_build_hint(format!(
            "failed to query rustc AMDGPU struct layouts\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(parse_rustc_type_size_layouts(&text, struct_names))
}

fn unique_build_suffix() -> Result<u128, String> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock before Unix epoch: {err}"))?
        .as_nanos())
}

fn parse_rustc_type_size_layouts(
    text: &str,
    struct_names: &BTreeSet<String>,
) -> BTreeMap<String, DeviceStructLayout> {
    let mut layouts = BTreeMap::new();
    let mut current: Option<(String, DeviceStructLayout, u32)> = None;

    for line in text.lines() {
        if let Some((type_name, size, align)) = parse_rustc_type_header(line) {
            if let Some((name, mut layout, offset)) = current.take() {
                finish_rustc_layout(&mut layout, offset);
                layouts.insert(name, layout);
            }
            if let Some(name) = struct_names
                .iter()
                .find(|name| rustc_type_name_matches(&type_name, name))
            {
                current = Some((
                    name.clone(),
                    DeviceStructLayout {
                        size,
                        align,
                        fields: Vec::new(),
                        padding: Vec::new(),
                    },
                    0,
                ));
            }
            continue;
        }

        let Some((_, layout, offset)) = current.as_mut() else {
            continue;
        };
        if let Some((field, size)) = parse_rustc_type_field(line) {
            layout.fields.push(DeviceStructLayoutField {
                name: field,
                ty: String::new(),
                offset: *offset,
                size,
            });
            *offset = offset.saturating_add(size);
        } else if let Some(size) = parse_rustc_type_padding(line) {
            layout.padding.push(DeviceStructPadding {
                offset: *offset,
                size,
            });
            *offset = offset.saturating_add(size);
        }
    }

    if let Some((name, mut layout, offset)) = current.take() {
        finish_rustc_layout(&mut layout, offset);
        layouts.insert(name, layout);
    }

    layouts
}

fn finish_rustc_layout(layout: &mut DeviceStructLayout, offset: u32) {
    if layout.size > offset {
        layout.padding.push(DeviceStructPadding {
            offset,
            size: layout.size - offset,
        });
    }
}

fn parse_rustc_type_header(line: &str) -> Option<(String, u32, u32)> {
    let rest = line.strip_prefix("print-type-size type: `")?;
    let (name, rest) = rest.split_once("`: ")?;
    let (size, rest) = rest.split_once(" bytes, alignment: ")?;
    let align = rest.strip_suffix(" bytes")?;
    Some((name.to_string(), size.parse().ok()?, align.parse().ok()?))
}

fn parse_rustc_type_field(line: &str) -> Option<(String, u32)> {
    let rest = line.strip_prefix("print-type-size     field `.")?;
    let (name, rest) = rest.split_once("`: ")?;
    let size = parse_rustc_byte_count(rest)?;
    Some((name.to_string(), size))
}

fn parse_rustc_type_padding(line: &str) -> Option<u32> {
    let rest = line.strip_prefix("print-type-size     padding: ")?;
    parse_rustc_byte_count(rest)
}

fn parse_rustc_byte_count(value: &str) -> Option<u32> {
    let byte_count = value.split_once(" bytes")?.0;
    byte_count.parse().ok()
}

fn rustc_type_name_matches(rustc_name: &str, struct_name: &str) -> bool {
    rustc_name == struct_name || rustc_name.ends_with(&format!("::{struct_name}"))
}

fn validate_rustc_layout(
    device_struct: &DeviceStruct,
    layout: &mut DeviceStructLayout,
) -> Result<(), String> {
    let fields = device_struct
        .fields
        .iter()
        .map(|field| (field.name.as_str(), field.ty.as_str()))
        .collect::<BTreeMap<_, _>>();
    for layout_field in &mut layout.fields {
        let Some(ty) = fields.get(layout_field.name.as_str()) else {
            return Err(format!(
                "{}: rustc reported unexpected field `{}` for device struct `{}`",
                device_struct.span, layout_field.name, device_struct.name
            ));
        };
        layout_field_type_supported(ty, device_struct)?;
        layout_field.ty = (*ty).to_string();
    }
    for field in &device_struct.fields {
        if !layout
            .fields
            .iter()
            .any(|layout_field| layout_field.name == field.name)
        {
            return Err(format!(
                "{}: rustc layout facts did not include field `{}` for device struct `{}`",
                device_struct.span, field.name, device_struct.name
            ));
        }
    }
    Ok(())
}

fn layout_field_type_supported(
    ty: &str,
    device_struct: &DeviceStruct,
) -> Result<(), String> {
    device_type_layout(ty).map(|_| ()).ok_or_else(|| {
        format!(
            "{}: unsupported field type `{ty}` in device struct `{}`; pass this payload through a DeviceSlice or add explicit layout support before using it by value",
            device_struct.span, device_struct.name
        )
    })
}

fn sanitize_rust_env(command: &mut Command) {
    command
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTC")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTDOC");
}

fn with_core_build_hint(err: String) -> String {
    if err.contains("can't find crate for `core`")
        || err.contains("build-std")
        || err.contains("the option `Z` is only accepted")
        || err.contains("rust-src")
    {
        format!(
            "{err}\n\nhint: ROCm-Oxide device compilation must build `core` for `{TARGET}` with nightly Rust and the `rust-src` component. This repo pins nightly in rust-toolchain.toml; run `rustup component add rust-src` if doctor reports it missing, then retry `cargo rocm-oxide doctor`."
        )
    } else {
        err
    }
}

fn discover_device_crate_bundle(root_crate: &Path) -> Result<Vec<PathBuf>, String> {
    let mut discovered = Vec::new();
    let mut seen = BTreeSet::new();
    discover_device_crate_bundle_inner(root_crate, true, &mut seen, &mut discovered)?;
    Ok(discovered)
}

fn discover_device_crate_bundle_inner(
    crate_path: &Path,
    include_even_without_kernels: bool,
    seen: &mut BTreeSet<PathBuf>,
    discovered: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let canonical = crate_path
        .canonicalize()
        .map_err(|err| format!("failed to canonicalize {}: {err}", crate_path.display()))?;
    if !seen.insert(canonical.clone()) {
        return Ok(());
    }

    if include_even_without_kernels || crate_contains_kernel_attribute(&canonical)? {
        discovered.push(canonical.clone());
    }

    for dependency in path_dependencies(&canonical)? {
        discover_device_crate_bundle_inner(&dependency, false, seen, discovered)?;
    }
    Ok(())
}

fn crate_contains_kernel_attribute(crate_path: &Path) -> Result<bool, String> {
    let src = crate_path.join("src");
    if !src.is_dir() {
        return Ok(false);
    }
    contains_kernel_attribute_in_dir(&src)
}

fn contains_kernel_attribute_in_dir(dir: &Path) -> Result<bool, String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            if contains_kernel_attribute_in_dir(&path)? {
                return Ok(true);
            }
        } else if path.extension() == Some(OsStr::new("rs")) {
            let source = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            if source.lines().any(|line| is_kernel_attribute(line.trim())) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn path_dependencies(crate_path: &Path) -> Result<Vec<PathBuf>, String> {
    let manifest = crate_path.join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .map_err(|err| format!("failed to read {}: {err}", manifest.display()))?;
    let mut deps = Vec::new();
    for line in text.lines() {
        let Some(path) = parse_inline_path_dependency(line) else {
            continue;
        };
        let dependency = crate_path.join(path);
        if dependency.join("Cargo.toml").is_file() {
            deps.push(dependency);
        }
    }
    Ok(deps)
}

fn parse_inline_path_dependency(line: &str) -> Option<PathBuf> {
    let line = line.trim();
    if line.starts_with('#') || !line.contains("path") {
        return None;
    }
    let path_pos = line.find("path")?;
    let after_path = &line[path_pos + "path".len()..];
    let (_, value) = after_path.split_once('=')?;
    let value = value.trim();
    let value = value.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(PathBuf::from(&value[..end]))
}

fn package_name(device_crate: &Path) -> Result<String, String> {
    let manifest = device_crate.join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .map_err(|err| format!("failed to read {}: {err}", manifest.display()))?;
    parse_package_name(&text)
        .ok_or_else(|| format!("failed to find [package] name in {}", manifest.display()))
}

fn parse_package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if in_package && line.starts_with("name") {
            let (_, value) = line.split_once('=')?;
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

fn newest_llvm_ir(deps_dir: &Path, package_name: &str) -> Result<PathBuf, String> {
    let stem = package_name.replace('-', "_");
    let entries = fs::read_dir(deps_dir)
        .map_err(|err| format!("failed to read {}: {err}", deps_dir.display()))?;
    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if path.extension() == Some(OsStr::new("ll"))
            && !name.ends_with(".kernel.ll")
            && name.starts_with(&format!("{stem}-"))
        {
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .map_err(|err| format!("failed to stat {}: {err}", path.display()))?;
            candidates.push((modified, path));
        }
    }

    candidates
        .into_iter()
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
        .ok_or_else(|| {
            format!(
                "no rustc-emitted .ll file for package `{package_name}` found in {}",
                deps_dir.display()
            )
        })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KernelDecl {
    name: String,
    args: Vec<KernelArg>,
    contracts: Vec<BufferContract>,
    span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceSpan {
    path: PathBuf,
    line: usize,
    signature: String,
}

impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}: {}",
            self.path.display(),
            self.line,
            self.signature.trim()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KernelArg {
    name: String,
    ty: String,
    kind: ArgKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceStruct {
    name: String,
    repr: DeviceStructRepr,
    fields: Vec<DeviceStructField>,
    layout: DeviceStructLayout,
    layout_source: DeviceStructLayoutSource,
    span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceStructField {
    name: String,
    ty: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceStructRepr {
    C,
    Rust,
}

impl DeviceStructRepr {
    fn as_str(self) -> &'static str {
        match self {
            DeviceStructRepr::C => "C",
            DeviceStructRepr::Rust => "Rust",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceStructLayout {
    size: u32,
    align: u32,
    fields: Vec<DeviceStructLayoutField>,
    padding: Vec<DeviceStructPadding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceStructLayoutField {
    name: String,
    ty: String,
    offset: u32,
    size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceStructPadding {
    offset: u32,
    size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceStructLayoutSource {
    Computed,
    Rustc,
}

impl DeviceStructLayoutSource {
    fn as_str(self) -> &'static str {
        match self {
            DeviceStructLayoutSource::Computed => "computed",
            DeviceStructLayoutSource::Rustc => "rustc-amdgpu",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceGlobal {
    name: String,
    ty: String,
    mutable: bool,
    kind: DeviceGlobalKind,
    span: SourceSpan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceGlobalKind {
    Global,
    Constant,
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArgKind {
    MutPtr(String),
    ConstPtr(String),
    MutSlice(String),
    ConstSlice(String),
    Scalar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BufferContract {
    buffer: String,
    required_len: LenExpr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LenExpr {
    source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LinkInput {
    package_name: String,
    llvm_ir: PathBuf,
    object: PathBuf,
    kernels: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CodeObjectMetadata {
    kernels: BTreeMap<String, KernelObjectMetadata>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct KernelObjectMetadata {
    kernarg_segment_size: Option<u32>,
    kernarg_segment_align: Option<u32>,
    max_flat_workgroup_size: Option<u32>,
    group_segment_fixed_size: Option<u32>,
    private_segment_fixed_size: Option<u32>,
    sgpr_count: Option<u32>,
    vgpr_count: Option<u32>,
    sgpr_spill_count: Option<u32>,
    vgpr_spill_count: Option<u32>,
    wavefront_size: Option<u32>,
    uses_dynamic_shared_mem: bool,
    uses_dynamic_stack: Option<bool>,
    args: BTreeMap<String, KernelArgObjectMetadata>,
}

impl KernelObjectMetadata {
    fn uses_dynamic_shared_mem(&self) -> bool {
        self.uses_dynamic_shared_mem
            || self
                .args
                .values()
                .any(|arg| arg.value_kind.as_deref() == Some("dynamic_shared_pointer"))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct KernelArgObjectMetadata {
    address_space: Option<String>,
    offset: Option<u32>,
    size: Option<u32>,
    value_kind: Option<String>,
}

fn discover_kernels(device_crate: &Path) -> Result<BTreeMap<String, KernelDecl>, String> {
    let src_dir = device_crate.join("src");
    let mut kernels = BTreeMap::new();
    discover_kernels_in_dir(&src_dir, &mut kernels)?;
    Ok(kernels)
}

fn discover_kernels_in_dir(
    dir: &Path,
    kernels: &mut BTreeMap<String, KernelDecl>,
) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            discover_kernels_in_dir(&path, kernels)?;
        } else if path.extension() == Some(OsStr::new("rs")) {
            let source = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            for kernel in discover_kernels_in_source_at(&source, &path)? {
                if kernels
                    .insert(kernel.name.clone(), kernel.clone())
                    .is_some()
                {
                    return Err(format!("duplicate #[kernel] function: {}", kernel.name));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn discover_kernels_in_source(source: &str) -> Result<Vec<KernelDecl>, String> {
    discover_kernels_in_source_at(source, Path::new("<source>"))
}

fn discover_kernels_in_source_at(source: &str, path: &Path) -> Result<Vec<KernelDecl>, String> {
    let mut kernels = Vec::new();
    let mut expect_function = false;
    let mut signature = String::new();
    let mut pending_contracts = Vec::new();
    let mut pending_monomorphizations = Vec::new();
    let mut signature_start_line = 0usize;

    for (line_index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(contract) = parse_contract_comment(trimmed)? {
            pending_contracts.push(contract);
            continue;
        }

        if let Some(monomorphizations) = parse_kernel_attribute(trimmed)? {
            expect_function = true;
            pending_monomorphizations = monomorphizations;
            signature_start_line = line_index + 1;
            continue;
        }

        if expect_function {
            signature.push(' ');
            signature.push_str(trimmed);

            if trimmed.contains('{') {
                let span = SourceSpan {
                    path: path.to_path_buf(),
                    line: signature_start_line,
                    signature: signature.clone(),
                };
                let mut parsed_kernels =
                    parse_kernel_decls(&signature, span, &pending_monomorphizations)?;
                let contracts = std::mem::take(&mut pending_contracts);
                for kernel in &mut parsed_kernels {
                    kernel.contracts = contracts.clone();
                    validate_contracts(kernel)?;
                }
                kernels.extend(parsed_kernels);
                signature.clear();
                pending_monomorphizations.clear();
                expect_function = false;
            } else if !trimmed.starts_with("#[") && !trimmed.is_empty() {
                continue;
            }
        } else if !trimmed.starts_with("#[") && !trimmed.is_empty() {
            pending_contracts.clear();
        }
    }

    Ok(kernels)
}

fn discover_device_structs(device_crate: &Path) -> Result<BTreeMap<String, DeviceStruct>, String> {
    let src_dir = device_crate.join("src");
    let mut structs = BTreeMap::new();
    discover_device_structs_in_dir(&src_dir, &mut structs)?;
    Ok(structs)
}

fn used_device_struct_names<'a>(
    kernels: impl Iterator<Item = &'a KernelDecl> + Clone,
    struct_names: impl Iterator<Item = &'a String>,
) -> BTreeSet<String> {
    struct_names
        .filter(|name| kernel_bundle_uses_type(kernels.clone(), name))
        .cloned()
        .collect()
}

fn kernel_bundle_uses_type<'a>(
    kernels: impl Iterator<Item = &'a KernelDecl>,
    type_name: &str,
) -> bool {
    kernels
        .flat_map(|kernel| kernel.args.iter())
        .any(|arg| kernel_arg_uses_type(arg, type_name))
}

fn kernel_arg_uses_type(arg: &KernelArg, type_name: &str) -> bool {
    match &arg.kind {
        ArgKind::MutPtr(inner)
        | ArgKind::ConstPtr(inner)
        | ArgKind::MutSlice(inner)
        | ArgKind::ConstSlice(inner) => type_leaf_name(inner) == type_name,
        ArgKind::Scalar => type_leaf_name(&arg.ty) == type_name,
    }
}

fn type_leaf_name(ty: &str) -> &str {
    ty.trim()
        .rsplit("::")
        .next()
        .unwrap_or(ty)
        .trim()
}

fn discover_device_structs_in_dir(
    dir: &Path,
    structs: &mut BTreeMap<String, DeviceStruct>,
) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            discover_device_structs_in_dir(&path, structs)?;
        } else if path.extension() == Some(OsStr::new("rs")) {
            let source = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            for device_struct in discover_device_structs_in_source_at(&source, &path)? {
                structs.insert(device_struct.name.clone(), device_struct);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn discover_device_structs_in_source(source: &str) -> Result<Vec<DeviceStruct>, String> {
    discover_device_structs_in_source_at(source, Path::new("<source>"))
}

fn discover_device_structs_in_source_at(
    source: &str,
    path: &Path,
) -> Result<Vec<DeviceStruct>, String> {
    let mut structs = Vec::new();
    let mut pending_repr = None;
    let mut pending_unsupported_repr: Option<(usize, String)> = None;
    let mut struct_source = String::new();
    let mut struct_start_line = 0usize;
    let mut struct_repr = DeviceStructRepr::Rust;

    for (line_index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(repr) = device_struct_repr_attribute(trimmed) {
            match repr {
                "C" => pending_repr = Some(DeviceStructRepr::C),
                "Rust" => pending_repr = Some(DeviceStructRepr::Rust),
                _ => pending_unsupported_repr = Some((line_index + 1, repr.to_string())),
            }
            struct_start_line = line_index + 1;
            continue;
        }
        if trimmed.starts_with("#[") {
            if pending_repr.is_none() {
                struct_start_line = line_index + 1;
            }
            continue;
        }
        if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
            if let Some((line, repr)) = pending_unsupported_repr.take() {
                return Err(format!(
                    "{}:{}: unsupported repr({repr}) on device struct; generated bindings currently support repr(C), repr(Rust), and default Rust layout",
                    path.display(),
                    line
                ));
            }
            struct_repr = pending_repr.take().unwrap_or(DeviceStructRepr::Rust);
            if struct_start_line == 0 {
                struct_start_line = line_index + 1;
            }
            struct_source.push_str(trimmed);
            if trimmed.contains('}') {
                structs.push(parse_device_struct(
                    &struct_source,
                    struct_repr,
                    SourceSpan {
                        path: path.to_path_buf(),
                        line: struct_start_line,
                        signature: struct_source.clone(),
                    },
                )?);
                struct_source.clear();
                struct_start_line = 0;
            }
            continue;
        }
        if !struct_source.is_empty() {
            struct_source.push(' ');
            struct_source.push_str(trimmed);
            if trimmed.contains('}') {
                structs.push(parse_device_struct(
                    &struct_source,
                    struct_repr,
                    SourceSpan {
                        path: path.to_path_buf(),
                        line: struct_start_line,
                        signature: struct_source.clone(),
                    },
                )?);
                struct_source.clear();
                struct_start_line = 0;
            }
            continue;
        }
        if !trimmed.is_empty() {
            pending_repr = None;
            pending_unsupported_repr = None;
            struct_start_line = 0;
        }
    }

    Ok(structs)
}

fn device_struct_repr_attribute(line: &str) -> Option<&str> {
    let inner = line.strip_prefix("#[")?.strip_suffix(']')?.trim();
    Some(inner.strip_prefix("repr(")?.strip_suffix(')')?.trim())
}

fn discover_device_globals(device_crate: &Path) -> Result<BTreeMap<String, DeviceGlobal>, String> {
    let src_dir = device_crate.join("src");
    let mut globals = BTreeMap::new();
    discover_device_globals_in_dir(&src_dir, &mut globals)?;
    Ok(globals)
}

fn discover_device_globals_in_dir(
    dir: &Path,
    globals: &mut BTreeMap<String, DeviceGlobal>,
) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            discover_device_globals_in_dir(&path, globals)?;
        } else if path.extension() == Some(OsStr::new("rs")) {
            let source = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            for global in discover_device_globals_in_source_at(&source, &path)? {
                globals.insert(global.name.clone(), global);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn discover_device_globals_in_source(source: &str) -> Result<Vec<DeviceGlobal>, String> {
    discover_device_globals_in_source_at(source, Path::new("<source>"))
}

fn discover_device_globals_in_source_at(
    source: &str,
    path: &Path,
) -> Result<Vec<DeviceGlobal>, String> {
    let mut globals = Vec::new();
    let mut pending_kind = None;
    let mut static_source = String::new();
    let mut static_start_line = 0usize;

    for (line_index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(kind) = device_global_attribute_kind(trimmed) {
            pending_kind = Some(kind);
            static_start_line = line_index + 1;
            continue;
        }

        if let Some(kind) = pending_kind {
            static_source.push(' ');
            static_source.push_str(trimmed);
            if trimmed.ends_with(';') {
                globals.push(parse_device_global(
                    &static_source,
                    kind,
                    SourceSpan {
                        path: path.to_path_buf(),
                        line: static_start_line,
                        signature: static_source.clone(),
                    },
                )?);
                static_source.clear();
                pending_kind = None;
            } else if !trimmed.starts_with("#[") && !trimmed.is_empty() {
                continue;
            }
        }
    }

    Ok(globals)
}

fn device_global_attribute_kind(line: &str) -> Option<DeviceGlobalKind> {
    match line {
        "#[device_global]"
        | "#[rocm_oxide_kernel::device_global]"
        | "#[::rocm_oxide_kernel::device_global]" => Some(DeviceGlobalKind::Global),
        "#[constant]" | "#[rocm_oxide_kernel::constant]" | "#[::rocm_oxide_kernel::constant]" => {
            Some(DeviceGlobalKind::Constant)
        }
        "#[shared]" | "#[rocm_oxide_kernel::shared]" | "#[::rocm_oxide_kernel::shared]" => {
            Some(DeviceGlobalKind::Shared)
        }
        _ => None,
    }
}

fn parse_device_global(
    source: &str,
    kind: DeviceGlobalKind,
    span: SourceSpan,
) -> Result<DeviceGlobal, String> {
    let static_pos = source
        .find("static ")
        .ok_or_else(|| format!("{}: marked device global must be a static item", span))?
        + "static ".len();
    let rest = source[static_pos..].trim_start();
    let (mutable, rest) = if let Some(rest) = rest.strip_prefix("mut ") {
        (true, rest.trim_start())
    } else {
        (false, rest)
    };
    let name_end = rest
        .find(':')
        .ok_or_else(|| format!("{}: marked device global is missing a type", span))?;
    let name = rest[..name_end].trim();
    if !is_identifier(name) {
        return Err(format!("{}: invalid device global name `{name}`", span));
    }
    let ty_start = name_end + 1;
    let ty_end = rest[ty_start..]
        .find('=')
        .ok_or_else(|| format!("{}: marked device global is missing an initializer", span))?
        + ty_start;
    let ty = rest[ty_start..ty_end].trim();
    if ty.is_empty() {
        return Err(format!("{}: marked device global is missing a type", span));
    }
    Ok(DeviceGlobal {
        name: name.to_string(),
        ty: ty.to_string(),
        mutable,
        kind,
        span,
    })
}

fn parse_device_struct(
    source: &str,
    repr: DeviceStructRepr,
    span: SourceSpan,
) -> Result<DeviceStruct, String> {
    let struct_pos = source
        .find("struct ")
        .ok_or_else(|| format!("{}: malformed device struct", span))?
        + "struct ".len();
    let name_end = source[struct_pos..]
        .find(|ch: char| ch == '{' || ch.is_whitespace())
        .ok_or_else(|| format!("{}: malformed device struct name", span))?
        + struct_pos;
    let name = source[struct_pos..name_end].trim().to_string();
    if name.contains('<') {
        return Err(format!(
            "{}: generic device structs are not supported in generated host bindings",
            span
        ));
    }
    let body_start = source
        .find('{')
        .ok_or_else(|| format!("{}: missing struct body", span))?
        + 1;
    let body_end = source
        .rfind('}')
        .ok_or_else(|| format!("{}: missing struct body terminator", span))?;
    let body = &source[body_start..body_end];
    let fields = body
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .map(|field| {
            let field = field.strip_prefix("pub ").unwrap_or(field).trim();
            let (name, ty) = field
                .split_once(':')
                .ok_or_else(|| format!("{}: malformed field `{field}`", span))?;
            Ok(DeviceStructField {
                name: name.trim().to_string(),
                ty: ty.trim().to_string(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let layout = compute_device_struct_layout(&name, &fields, &span)?;
    Ok(DeviceStruct {
        name,
        repr,
        fields,
        layout,
        layout_source: DeviceStructLayoutSource::Computed,
        span,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TypeLayout {
    size: u32,
    align: u32,
}

fn compute_device_struct_layout(
    struct_name: &str,
    fields: &[DeviceStructField],
    span: &SourceSpan,
) -> Result<DeviceStructLayout, String> {
    let mut offset = 0u32;
    let mut max_align = 1u32;
    let mut layout_fields = Vec::new();
    let mut padding = Vec::new();

    for field in fields {
        let field_layout = device_type_layout(&field.ty).ok_or_else(|| {
            format!(
                "{}: unsupported field type `{}` in device struct `{struct_name}`; supported layout fields are raw pointers, scalar integers/floats/bool, and fixed-size arrays of those types",
                span, field.ty
            )
        })?;
        max_align = max_align.max(field_layout.align);
        let aligned_offset = align_up_u32(offset, field_layout.align)?;
        if aligned_offset > offset {
            padding.push(DeviceStructPadding {
                offset,
                size: aligned_offset - offset,
            });
        }
        offset = aligned_offset;
        layout_fields.push(DeviceStructLayoutField {
            name: field.name.clone(),
            ty: field.ty.clone(),
            offset,
            size: field_layout.size,
        });
        offset = offset
            .checked_add(field_layout.size)
            .ok_or_else(|| format!("{}: device struct `{struct_name}` layout overflowed", span))?;
    }

    let size = align_up_u32(offset, max_align)?;
    if size > offset {
        padding.push(DeviceStructPadding {
            offset,
            size: size - offset,
        });
    }

    Ok(DeviceStructLayout {
        size,
        align: max_align,
        fields: layout_fields,
        padding,
    })
}

fn device_type_layout(ty: &str) -> Option<TypeLayout> {
    let ty = ty.trim();
    if let Some((inner, len)) = parse_array_type(ty) {
        let inner = device_type_layout(inner)?;
        let len = len.parse::<u32>().ok()?;
        let size = inner.size.checked_mul(len)?;
        return Some(TypeLayout {
            size,
            align: inner.align,
        });
    }
    if is_raw_pointer_type(ty) {
        return Some(TypeLayout { size: 8, align: 8 });
    }
    match ty {
        "bool" | "u8" | "i8" => Some(TypeLayout { size: 1, align: 1 }),
        "u16" | "i16" => Some(TypeLayout { size: 2, align: 2 }),
        "u32" | "i32" | "f32" => Some(TypeLayout { size: 4, align: 4 }),
        "usize" | "isize" | "u64" | "i64" | "f64" => Some(TypeLayout { size: 8, align: 8 }),
        _ => None,
    }
}

fn parse_array_type(ty: &str) -> Option<(&str, &str)> {
    let inner = ty.strip_prefix('[')?.strip_suffix(']')?;
    let (element, len) = inner.rsplit_once(';')?;
    Some((element.trim(), len.trim()))
}

fn is_raw_pointer_type(ty: &str) -> bool {
    let ty = ty.trim();
    ty.strip_prefix("*const ").is_some() || ty.strip_prefix("*mut ").is_some()
}

fn align_up_u32(value: u32, align: u32) -> Result<u32, String> {
    if align == 0 || !align.is_power_of_two() {
        return Err(format!("invalid ABI alignment {align}"));
    }
    let addend = align - 1;
    value
        .checked_add(addend)
        .map(|value| value & !addend)
        .ok_or_else(|| "ABI layout overflowed while aligning a device struct".to_string())
}

fn is_kernel_attribute(line: &str) -> bool {
    parse_kernel_attribute(line).is_ok_and(|value| value.is_some())
}

fn parse_kernel_attribute(line: &str) -> Result<Option<Vec<Vec<String>>>, String> {
    let Some(inner) = line
        .strip_prefix("#[")
        .and_then(|rest| rest.strip_suffix(']'))
    else {
        return Ok(None);
    };
    let inner = inner.trim();
    for name in [
        "kernel",
        "rocm_oxide_kernel::kernel",
        "::rocm_oxide_kernel::kernel",
    ] {
        if inner == name {
            return Ok(Some(Vec::new()));
        }
        if let Some(rest) = inner.strip_prefix(name) {
            let rest = rest.trim_start();
            if rest.starts_with('(') {
                let close = find_matching_delimiter(rest, 0, '(', ')')?;
                if rest[close + 1..].trim().is_empty() {
                    return Ok(Some(parse_kernel_monomorphizations(&rest[1..close])?));
                }
            }
        }
    }
    Ok(None)
}

fn parse_kernel_monomorphizations(source: &str) -> Result<Vec<Vec<String>>, String> {
    let mut rest = source.trim();
    if rest.is_empty() {
        return Ok(Vec::new());
    }

    let mut monomorphizations = Vec::new();
    while !rest.is_empty() {
        let Some(after_keyword) = rest.strip_prefix("monomorphize") else {
            return Err(format!(
                "unsupported #[kernel] argument `{rest}`; expected monomorphize(...)"
            ));
        };
        let after_keyword = after_keyword.trim_start();
        if !after_keyword.starts_with('(') {
            return Err("expected monomorphize(...) in #[kernel]".to_string());
        }
        let close = find_matching_delimiter(after_keyword, 0, '(', ')')?;
        let concrete_types = split_top_level(&after_keyword[1..close], ',')
            .into_iter()
            .map(|ty| ty.trim().to_string())
            .filter(|ty| !ty.is_empty())
            .collect::<Vec<_>>();
        if concrete_types.is_empty() {
            return Err("monomorphize(...) must include at least one type".to_string());
        }
        monomorphizations.push(concrete_types);
        rest = after_keyword[close + 1..].trim_start();
        if let Some(next) = rest.strip_prefix(',') {
            rest = next.trim_start();
        } else if !rest.is_empty() {
            return Err(format!("unexpected #[kernel] argument tail `{rest}`"));
        }
    }
    Ok(monomorphizations)
}

fn parse_kernel_decls(
    signature: &str,
    span: SourceSpan,
    monomorphizations: &[Vec<String>],
) -> Result<Vec<KernelDecl>, String> {
    let fn_pos = signature
        .find("fn ")
        .ok_or_else(|| format!("{}: malformed #[kernel] signature", span))?
        + 3;
    let name_start = fn_pos;
    let name_ident_end = signature[name_start..]
        .find(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
        .ok_or_else(|| format!("{}: malformed #[kernel] signature", span))?
        + name_start;
    let mut cursor = skip_ws(signature, name_ident_end);
    let name_end = if signature[cursor..].starts_with('<') {
        cursor = find_matching_delimiter(signature, cursor, '<', '>')? + 1;
        skip_ws(signature, cursor)
    } else {
        cursor
    };
    let raw_name = signature[name_start..name_end].trim();
    let (name, generic_params) = parse_kernel_name(raw_name, &span)?;
    if name.is_empty() {
        return Err(format!(
            "{}: missing function name in #[kernel] signature",
            span
        ));
    }

    if !generic_params.is_empty() && monomorphizations.is_empty() {
        return Err(format!(
            "{}: generic #[kernel] functions require #[kernel(monomorphize(Ty, ...))] so rocm-oxide-build can emit concrete AMDGPU entry points",
            span
        ));
    }
    if generic_params.is_empty() && !monomorphizations.is_empty() {
        return Err(format!(
            "{}: #[kernel(monomorphize(...))] requires a generic function",
            span
        ));
    }

    if !signature[name_end..].starts_with('(') {
        return Err(format!("{}: malformed #[kernel] argument list", span));
    }
    let args_start = name_end + 1;
    let args_end = signature[args_start..]
        .find(')')
        .ok_or_else(|| format!("{}: malformed #[kernel] argument list", span))?
        + args_start;
    let raw_args = split_top_level(&signature[args_start..args_end], ',')
        .into_iter()
        .map(str::trim)
        .filter(|arg| !arg.is_empty())
        .collect::<Vec<_>>();

    if generic_params.is_empty() {
        let args = raw_args
            .into_iter()
            .map(parse_kernel_arg)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(vec![KernelDecl {
            name,
            args,
            contracts: Vec::new(),
            span,
        }]);
    }

    monomorphizations
        .iter()
        .map(|concrete_types| {
            if concrete_types.len() != generic_params.len() {
                return Err(format!(
                    "{}: kernel `{}` expects {} generic argument(s), but monomorphize(...) supplied {}",
                    span,
                    name,
                    generic_params.len(),
                    concrete_types.len()
                ));
            }
            let args = raw_args
                .iter()
                .copied()
                .map(|arg| parse_kernel_arg_with_types(arg, &generic_params, concrete_types))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(KernelDecl {
                name: monomorphized_kernel_name(&name, concrete_types),
                args,
                contracts: Vec::new(),
                span: span.clone(),
            })
        })
        .collect()
}

fn parse_kernel_name(raw_name: &str, span: &SourceSpan) -> Result<(String, Vec<String>), String> {
    let raw_name = raw_name.trim();
    if let Some(generic_start) = raw_name.find('<') {
        let generic_end = raw_name
            .rfind('>')
            .ok_or_else(|| format!("{}: malformed generic #[kernel] signature", span))?;
        let name = raw_name[..generic_start].trim().to_string();
        let generic_params = parse_generic_params(&raw_name[generic_start + 1..generic_end])?;
        Ok((name, generic_params))
    } else {
        Ok((raw_name.to_string(), Vec::new()))
    }
}

fn parse_kernel_arg(arg: &str) -> Result<KernelArg, String> {
    let (name, ty) = arg
        .split_once(':')
        .ok_or_else(|| format!("malformed kernel argument: {arg}"))?;
    let name = name.trim().to_string();
    let ty = ty.trim().to_string();
    kernel_arg_from_parts(name, ty)
}

fn parse_kernel_arg_with_types(
    arg: &str,
    generic_params: &[String],
    concrete_types: &[String],
) -> Result<KernelArg, String> {
    let (name, ty) = arg
        .split_once(':')
        .ok_or_else(|| format!("malformed kernel argument: {arg}"))?;
    kernel_arg_from_parts(
        name.trim().to_string(),
        substitute_generic_types(ty.trim(), generic_params, concrete_types),
    )
}

fn kernel_arg_from_parts(name: String, ty: String) -> Result<KernelArg, String> {
    if name.is_empty() || ty.is_empty() {
        return Err("malformed kernel argument".to_string());
    }

    let kind = if let Some(inner) = ty.strip_prefix("*mut ") {
        ArgKind::MutPtr(inner.trim().to_string())
    } else if let Some(inner) = ty.strip_prefix("*const ") {
        ArgKind::ConstPtr(inner.trim().to_string())
    } else if let Some(inner) = parse_device_slice_ty(&ty, "DeviceSliceMut") {
        ArgKind::MutSlice(inner)
    } else if let Some(inner) = parse_device_slice_ty(&ty, "DeviceSlice") {
        ArgKind::ConstSlice(inner)
    } else {
        ArgKind::Scalar
    };

    Ok(KernelArg { name, ty, kind })
}

fn parse_device_slice_ty(ty: &str, slice_name: &str) -> Option<String> {
    let normalized = ty
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let prefixes = [
        format!("{slice_name}<"),
        format!("gpu::{slice_name}<"),
        format!("rocm_oxide_device::{slice_name}<"),
        format!("::rocm_oxide_device::{slice_name}<"),
    ];
    prefixes.iter().find_map(|prefix| {
        normalized
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
            .filter(|inner| !inner.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn parse_contract_comment(line: &str) -> Result<Option<BufferContract>, String> {
    let Some(rest) = line.strip_prefix("// rocm-oxide:") else {
        return Ok(None);
    };
    let rest = rest.trim();
    let Some(rest) = rest.strip_prefix("len(") else {
        return Err(format!("unsupported rocm-oxide contract: {line}"));
    };
    let (buffer, rest) = rest
        .split_once(")=")
        .ok_or_else(|| format!("malformed rocm-oxide length contract: {line}"))?;
    let buffer = buffer.trim();
    if !is_identifier(buffer) {
        return Err(format!(
            "invalid buffer name in rocm-oxide contract: {line}"
        ));
    }

    let expr = rest.trim();
    validate_len_expr(expr)?;
    Ok(Some(BufferContract {
        buffer: buffer.to_string(),
        required_len: LenExpr {
            source: expr.to_string(),
        },
    }))
}

fn validate_contracts(kernel: &KernelDecl) -> Result<(), String> {
    let buffer_args = kernel
        .args
        .iter()
        .filter(|arg| arg.kind.is_buffer())
        .map(|arg| arg.name.as_str())
        .collect::<BTreeSet<_>>();
    let scalar_args = kernel
        .args
        .iter()
        .filter(|arg| matches!(arg.kind, ArgKind::Scalar))
        .map(|arg| arg.name.as_str())
        .collect::<BTreeSet<_>>();

    let mut seen = BTreeSet::new();
    for contract in &kernel.contracts {
        if !seen.insert(contract.buffer.as_str()) {
            return Err(format!(
                "duplicate length contract for `{}` in kernel `{}`",
                contract.buffer, kernel.name
            ));
        }
        if !buffer_args.contains(contract.buffer.as_str()) {
            return Err(format!(
                "length contract for `{}` in kernel `{}` does not match a buffer argument",
                contract.buffer, kernel.name
            ));
        }
        for ident in contract.required_len.identifiers() {
            if !scalar_args.contains(ident.as_str()) {
                return Err(format!(
                    "length contract for `{}` in kernel `{}` references non-scalar `{ident}`",
                    contract.buffer, kernel.name
                ));
            }
        }
    }

    Ok(())
}

fn validate_len_expr(expr: &str) -> Result<(), String> {
    let tokens = tokenize_len_expr(expr)?;
    if tokens.is_empty() {
        return Err("empty length contract expression".to_string());
    }
    let mut expect_value = true;
    for token in tokens {
        if expect_value {
            if !is_identifier(&token) && !token.chars().all(|ch| ch.is_ascii_digit()) {
                return Err(format!("invalid length expression token `{token}`"));
            }
        } else if token != "*" && token != "/" && token != "+" && token != "-" {
            return Err(format!("invalid length expression operator `{token}`"));
        }
        expect_value = !expect_value;
    }
    if expect_value {
        return Err(format!("length expression ends with an operator: {expr}"));
    }
    Ok(())
}

fn tokenize_len_expr(expr: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in expr.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if matches!(ch, '*' | '/' | '+' | '-') {
            if current.is_empty() {
                return Err(format!("operator `{ch}` without left operand in `{expr}`"));
            }
            tokens.push(std::mem::take(&mut current));
            tokens.push(ch.to_string());
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            return Err(format!(
                "unsupported character `{ch}` in length expression `{expr}`"
            ));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Ok(tokens)
}

fn parse_generic_params(source: &str) -> Result<Vec<String>, String> {
    split_top_level(source, ',')
        .into_iter()
        .filter(|param| !param.trim().is_empty())
        .map(|param| {
            let trimmed = param.trim();
            if trimmed.starts_with('\'') || trimmed.starts_with("const ") {
                return Err(format!(
                    "unsupported generic kernel parameter `{trimmed}`; only type parameters are supported"
                ));
            }
            let name = trimmed
                .split(|ch: char| ch == ':' || ch == '=' || ch.is_whitespace())
                .next()
                .unwrap_or("")
                .trim();
            if is_identifier(name) {
                Ok(name.to_string())
            } else {
                Err(format!("unsupported generic kernel parameter `{trimmed}`"))
            }
        })
        .collect()
}

fn monomorphized_kernel_name(base: &str, concrete_types: &[String]) -> String {
    let suffix = concrete_types
        .iter()
        .map(|ty| sanitize_type_suffix(ty))
        .collect::<Vec<_>>()
        .join("_");
    format!("{base}_{suffix}")
}

fn sanitize_type_suffix(ty: &str) -> String {
    let mut out = String::new();
    let mut previous_underscore = false;
    for ch in ty.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            previous_underscore = false;
        } else if !previous_underscore {
            out.push('_');
            previous_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn substitute_generic_types(
    source: &str,
    generic_params: &[String],
    concrete_types: &[String],
) -> String {
    let mut output = String::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch == '_' || ch.is_ascii_alphabetic() {
            let mut end = start + ch.len_utf8();
            while let Some((next_index, next)) = chars.peek().copied() {
                if next == '_' || next.is_ascii_alphanumeric() {
                    chars.next();
                    end = next_index + next.len_utf8();
                } else {
                    break;
                }
            }
            let ident = &source[start..end];
            if let Some(index) = generic_params.iter().position(|param| param == ident) {
                output.push_str(&concrete_types[index]);
            } else {
                output.push_str(ident);
            }
        } else {
            output.push(ch);
        }
    }
    output
}

fn skip_ws(source: &str, mut index: usize) -> usize {
    while source[index..]
        .chars()
        .next()
        .is_some_and(|ch| ch.is_whitespace())
    {
        index += source[index..].chars().next().unwrap().len_utf8();
    }
    index
}

fn find_matching_delimiter(
    source: &str,
    open_index: usize,
    open: char,
    close: char,
) -> Result<usize, String> {
    let mut depth = 0usize;
    for (index, ch) in source[open_index..].char_indices() {
        let absolute = open_index + index;
        if ch == open {
            depth += 1;
        } else if ch == close {
            if close == '>' && source[..absolute].ends_with('-') {
                continue;
            }
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Ok(absolute);
            }
        }
    }
    Err(format!("missing matching `{close}`"))
}

fn split_top_level(source: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut paren = 0usize;
    let mut angle = 0usize;
    let mut bracket = 0usize;
    for (index, ch) in source.char_indices() {
        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '<' => angle += 1,
            '>' => angle = angle.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            _ if ch == delimiter && paren == 0 && angle == 0 && bracket == 0 => {
                parts.push(&source[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&source[start..]);
    parts
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn strip_rocm_llc_unsupported_debug_metadata(ir: &str) -> String {
    if !ir.contains("dwarfAddressSpace:") {
        return ir.to_string();
    }
    let mut output = String::with_capacity(ir.len());
    for line in ir.lines() {
        output.push_str(&strip_dwarf_address_space_field(line));
        output.push('\n');
    }
    if !ir.ends_with('\n') {
        output.pop();
    }
    output
}

fn strip_dwarf_address_space_field(line: &str) -> String {
    const FIELD: &str = ", dwarfAddressSpace:";
    let Some(_) = line.find(FIELD) else {
        return line.to_string();
    };

    let mut remaining = line;
    let mut output = String::with_capacity(line.len());
    while let Some(pos) = remaining.find(FIELD) {
        output.push_str(&remaining[..pos]);
        let after_field = &remaining[pos + FIELD.len()..];
        let after_space = after_field.trim_start();
        let consumed_space = after_field.len() - after_space.len();
        let digits_len = after_space
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .map(char::len_utf8)
            .sum::<usize>();
        if digits_len == 0 {
            output.push_str(&remaining[pos..]);
            return output;
        }
        remaining = &after_field[consumed_space + digits_len..];
    }
    output.push_str(remaining);
    output
}

fn transform_ir(
    source: &str,
    kernel_names: &BTreeSet<String>,
    kernels: &BTreeMap<String, KernelDecl>,
    device_globals: &BTreeMap<String, DeviceGlobal>,
) -> Result<String, String> {
    let mut output = Vec::new();
    let mut lines = source.lines().peekable();
    let mut found_kernels = Vec::new();

    while let Some(line) = lines.next() {
        if let Some(kernel) = KernelSignature::parse(line, kernel_names)? {
            found_kernels.push(kernel.name.clone());
            let mut body = vec![kernel.rewritten_signature()];
            for line in lines.by_ref() {
                body.push(line.to_string());
                if line == "}" {
                    break;
                }
            }
            let declaration = kernels
                .get(&kernel.name)
                .ok_or_else(|| format!("kernel `{}` missing from source map", kernel.name))?;
            output.extend(rewrite_kernel_body(
                body,
                declaration,
                kernel.pointer_addrspaces,
                device_globals,
            )?);
        } else {
            if atomic_scope_marker(line).is_some() || is_atomic_scope_marker_declaration(line) {
                continue;
            }
            output.push(rewrite_kernel_attributes(&rewrite_marked_globals(
                line,
                device_globals,
            )));
        }
    }

    if found_kernels.is_empty() {
        return Err(format!(
            "none of the marked kernels were found in LLVM IR:\n{}",
            kernel_names
                .iter()
                .map(|name| {
                    kernels
                        .get(name)
                        .map(|kernel| format!("  - {name} declared at {}", kernel.span))
                        .unwrap_or_else(|| format!("  - {name}"))
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    let found = found_kernels.iter().cloned().collect::<BTreeSet<_>>();
    let missing = kernel_names.difference(&found).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "marked kernels missing from LLVM IR:\n{}",
            missing
                .into_iter()
                .map(|name| {
                    kernels
                        .get(&name)
                        .map(|kernel| format!("  - {name} declared at {}", kernel.span))
                        .unwrap_or_else(|| format!("  - {name}"))
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    Ok(output.join("\n") + "\n")
}

#[derive(Debug)]
struct KernelSignature {
    name: String,
    rewritten: String,
    pointer_addrspaces: BTreeMap<String, String>,
}

impl KernelSignature {
    fn parse(line: &str, kernel_names: &BTreeSet<String>) -> Result<Option<Self>, String> {
        if !line.starts_with("define void @") {
            return Ok(None);
        }

        let name_start = line
            .find('@')
            .ok_or_else(|| format!("malformed kernel signature: {line}"))?
            + 1;
        let name_end = line[name_start..]
            .find('(')
            .ok_or_else(|| format!("malformed kernel signature: {line}"))?
            + name_start;
        let name = line[name_start..name_end].to_string();
        if !kernel_names.contains(&name) {
            return Ok(None);
        }

        let args_end = line
            .rfind(')')
            .ok_or_else(|| format!("malformed kernel signature: {line}"))?;
        let args = &line[name_end + 1..args_end];
        let mut pointer_addrspaces = BTreeMap::new();
        let rewritten_args = split_args(args)
            .into_iter()
            .map(|arg| {
                let trimmed = arg.trim();
                if trimmed.starts_with("ptr ") {
                    if let Some(name) = argument_name(trimmed) {
                        pointer_addrspaces.insert(name, "1".to_string());
                    }
                    trimmed.replacen("ptr ", "ptr addrspace(1) ", 1)
                } else {
                    trimmed.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        let suffix = line[args_end + 1..].replace(" unnamed_addr", " local_unnamed_addr");
        let rewritten =
            format!("define protected amdgpu_kernel void @{name}({rewritten_args}){suffix}");

        Ok(Some(Self {
            name,
            rewritten,
            pointer_addrspaces,
        }))
    }

    fn rewritten_signature(&self) -> String {
        self.rewritten.clone()
    }
}

fn split_args(args: &str) -> Vec<&str> {
    if args.trim().is_empty() {
        Vec::new()
    } else {
        args.split(',').collect()
    }
}

fn argument_name(arg: &str) -> Option<String> {
    arg.split_whitespace()
        .last()
        .and_then(|token| token.strip_prefix('%'))
        .map(ToOwned::to_owned)
}

fn rewrite_kernel_body(
    lines: Vec<String>,
    kernel: &KernelDecl,
    mut pointer_addrspaces: BTreeMap<String, String>,
    device_globals: &BTreeMap<String, DeviceGlobal>,
) -> Result<Vec<String>, String> {
    let mut rewritten = Vec::with_capacity(lines.len());
    let mut pending_atomic_scope = None;

    for line in lines {
        let mut current = rewrite_marked_globals(&line, device_globals);
        if let Some(op) = unsupported_pointer_integer_cast(&current) {
            return Err(format!(
                "{}: unsupported pointer/integer cast `{op}` in kernel `{}`; ROCm-Oxide cannot prove address-space-preserving semantics for this cast yet\n  LLVM IR: {}",
                kernel.span,
                kernel.name,
                current.trim()
            ));
        }
        for (name, address_space) in &pointer_addrspaces {
            current = rewrite_pointer_operand(&current, name, address_space);
        }
        if let Some(scope) = atomic_scope_marker(&current) {
            pending_atomic_scope = Some(scope);
            continue;
        }
        if current.contains(" phi ptr ")
            && let Some(address_space) = pointer_addrspaces
                .iter()
                .find_map(|(name, space)| contains_ssa_value(&current, name).then(|| space.clone()))
        {
            current = current.replacen(
                " phi ptr ",
                &format!(" phi ptr addrspace({address_space}) "),
                1,
            );
        }
        if let Some(address_space) = pointer_addrspaces
            .values()
            .find(|space| current.contains(&format!(" load ptr, ptr addrspace({space})")))
        {
            current = current.replacen(
                " load ptr,",
                &format!(" load ptr addrspace({address_space}),"),
                1,
            );
        }

        if produces_address_space_pointer(&current)
            && let Some(result) = assigned_value(&current)
            && let Some(address_space) = pointer_address_space(&current)
            && matches!(address_space.as_str(), "1" | "3" | "4")
        {
            pointer_addrspaces.insert(result, address_space);
        }
        if let Some(scope) = pending_atomic_scope
            && is_atomic_instruction(&current)
        {
            current = rewrite_atomic_syncscope(&current, scope);
            pending_atomic_scope = None;
        }

        rewritten.push(rewrite_kernel_attributes(&current));
    }

    Ok(rewritten)
}

fn unsupported_pointer_integer_cast(line: &str) -> Option<&'static str> {
    if line.contains(" inttoptr ") || line.contains("= inttoptr ") {
        Some("inttoptr")
    } else if line.contains(" ptrtoint ") || line.contains("= ptrtoint ") {
        Some("ptrtoint")
    } else {
        None
    }
}

#[derive(Clone, Copy)]
enum AtomicSyncScope {
    Workgroup,
    Agent,
    System,
}

impl AtomicSyncScope {
    fn llvm_name(self) -> Option<&'static str> {
        match self {
            Self::Workgroup => Some("workgroup"),
            Self::Agent => Some("agent"),
            Self::System => None,
        }
    }
}

fn atomic_scope_marker(line: &str) -> Option<AtomicSyncScope> {
    if line.contains("@__rocm_oxide_atomic_scope_workgroup(") {
        Some(AtomicSyncScope::Workgroup)
    } else if line.contains("@__rocm_oxide_atomic_scope_device(") {
        Some(AtomicSyncScope::Agent)
    } else if line.contains("@__rocm_oxide_atomic_scope_system(") {
        Some(AtomicSyncScope::System)
    } else {
        None
    }
}

fn is_atomic_scope_marker_declaration(line: &str) -> bool {
    line.trim_start().starts_with("declare ") && line.contains("@__rocm_oxide_atomic_scope_")
}

fn is_atomic_instruction(line: &str) -> bool {
    line.contains(" atomicrmw ")
        || line.trim_start().starts_with("atomicrmw ")
        || line.contains(" load atomic ")
        || line.trim_start().starts_with("load atomic ")
        || line.contains(" store atomic ")
        || line.trim_start().starts_with("store atomic ")
        || line.contains(" cmpxchg ")
        || line.trim_start().starts_with("cmpxchg ")
}

fn rewrite_atomic_syncscope(line: &str, scope: AtomicSyncScope) -> String {
    if line.contains(" syncscope(") {
        return line.to_string();
    }
    let Some(scope_name) = scope.llvm_name() else {
        return line.to_string();
    };
    let orderings = [
        "unordered",
        "monotonic",
        "acquire",
        "release",
        "acq_rel",
        "seq_cst",
    ];

    if line.contains(" cmpxchg ") || line.trim_start().starts_with("cmpxchg ") {
        for ordering in orderings {
            let needle = format!(" {ordering} ");
            if let Some(pos) = line.find(&needle) {
                let mut rewritten = line.to_string();
                rewritten.insert_str(pos, &format!(" syncscope(\"{scope_name}\")"));
                return rewritten;
            }
        }
    }

    for ordering in orderings {
        for needle in [format!(" {ordering},"), format!(" {ordering} ")] {
            if let Some(pos) = line.find(&needle) {
                let mut rewritten = line.to_string();
                rewritten.insert_str(pos, &format!(" syncscope(\"{scope_name}\")"));
                return rewritten;
            }
        }
        if let Some(prefix) = line.strip_suffix(ordering)
            && prefix.ends_with(' ')
        {
            let pos = prefix.len() - 1;
            let mut rewritten = line.to_string();
            rewritten.insert_str(pos, &format!(" syncscope(\"{scope_name}\")"));
            return rewritten;
        }
    }

    line.to_string()
}

fn rewrite_pointer_operand(line: &str, name: &str, address_space: &str) -> String {
    let replacement = format!("ptr addrspace({address_space}) %{name}");
    let mut rewritten =
        rewrite_pointer_operand_with_needle(line, &format!("ptr %{name}"), &replacement);
    for existing_address_space in ["0", "1", "3", "4", "5"] {
        rewritten = rewrite_pointer_operand_with_needle(
            &rewritten,
            &format!("ptr addrspace({existing_address_space}) %{name}"),
            &replacement,
        );
    }
    rewritten
}

fn rewrite_pointer_operand_with_needle(line: &str, needle: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(line.len() + replacement.len());
    let mut cursor = 0usize;

    while let Some(relative) = line[cursor..].find(needle) {
        let start = cursor + relative;
        let end = start + needle.len();
        output.push_str(&line[cursor..start]);
        if line[end..]
            .chars()
            .next()
            .is_some_and(is_llvm_name_continue)
        {
            output.push_str(&line[start..end]);
        } else {
            output.push_str(replacement);
        }
        cursor = end;
    }

    output.push_str(&line[cursor..]);
    output
}

fn contains_ssa_value(line: &str, name: &str) -> bool {
    let needle = format!("%{name}");
    let mut cursor = 0usize;
    while let Some(relative) = line[cursor..].find(&needle) {
        let start = cursor + relative;
        let before = line[..start].chars().next_back();
        let after = line[start + needle.len()..].chars().next();
        let starts_at_boundary = before.is_none_or(|ch| !is_llvm_name_continue(ch));
        let ends_at_boundary = after.is_none_or(|ch| !is_llvm_name_continue(ch));
        if starts_at_boundary && ends_at_boundary {
            return true;
        }
        cursor = start + needle.len();
    }
    false
}

fn is_llvm_name_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-')
}

fn rewrite_marked_globals(line: &str, device_globals: &BTreeMap<String, DeviceGlobal>) -> String {
    let mut rewritten = line.to_string();
    for global in device_globals.values() {
        rewritten = rewrite_marked_global_definition(&rewritten, global);
        rewritten = rewritten.replace(
            &format!("ptr @{}", global.name),
            &format!(
                "ptr addrspace({}) @{}",
                global.kind.address_space(),
                global.name
            ),
        );
        for address_space in ["0", "1", "3", "4", "5"] {
            rewritten = rewritten.replace(
                &format!("ptr addrspace({address_space}) @{}", global.name),
                &format!(
                    "ptr addrspace({}) @{}",
                    global.kind.address_space(),
                    global.name
                ),
            );
        }
    }
    rewritten
}

fn rewrite_marked_global_definition(line: &str, global: &DeviceGlobal) -> String {
    let trimmed = line.trim_start();
    if !trimmed.starts_with(&format!("@{} =", global.name)) {
        return line.to_string();
    }
    if let Some(start) = line.find("addrspace(")
        && let Some(relative_end) = line[start..].find(')')
    {
        let end = start + relative_end + 1;
        let mut rewritten = line.to_string();
        rewritten.replace_range(
            start..end,
            &format!("addrspace({})", global.kind.address_space()),
        );
        return rewrite_shared_global_initializer(&rewritten, global);
    }
    for keyword in [" global ", " constant "] {
        if let Some(pos) = line.find(keyword) {
            let mut rewritten = line.to_string();
            rewritten.insert_str(
                pos + 1,
                &format!("addrspace({}) ", global.kind.address_space()),
            );
            return rewrite_shared_global_initializer(&rewritten, global);
        }
    }
    rewrite_shared_global_initializer(line, global)
}

fn rewrite_shared_global_initializer(line: &str, global: &DeviceGlobal) -> String {
    if global.kind != DeviceGlobalKind::Shared {
        return line.to_string();
    }
    line.replace(" zeroinitializer", " undef")
}

fn produces_address_space_pointer(line: &str) -> bool {
    line.contains("ptr addrspace(")
        && (line.contains("getelementptr")
            || line.contains("addrspacecast")
            || line.contains("bitcast")
            || line.contains("select ")
            || line.contains(" phi ")
            || line.contains(" load ptr"))
}

fn pointer_address_space(line: &str) -> Option<String> {
    let (_, rest) = line.split_once("ptr addrspace(")?;
    let (address_space, _) = rest.split_once(')')?;
    address_space
        .chars()
        .all(|ch| ch.is_ascii_digit())
        .then(|| address_space.to_string())
}

fn assigned_value(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('%')?;
    let (name, _) = rest.split_once(" = ")?;
    Some(name.to_string())
}

fn rewrite_kernel_attributes(line: &str) -> String {
    let mut line = strip_target_memory_effects(&line.replace(" nocreateundeforpoison", ""));
    if !line.starts_with("attributes #")
        || !line.contains("\"target-cpu\"=")
    {
        return line;
    }
    if !line.contains("\"amdgpu-flat-work-group-size\"=") {
        line = line.replacen(
            "\"target-cpu\"=",
            "\"amdgpu-flat-work-group-size\"=\"1,1024\" \"target-cpu\"=",
            1,
        );
    }
    if !line.contains("\"amdgpu-no-hostcall-ptr\"") {
        line = line.replacen(
            "\"target-cpu\"=",
            "\"amdgpu-no-hostcall-ptr\" \"target-cpu\"=",
            1,
        );
    }
    line
}

fn strip_target_memory_effects(line: &str) -> String {
    let mut current = line.to_string();
    while let Some(start) = current.find(", target_mem") {
        let rest = &current[start + 2..];
        let Some(end_relative) = rest.find([',', ')']) else {
            break;
        };
        let end = start + 2 + end_relative;
        current.replace_range(start..end, "");
    }
    current
}

fn validate_code_object(
    hsaco: &Path,
    expected_kernels: &BTreeSet<String>,
    llvm_readelf: &Path,
) -> Result<(), String> {
    let output = Command::new(llvm_readelf)
        .arg("-s")
        .arg(hsaco)
        .output()
        .map_err(|err| format!("failed to run llvm-readelf: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "llvm-readelf failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut kernels = BTreeSet::new();
    let mut descriptors = BTreeSet::new();
    for line in stdout.lines() {
        let is_function = line.contains(" FUNC ");
        let is_object = line.contains(" OBJECT ");
        for token in line.split_whitespace() {
            if let Some(name) = token.strip_suffix(".kd") {
                if is_object && expected_kernels.contains(name) {
                    descriptors.insert(name.to_string());
                }
            } else if is_function && expected_kernels.contains(token) {
                kernels.insert(token.to_string());
            }
        }
    }

    let missing_functions = expected_kernels
        .difference(&kernels)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_functions.is_empty() {
        return Err(format!(
            "linked code object is missing kernel functions for: {}",
            missing_functions.join(", ")
        ));
    }

    let missing = expected_kernels
        .difference(&descriptors)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "linked code object is missing kernel descriptors for: {}",
            missing.join(", ")
        ));
    }

    Ok(())
}

fn read_code_object_metadata(
    hsaco: &Path,
    llvm_readelf: &Path,
) -> Result<CodeObjectMetadata, String> {
    let output = Command::new(llvm_readelf)
        .arg("-n")
        .arg(hsaco)
        .output()
        .map_err(|err| format!("failed to run llvm-readelf -n: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "llvm-readelf -n failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_code_object_metadata(&String::from_utf8_lossy(&output.stdout))
}

fn validate_code_object_metadata(
    metadata: &CodeObjectMetadata,
    expected_kernels: &BTreeSet<String>,
) -> Result<(), String> {
    let missing = expected_kernels
        .iter()
        .filter(|name| !metadata.kernels.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "linked code object metadata is missing kernel resource rows for: {}",
            missing.join(", ")
        ))
    }
}

fn annotate_dynamic_shared_mem_from_ir(
    metadata: &mut CodeObjectMetadata,
    kernel_ir: &Path,
) -> Result<(), String> {
    let text = fs::read_to_string(kernel_ir)
        .map_err(|err| format!("failed to read {}: {err}", kernel_ir.display()))?;
    let dynamic_symbols = dynamic_shared_symbols(&text);
    if dynamic_symbols.is_empty() {
        return Ok(());
    }

    let mut current_kernel: Option<String> = None;
    for line in text.lines() {
        if let Some(name) = amdgpu_kernel_name(line) {
            current_kernel = Some(name.to_string());
            continue;
        }
        if current_kernel.is_some() && line.trim() == "}" {
            current_kernel = None;
            continue;
        }
        let Some(kernel_name) = current_kernel.as_deref() else {
            continue;
        };
        if dynamic_symbols
            .iter()
            .any(|symbol| line.contains(&format!("@{symbol}")))
            && let Some(kernel) = metadata.kernels.get_mut(kernel_name)
        {
            kernel.uses_dynamic_shared_mem = true;
        }
    }

    Ok(())
}

fn verify_lds_artifacts(
    kernel_ir: &Path,
    object: &Path,
    llvm_objdump: &Path,
) -> Result<(), String> {
    let ir = fs::read_to_string(kernel_ir)
        .map_err(|err| format!("failed to read {}: {err}", kernel_ir.display()))?;
    verify_lds_ir(&ir)?;

    let output = Command::new(llvm_objdump)
        .arg("-d")
        .arg(object)
        .output()
        .map_err(|err| format!("failed to run {} -d: {err}", llvm_objdump.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} -d {} failed:\n{}",
            llvm_objdump.display(),
            object.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    verify_lds_isa(&String::from_utf8_lossy(&output.stdout))
}

fn verify_lds_ir(ir: &str) -> Result<(), String> {
    if let Some(body) = llvm_function_body(ir, "lds_block_sum") {
        let dynamic_symbols = dynamic_shared_symbols(ir);
        let has_dynamic_symbol = dynamic_symbols.iter().any(|symbol| {
            body.contains(&format!("@{symbol}")) && body.contains("ptr addrspace(3)")
        });
        if dynamic_symbols.is_empty() || !has_dynamic_symbol {
            return Err(format!(
                "lds_block_sum IR did not preserve dynamic LDS in addrspace(3)\n  dynamic symbols: {}\n  body has addrspace(3): {}",
                dynamic_symbols
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
                body.contains("ptr addrspace(3)")
            ));
        }
    }

    if let Some(body) = llvm_function_body(ir, "static_lds_reverse") {
        let definition = ir
            .lines()
            .find(|line| line.trim_start().starts_with("@STATIC_LDS_U32 ="))
            .ok_or_else(|| "static_lds_reverse IR missing STATIC_LDS_U32 definition".to_string())?;
        let has_shared_definition = definition.contains("addrspace(3)")
            && definition.contains(" global ")
            && definition.contains(" undef")
            && !definition.contains("zeroinitializer");
        let has_shared_store = body
            .lines()
            .any(|line| line.contains("store ") && line.contains("ptr addrspace(3)"));
        let has_shared_load = body
            .lines()
            .any(|line| line.contains("load ") && line.contains("ptr addrspace(3)"));
        let references_symbol = body.contains("@STATIC_LDS_U32");
        if !has_shared_definition || !references_symbol || !has_shared_store || !has_shared_load {
            return Err(format!(
                "static_lds_reverse IR did not preserve static LDS in addrspace(3)\n  shared definition: {has_shared_definition}\n  references symbol: {references_symbol}\n  addrspace(3) store: {has_shared_store}\n  addrspace(3) load: {has_shared_load}\n  definition: {}",
                definition.trim()
            ));
        }
    }

    Ok(())
}

fn verify_lds_isa(disassembly: &str) -> Result<(), String> {
    for symbol in ["lds_block_sum", "static_lds_reverse"] {
        let Some(body) = disassembly_symbol_body(disassembly, symbol) else {
            continue;
        };
        let has_store = has_lds_store(&body);
        let has_load = has_lds_load(&body);
        if !has_store || !has_load {
            let lds_lines = body
                .lines()
                .filter(|line| line.contains("ds_"))
                .map(str::trim)
                .collect::<Vec<_>>();
            return Err(format!(
                "{symbol} ISA did not contain expected LDS DS load/store instructions\n  LDS store: {has_store}\n  LDS load: {has_load}\n  DS lines:\n{}",
                lds_lines.join("\n")
            ));
        }
    }

    Ok(())
}

fn has_lds_store(body: &str) -> bool {
    body.contains("ds_store") || body.contains("ds_write")
}

fn has_lds_load(body: &str) -> bool {
    body.contains("ds_load") || body.contains("ds_read")
}

fn verify_scoped_atomic_artifacts(
    kernel_ir: &Path,
    object: &Path,
    llvm_objdump: &Path,
) -> Result<(), String> {
    let ir = fs::read_to_string(kernel_ir)
        .map_err(|err| format!("failed to read {}: {err}", kernel_ir.display()))?;
    verify_scoped_atomic_ir(&ir)?;

    let output = Command::new(llvm_objdump)
        .arg("-d")
        .arg(object)
        .output()
        .map_err(|err| format!("failed to run {} -d: {err}", llvm_objdump.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} -d {} failed:\n{}",
            llvm_objdump.display(),
            object.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    verify_scoped_atomic_isa(&String::from_utf8_lossy(&output.stdout))
}

fn verify_scoped_atomic_ir(ir: &str) -> Result<(), String> {
    let body = llvm_function_body(ir, "scoped_atomics")
        .ok_or_else(|| "scoped_atomics kernel missing from transformed LLVM IR".to_string())?;
    if body.contains("__rocm_oxide_atomic_scope_") {
        return Err(
            "scoped_atomics IR still contains internal atomic scope marker calls".to_string(),
        );
    }

    let atomic_lines = body
        .lines()
        .filter(|line| is_atomic_instruction(line))
        .collect::<Vec<_>>();
    let has_workgroup = atomic_lines
        .iter()
        .any(|line| line.contains("syncscope(\"workgroup\")"));
    let has_agent = atomic_lines
        .iter()
        .any(|line| line.contains("syncscope(\"agent\")"));
    let has_system_default = atomic_lines.iter().any(|line| !line.contains("syncscope("));

    if !has_workgroup || !has_agent || !has_system_default {
        return Err(format!(
            "scoped_atomics IR did not preserve expected scope mapping\n  workgroup syncscope: {has_workgroup}\n  agent syncscope: {has_agent}\n  system backend default: {has_system_default}\n  atomic lines:\n{}",
            atomic_lines
                .iter()
                .map(|line| format!("    {}", line.trim()))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    Ok(())
}

fn verify_scoped_atomic_isa(disassembly: &str) -> Result<(), String> {
    let body = disassembly_symbol_body(disassembly, "scoped_atomics")
        .ok_or_else(|| "scoped_atomics symbol missing from object disassembly".to_string())?;
    let atomic_lines = body
        .lines()
        .filter(|line| is_global_atomic_add_isa_line(line))
        .map(str::trim)
        .collect::<Vec<_>>();
    let has_global_atomic = !atomic_lines.is_empty();
    let has_expected_atomic_count = atomic_lines.len() >= 3;
    let has_scope_annotations = atomic_lines.iter().any(|line| line.contains("scope:"));
    let has_workgroup_scope = atomic_lines
        .iter()
        .any(|line| line.contains("scope:SCOPE_SE"));
    let has_device_scope = atomic_lines
        .iter()
        .any(|line| line.contains("scope:SCOPE_DEV"));
    let has_system_scope = atomic_lines
        .iter()
        .any(|line| line.contains("scope:SCOPE_SYS"));

    if !has_global_atomic
        || !has_expected_atomic_count
        || (has_scope_annotations
            && (!has_workgroup_scope || !has_device_scope || !has_system_scope))
    {
        return Err(format!(
            "scoped_atomics ISA did not contain expected AMDGPU atomic scopes\n  global/flat/buffer atomic add: {has_global_atomic}\n  atomic add count: {}\n  scope annotations present: {has_scope_annotations}\n  workgroup/SCOPE_SE: {has_workgroup_scope}\n  device/SCOPE_DEV: {has_device_scope}\n  system/SCOPE_SYS: {has_system_scope}\n  atomic lines:\n{}",
            atomic_lines.len(),
            atomic_lines.join("\n")
        ));
    }

    Ok(())
}

fn is_global_atomic_add_isa_line(line: &str) -> bool {
    line.contains("global_atomic_add_u32")
        || line.contains("flat_atomic_add_u32")
        || line.contains("buffer_atomic_add_u32")
}

fn llvm_function_body(text: &str, name: &str) -> Option<String> {
    let needle = format!("@{name}(");
    let mut body = Vec::new();
    let mut depth = 0i32;
    let mut capturing = false;

    for line in text.lines() {
        if !capturing && line.trim_start().starts_with("define ") && line.contains(&needle) {
            capturing = true;
        }
        if capturing {
            depth += line.chars().filter(|ch| *ch == '{').count() as i32;
            depth -= line.chars().filter(|ch| *ch == '}').count() as i32;
            body.push(line.to_string());
            if depth == 0 && line.trim() == "}" {
                return Some(body.join("\n"));
            }
        }
    }

    None
}

fn disassembly_symbol_body(text: &str, name: &str) -> Option<String> {
    let symbol = format!("<{name}>:");
    let mut body = Vec::new();
    let mut capturing = false;

    for line in text.lines() {
        if !capturing {
            if line.contains(&symbol) {
                capturing = true;
                body.push(line.to_string());
            }
            continue;
        }

        if !body.is_empty() && line.trim().is_empty() {
            break;
        }
        if line.contains(">:") && line.contains('<') && !line.contains(&symbol) {
            break;
        }
        body.push(line.to_string());
    }

    capturing.then(|| body.join("\n"))
}

fn dynamic_shared_symbols(text: &str) -> BTreeSet<String> {
    text.lines()
        .filter(|line| line.contains("external") && line.contains("addrspace(3) global [0 x i8]"))
        .filter_map(|line| line.trim_start().strip_prefix('@'))
        .filter_map(|line| line.split_once(' ').map(|(name, _)| name.to_string()))
        .collect()
}

fn amdgpu_kernel_name(line: &str) -> Option<&str> {
    let (_, rest) = line.split_once("amdgpu_kernel void @")?;
    rest.split_once('(').map(|(name, _)| name)
}

fn parse_code_object_metadata(text: &str) -> Result<CodeObjectMetadata, String> {
    let mut metadata = CodeObjectMetadata::default();
    let mut block = Vec::new();
    for line in text.lines() {
        if line.trim() == "- .args:" {
            parse_kernel_metadata_block(&mut metadata, &block)?;
            block.clear();
        }
        if !block.is_empty() || line.trim() == "- .args:" {
            block.push(line.to_string());
        }
    }
    parse_kernel_metadata_block(&mut metadata, &block)?;
    Ok(metadata)
}

fn parse_kernel_metadata_block(
    metadata: &mut CodeObjectMetadata,
    block: &[String],
) -> Result<(), String> {
    if block.is_empty() {
        return Ok(());
    }
    let Some(name) = block
        .iter()
        .find_map(|line| line.strip_prefix("    .name:").map(clean_metadata_string))
    else {
        return Ok(());
    };

    let mut kernel = KernelObjectMetadata::default();
    let mut pending_arg = KernelArgObjectMetadata::default();
    let mut pending_arg_name: Option<String> = None;
    for line in block {
        let trimmed = line.trim();
        if let Some(field) = line.strip_prefix("      - ") {
            flush_kernel_arg(&mut kernel, &mut pending_arg_name, &mut pending_arg);
            parse_arg_metadata_field(field.trim(), &mut pending_arg, &mut pending_arg_name);
            continue;
        }
        if let Some(field) = line.strip_prefix("        .") {
            parse_arg_metadata_field(
                &format!(".{}", field.trim()),
                &mut pending_arg,
                &mut pending_arg_name,
            );
            continue;
        }

        if let Some(value) = metadata_u32(trimmed, ".kernarg_segment_size:") {
            kernel.kernarg_segment_size = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".kernarg_segment_align:") {
            kernel.kernarg_segment_align = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".max_flat_workgroup_size:") {
            kernel.max_flat_workgroup_size = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".group_segment_fixed_size:") {
            kernel.group_segment_fixed_size = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".private_segment_fixed_size:") {
            kernel.private_segment_fixed_size = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".sgpr_count:") {
            kernel.sgpr_count = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".vgpr_count:") {
            kernel.vgpr_count = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".sgpr_spill_count:") {
            kernel.sgpr_spill_count = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".vgpr_spill_count:") {
            kernel.vgpr_spill_count = Some(value);
        } else if let Some(value) = metadata_u32(trimmed, ".wavefront_size:") {
            kernel.wavefront_size = Some(value);
        } else if let Some(value) = metadata_bool(trimmed, ".uses_dynamic_stack:") {
            kernel.uses_dynamic_stack = Some(value);
        }
    }
    flush_kernel_arg(&mut kernel, &mut pending_arg_name, &mut pending_arg);
    metadata.kernels.insert(name, kernel);
    Ok(())
}

fn parse_arg_metadata_field(
    field: &str,
    arg: &mut KernelArgObjectMetadata,
    arg_name: &mut Option<String>,
) {
    if let Some(value) = metadata_scalar(field, ".address_space:") {
        arg.address_space = Some(clean_metadata_string(value));
    } else if let Some(value) = metadata_scalar(field, ".name:") {
        *arg_name = Some(clean_metadata_string(value));
    } else if let Some(value) = metadata_u32(field, ".offset:") {
        arg.offset = Some(value);
    } else if let Some(value) = metadata_u32(field, ".size:") {
        arg.size = Some(value);
    } else if let Some(value) = metadata_scalar(field, ".value_kind:") {
        arg.value_kind = Some(clean_metadata_string(value));
    }
}

fn flush_kernel_arg(
    kernel: &mut KernelObjectMetadata,
    arg_name: &mut Option<String>,
    arg: &mut KernelArgObjectMetadata,
) {
    if let Some(name) = arg_name.take() {
        kernel.args.insert(name, std::mem::take(arg));
    } else {
        *arg = KernelArgObjectMetadata::default();
    }
}

fn metadata_scalar<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.strip_prefix(key).map(str::trim)
}

fn metadata_u32(line: &str, key: &str) -> Option<u32> {
    metadata_scalar(line, key)?.parse::<u32>().ok()
}

fn metadata_bool(line: &str, key: &str) -> Option<bool> {
    match metadata_scalar(line, key)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn clean_metadata_string(value: &str) -> String {
    value
        .trim()
        .strip_prefix("!str ")
        .unwrap_or(value.trim())
        .trim_matches('"')
        .to_string()
}

fn write_metadata(
    path: &Path,
    arch: &str,
    hsaco: &Path,
    link_inputs: &[LinkInput],
    kernels: &BTreeMap<String, KernelDecl>,
    device_structs: &BTreeMap<String, DeviceStruct>,
    device_globals: &BTreeMap<String, DeviceGlobal>,
    code_object_metadata: &CodeObjectMetadata,
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"target\": \"{}\",\n", json_escape(TARGET)));
    out.push_str(&format!("  \"arch\": \"{}\",\n", json_escape(arch)));
    out.push_str(&format!(
        "  \"hsaco\": \"{}\",\n",
        json_escape(&hsaco.display().to_string())
    ));
    out.push_str("  \"link\": {\n");
    out.push_str("    \"objects\": [\n");
    for (input_index, input) in link_inputs.iter().enumerate() {
        if input_index > 0 {
            out.push_str(",\n");
        }
        out.push_str("      {\n");
        out.push_str(&format!(
            "        \"package\": \"{}\",\n",
            json_escape(&input.package_name)
        ));
        out.push_str(&format!(
            "        \"llvm_ir\": \"{}\",\n",
            json_escape(&input.llvm_ir.display().to_string())
        ));
        out.push_str(&format!(
            "        \"object\": \"{}\",\n",
            json_escape(&input.object.display().to_string())
        ));
        out.push_str("        \"kernels\": [");
        for (kernel_index, kernel) in input.kernels.iter().enumerate() {
            if kernel_index > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\"{}\"", json_escape(kernel)));
        }
        out.push_str("]\n");
        out.push_str("      }");
    }
    out.push_str("\n    ]\n");
    out.push_str("  },\n");
    out.push_str("  \"kernels\": [\n");

    for (kernel_index, kernel) in kernels.values().enumerate() {
        if kernel_index > 0 {
            out.push_str(",\n");
        }
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"name\": \"{}\",\n",
            json_escape(&kernel.name)
        ));
        let object_metadata = code_object_metadata.kernels.get(&kernel.name);
        out.push_str("      \"args\": [\n");
        for (arg_index, arg) in kernel.args.iter().enumerate() {
            if arg_index > 0 {
                out.push_str(",\n");
            }
            out.push_str("        {\n");
            out.push_str(&format!(
                "          \"name\": \"{}\",\n",
                json_escape(&arg.name)
            ));
            out.push_str(&format!(
                "          \"type\": \"{}\",\n",
                json_escape(&arg.ty)
            ));
            out.push_str(&format!("          \"kind\": \"{}\"", arg.kind.as_str()));
            if let Some(object_arg) = object_metadata.and_then(|m| m.args.get(&arg.name)) {
                if let Some(value) = object_arg.address_space.as_deref() {
                    out.push_str(&format!(
                        ",\n          \"address_space\": \"{}\"",
                        json_escape(value)
                    ));
                }
                if let Some(value) = object_arg.offset {
                    out.push_str(&format!(",\n          \"offset\": {value}"));
                }
                if let Some(value) = object_arg.size {
                    out.push_str(&format!(",\n          \"abi_size\": {value}"));
                }
                if let Some(value) = object_arg.value_kind.as_deref() {
                    out.push_str(&format!(
                        ",\n          \"value_kind\": \"{}\"",
                        json_escape(value)
                    ));
                }
            } else {
                out.push_str(&format!(
                    ",\n          \"abi_size\": {}",
                    fallback_abi_size(arg, device_structs)
                ));
                if !matches!(arg.kind, ArgKind::Scalar) {
                    out.push_str(",\n          \"address_space\": \"global\"");
                }
            }
            out.push('\n');
            out.push_str("        }");
        }
        out.push_str("\n      ],\n");
        out.push_str("      \"contracts\": [\n");
        for (contract_index, contract) in kernel.contracts.iter().enumerate() {
            if contract_index > 0 {
                out.push_str(",\n");
            }
            out.push_str("        {\n");
            out.push_str(&format!(
                "          \"buffer\": \"{}\",\n",
                json_escape(&contract.buffer)
            ));
            out.push_str(&format!(
                "          \"required_len\": \"{}\"\n",
                json_escape(&contract.required_len.source)
            ));
            out.push_str("        }");
        }
        out.push_str("\n      ],\n");
        out.push_str("      \"code_object\": ");
        write_kernel_object_metadata(&mut out, object_metadata);
        out.push('\n');
        out.push_str("    }");
    }

    out.push_str("\n  ],\n");
    out.push_str("  \"structs\": [\n");
    for (struct_index, device_struct) in device_structs.values().enumerate() {
        if struct_index > 0 {
            out.push_str(",\n");
        }
        write_device_struct_metadata(&mut out, device_struct);
    }
    out.push_str("\n  ],\n");
    out.push_str("  \"globals\": [\n");
    for (global_index, global) in device_globals.values().enumerate() {
        if global_index > 0 {
            out.push_str(",\n");
        }
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"name\": \"{}\",\n",
            json_escape(&global.name)
        ));
        out.push_str(&format!(
            "      \"type\": \"{}\",\n",
            json_escape(&global.ty)
        ));
        out.push_str(&format!("      \"kind\": \"{}\",\n", global.kind.as_str()));
        out.push_str(&format!(
            "      \"mutable\": {},\n",
            if global.mutable { "true" } else { "false" }
        ));
        out.push_str(&format!(
            "      \"address_space\": \"{}\"\n",
            global.kind.address_space()
        ));
        out.push_str("    }");
    }
    out.push_str("\n  ]\n");
    out.push_str("}\n");
    fs::write(path, out).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn write_device_struct_metadata(out: &mut String, device_struct: &DeviceStruct) {
    out.push_str("    {\n");
    out.push_str(&format!(
        "      \"name\": \"{}\",\n",
        json_escape(&device_struct.name)
    ));
    out.push_str(&format!(
        "      \"repr\": \"{}\",\n",
        device_struct.repr.as_str()
    ));
    out.push_str(&format!(
        "      \"layout_source\": \"{}\",\n",
        device_struct.layout_source.as_str()
    ));
    out.push_str(&format!(
        "      \"abi_size\": {},\n",
        device_struct.layout.size
    ));
    out.push_str(&format!("      \"align\": {},\n", device_struct.layout.align));
    out.push_str("      \"fields\": [\n");
    for (field_index, field) in device_struct.layout.fields.iter().enumerate() {
        if field_index > 0 {
            out.push_str(",\n");
        }
        out.push_str("        {\n");
        out.push_str(&format!(
            "          \"name\": \"{}\",\n",
            json_escape(&field.name)
        ));
        out.push_str(&format!(
            "          \"type\": \"{}\",\n",
            json_escape(&field.ty)
        ));
        out.push_str(&format!("          \"offset\": {},\n", field.offset));
        out.push_str(&format!("          \"size\": {}\n", field.size));
        out.push_str("        }");
    }
    out.push_str("\n      ],\n");
    out.push_str("      \"padding\": [\n");
    for (padding_index, padding) in device_struct.layout.padding.iter().enumerate() {
        if padding_index > 0 {
            out.push_str(",\n");
        }
        out.push_str("        {\n");
        out.push_str(&format!("          \"offset\": {},\n", padding.offset));
        out.push_str(&format!("          \"size\": {}\n", padding.size));
        out.push_str("        }");
    }
    out.push_str("\n      ]\n");
    out.push_str("    }");
}

fn write_kernel_object_metadata(out: &mut String, metadata: Option<&KernelObjectMetadata>) {
    let Some(metadata) = metadata else {
        out.push_str("null");
        return;
    };

    out.push_str("{\n");
    write_json_u32_field(
        out,
        "kernarg_segment_size",
        metadata.kernarg_segment_size,
        true,
    );
    write_json_u32_field(
        out,
        "kernarg_segment_align",
        metadata.kernarg_segment_align,
        false,
    );
    write_json_u32_field(
        out,
        "max_flat_workgroup_size",
        metadata.max_flat_workgroup_size,
        false,
    );
    write_json_u32_field(
        out,
        "group_segment_fixed_size",
        metadata.group_segment_fixed_size,
        false,
    );
    write_json_u32_field(
        out,
        "private_segment_fixed_size",
        metadata.private_segment_fixed_size,
        false,
    );
    write_json_u32_field(out, "sgpr_count", metadata.sgpr_count, false);
    write_json_u32_field(out, "vgpr_count", metadata.vgpr_count, false);
    write_json_u32_field(out, "sgpr_spill_count", metadata.sgpr_spill_count, false);
    write_json_u32_field(out, "vgpr_spill_count", metadata.vgpr_spill_count, false);
    write_json_u32_field(out, "wavefront_size", metadata.wavefront_size, false);
    out.push_str(&format!(
        ",\n        \"uses_dynamic_shared_mem\": {}",
        metadata.uses_dynamic_shared_mem()
    ));
    if let Some(value) = metadata.uses_dynamic_stack {
        out.push_str(&format!(",\n        \"uses_dynamic_stack\": {value}"));
    }
    out.push_str("\n      }");
}

fn write_json_u32_field(out: &mut String, key: &str, value: Option<u32>, first: bool) {
    if let Some(value) = value {
        if !first {
            out.push_str(",\n");
        }
        out.push_str(&format!("        \"{key}\": {value}"));
    }
}

fn fallback_abi_size(arg: &KernelArg, device_structs: &BTreeMap<String, DeviceStruct>) -> u32 {
    match &arg.kind {
        ArgKind::MutPtr(_) | ArgKind::ConstPtr(_) => 8,
        ArgKind::MutSlice(_) | ArgKind::ConstSlice(_) => 16,
        ArgKind::Scalar => primitive_abi_size(&arg.ty)
            .or_else(|| {
                device_structs
                    .get(type_leaf_name(&arg.ty))
                    .map(|device_struct| device_struct.layout.size)
            })
            .unwrap_or(8),
    }
}

fn primitive_abi_size(ty: &str) -> Option<u32> {
    match ty.trim() {
        "usize" | "isize" | "u64" | "i64" | "f64" => Some(8),
        "u32" | "i32" | "f32" => Some(4),
        "u16" | "i16" => Some(2),
        "u8" | "i8" | "bool" => Some(1),
        _ => None,
    }
}

fn write_bindings(
    path: &Path,
    hsaco: &Path,
    kernels: &BTreeMap<String, KernelDecl>,
    device_structs: &BTreeMap<String, DeviceStruct>,
    device_globals: &BTreeMap<String, DeviceGlobal>,
    code_object_metadata: &CodeObjectMetadata,
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("// Generated by rocm-oxide-build. Do not edit by hand.\n");
    out.push_str("use std::path::Path;\n\n");
    let hsaco_file = hsaco
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("invalid hsaco path: {}", hsaco.display()))?;
    let metadata_file = hsaco
        .file_stem()
        .and_then(OsStr::to_str)
        .map(|stem| format!("{stem}.metadata.json"))
        .ok_or_else(|| format!("invalid hsaco path: {}", hsaco.display()))?;
    out.push_str(&format!(
        "pub const DEVICE_HSACO_BYTES: &[u8] = include_bytes!(\"{}\");\n\n",
        hsaco_file
    ));
    out.push_str(&format!(
        "#[allow(dead_code)]\npub const DEVICE_METADATA_JSON: &str = include_str!(\"{}\");\n\n",
        metadata_file
    ));
    out.push_str("pub const DEVICE_KERNEL_RESOURCES: &[rocm_oxide::KernelResource] = &[\n");
    for kernel in kernels.values() {
        out.push_str(&generate_kernel_resource_binding(
            kernel,
            code_object_metadata.kernels.get(&kernel.name),
        ));
    }
    out.push_str("];\n\n");
    for device_struct in device_structs.values() {
        out.push_str(&generate_device_struct_binding(device_struct));
        out.push('\n');
    }
    out.push_str("#[allow(dead_code)]\n");
    out.push_str("pub struct DeviceKernels {\n");
    out.push_str("    module: std::sync::Arc<rocm_oxide::Module>,\n");
    for kernel in kernels.values() {
        out.push_str(&format!(
            "    {}: std::sync::Arc<rocm_oxide::Kernel>,\n",
            kernel_field_name(&kernel.name)
        ));
    }
    out.push_str("}\n\n");
    out.push_str("#[allow(dead_code)]\n");
    out.push_str("impl DeviceKernels {\n");
    out.push_str("    pub fn load(device: &rocm_oxide::Device, hsaco: impl AsRef<Path>) -> rocm_oxide::Result<Self> {\n");
    out.push_str("        Self::from_module(std::sync::Arc::new(device.load_code_object_file(hsaco)?))\n");
    out.push_str("    }\n\n");
    out.push_str(
        "    pub fn load_embedded(device: &rocm_oxide::Device) -> rocm_oxide::Result<Self> {\n",
    );
    out.push_str("        Self::from_module(std::sync::Arc::new(device.load_code_object(DEVICE_HSACO_BYTES)?))\n");
    out.push_str("    }\n\n");
    out.push_str("    fn from_module(module: std::sync::Arc<rocm_oxide::Module>) -> rocm_oxide::Result<Self> {\n");
    out.push_str("        Ok(Self {\n");
    out.push_str("            module: std::sync::Arc::clone(&module),\n");
    for kernel in kernels.values() {
        let kernel_metadata =
            generated_kernel_metadata(code_object_metadata.kernels.get(&kernel.name));
        out.push_str(&format!(
            "            {}: std::sync::Arc::new(module.kernel_with_metadata(c\"{}\", {kernel_metadata})?),\n",
            kernel_field_name(&kernel.name),
            kernel.name
        ));
    }
    out.push_str("        })\n");
    out.push_str("    }\n\n");
    out.push_str("    pub fn module(&self) -> &rocm_oxide::Module {\n");
    out.push_str("        self.module.as_ref()\n");
    out.push_str("    }\n\n");
    out.push_str("    pub fn kernel(&self, name: &str) -> Option<&rocm_oxide::Kernel> {\n");
    out.push_str("        match name {\n");
    for kernel in kernels.values() {
        out.push_str(&format!(
            "            \"{}\" => Some(self.{}.as_ref()),\n",
            kernel.name,
            kernel_field_name(&kernel.name)
        ));
    }
    out.push_str("            _ => None,\n");
    out.push_str("        }\n");
    out.push_str("    }\n\n");
    out.push_str("    pub const fn resources(&self) -> &'static [rocm_oxide::KernelResource] {\n");
    out.push_str("        DEVICE_KERNEL_RESOURCES\n");
    out.push_str("    }\n\n");
    out.push_str(
        "    pub fn resource(&self, name: &str) -> Option<&'static rocm_oxide::KernelResource> {\n",
    );
    out.push_str("        DEVICE_KERNEL_RESOURCES.iter().find(|resource| resource.name == name)\n");
    out.push_str("    }\n\n");
    out.push_str("    pub fn recommend_1d_launch(&self, name: &str, num_elems: usize, dynamic_shared_mem_bytes: u32, block_size_limit: u32) -> rocm_oxide::Result<rocm_oxide::LaunchRecommendation> {\n");
    out.push_str("        let kernel = self.kernel(name).ok_or_else(|| rocm_oxide::Error::InvalidLaunch(format!(\"unknown generated kernel `{name}`\")))?;\n");
    out.push_str("        kernel.recommend_1d_launch(num_elems, dynamic_shared_mem_bytes, block_size_limit)\n");
    out.push_str("    }\n\n");

    for global in device_globals
        .values()
        .filter(|global| global.kind.has_host_binding())
    {
        out.push_str(&generate_device_global_binding(global));
        out.push('\n');
    }

    for kernel in kernels.values() {
        out.push_str(&generate_kernel_binding(
            kernel,
            device_structs,
            code_object_metadata.kernels.get(&kernel.name),
        )?);
        out.push('\n');
    }

    out.push_str("}\n");
    fs::write(path, out).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn generate_device_struct_binding(device_struct: &DeviceStruct) -> String {
    let mut out = String::new();
    if device_struct.repr == DeviceStructRepr::C {
        out.push_str("#[repr(C)]\n");
    }
    out.push_str("#[derive(Clone, Copy, Debug, Default)]\n");
    out.push_str(&format!("pub struct {} {{\n", device_struct.name));
    for field in &device_struct.fields {
        out.push_str(&format!("    pub {}: {},\n", field.name, field.ty));
    }
    out.push_str("}\n");
    out.push_str("const _: () = {\n");
    out.push_str(&format!(
        "    assert!(std::mem::size_of::<{}>() == {});\n",
        device_struct.name, device_struct.layout.size
    ));
    out.push_str(&format!(
        "    assert!(std::mem::align_of::<{}>() == {});\n",
        device_struct.name, device_struct.layout.align
    ));
    for field in &device_struct.layout.fields {
        out.push_str(&format!(
            "    assert!(std::mem::offset_of!({}, {}) == {});\n",
            device_struct.name, field.name, field.offset
        ));
    }
    out.push_str("};\n");
    out
}

fn generate_device_global_binding(global: &DeviceGlobal) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "    pub fn {}(&self) -> rocm_oxide::Result<rocm_oxide::Global<{}>> {{\n",
        device_global_method_name(&global.name),
        global.ty
    ));
    out.push_str(&format!(
        "        self.module.global(c\"{}\")\n",
        global.name
    ));
    out.push_str("    }\n");
    out
}

fn generate_kernel_resource_binding(
    kernel: &KernelDecl,
    metadata: Option<&KernelObjectMetadata>,
) -> String {
    let default = KernelObjectMetadata::default();
    let metadata = metadata.unwrap_or(&default);
    format!(
        "    rocm_oxide::KernelResource {{ name: \"{}\", kernarg_segment_size: {}, kernarg_segment_align: {}, max_flat_workgroup_size: {}, group_segment_fixed_size: {}, private_segment_fixed_size: {}, sgpr_count: {}, vgpr_count: {}, sgpr_spill_count: {}, vgpr_spill_count: {}, wavefront_size: {}, uses_dynamic_shared_mem: {}, uses_dynamic_stack: {} }},\n",
        json_escape(&kernel.name),
        generated_option_u32(metadata.kernarg_segment_size),
        generated_option_u32(metadata.kernarg_segment_align),
        generated_option_u32(metadata.max_flat_workgroup_size),
        generated_option_u32(metadata.group_segment_fixed_size),
        generated_option_u32(metadata.private_segment_fixed_size),
        generated_option_u32(metadata.sgpr_count),
        generated_option_u32(metadata.vgpr_count),
        generated_option_u32(metadata.sgpr_spill_count),
        generated_option_u32(metadata.vgpr_spill_count),
        generated_option_u32(metadata.wavefront_size),
        metadata.uses_dynamic_shared_mem(),
        generated_option_bool(metadata.uses_dynamic_stack),
    )
}

fn device_global_method_name(name: &str) -> String {
    format!("global_{}", to_snake_case(name))
}

fn to_snake_case(name: &str) -> String {
    let mut out = String::new();
    let mut previous_lower_or_digit = false;
    for ch in name.chars() {
        if ch == '_' {
            if !out.ends_with('_') {
                out.push('_');
            }
            previous_lower_or_digit = false;
        } else if ch.is_ascii_uppercase() {
            if previous_lower_or_digit && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = false;
        } else {
            out.push(ch);
            previous_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out.trim_matches('_').to_string()
}

fn generate_kernel_binding(
    kernel: &KernelDecl,
    device_structs: &BTreeMap<String, DeviceStruct>,
    metadata: Option<&KernelObjectMetadata>,
) -> Result<String, String> {
    let mut params = vec!["config: rocm_oxide::LaunchConfig".to_string()];
    let mut operation_params = vec!["config: rocm_oxide::LaunchConfig".to_string()];
    let mut launch_args = Vec::new();
    let mut buffer_arg_names = Vec::new();
    let mut indirect_scalar_buffer_arg_names = Vec::new();
    let mut keep_alive_arg_names = Vec::new();
    let mut operation_supported = true;
    let has_len_arg = kernel
        .args
        .iter()
        .any(|arg| arg.name == "n" && matches!(arg.kind, ArgKind::Scalar));
    let has_block_x_arg = kernel
        .args
        .iter()
        .any(|arg| arg.name == "block_x" && matches!(arg.kind, ArgKind::Scalar));

    for arg in &kernel.args {
        match &arg.kind {
            ArgKind::MutPtr(inner) => {
                params.push(format!(
                    "{}: &rocm_oxide::DeviceBuffer<{}>",
                    arg.name, inner
                ));
                operation_params.push(format!(
                    "{}: std::sync::Arc<rocm_oxide::DeviceBuffer<{}>>",
                    arg.name, inner
                ));
                launch_args.push(format!("{}.as_mut_ptr()", arg.name));
                buffer_arg_names.push((arg.name.clone(), true));
                keep_alive_arg_names.push(arg.name.clone());
            }
            ArgKind::ConstPtr(inner) => {
                params.push(format!(
                    "{}: &rocm_oxide::DeviceBuffer<{}>",
                    arg.name, inner
                ));
                operation_params.push(format!(
                    "{}: std::sync::Arc<rocm_oxide::DeviceBuffer<{}>>",
                    arg.name, inner
                ));
                launch_args.push(format!("{}.as_ptr()", arg.name));
                buffer_arg_names.push((arg.name.clone(), false));
                keep_alive_arg_names.push(arg.name.clone());
            }
            ArgKind::MutSlice(inner) => {
                params.push(format!(
                    "{}: &rocm_oxide::DeviceBuffer<{}>",
                    arg.name, inner
                ));
                operation_params.push(format!(
                    "{}: std::sync::Arc<rocm_oxide::DeviceBuffer<{}>>",
                    arg.name, inner
                ));
                launch_args.push(format!("{}.as_mut_ptr()", arg.name));
                launch_args.push(format!("{}.len()", arg.name));
                buffer_arg_names.push((arg.name.clone(), true));
                keep_alive_arg_names.push(arg.name.clone());
            }
            ArgKind::ConstSlice(inner) => {
                params.push(format!(
                    "{}: &rocm_oxide::DeviceBuffer<{}>",
                    arg.name, inner
                ));
                operation_params.push(format!(
                    "{}: std::sync::Arc<rocm_oxide::DeviceBuffer<{}>>",
                    arg.name, inner
                ));
                launch_args.push(format!("{}.as_ptr()", arg.name));
                launch_args.push(format!("{}.len()", arg.name));
                buffer_arg_names.push((arg.name.clone(), false));
                keep_alive_arg_names.push(arg.name.clone());
            }
            ArgKind::Scalar => {
                if let Some(device_struct) = device_structs.get(type_leaf_name(&arg.ty)) {
                    if scalar_arg_is_indirect_global_buffer(metadata, &arg.name) {
                        params.push(format!(
                            "{}: &rocm_oxide::DeviceBuffer<{}>",
                            arg.name, device_struct.name
                        ));
                        operation_params.push(format!(
                            "{}: std::sync::Arc<rocm_oxide::DeviceBuffer<{}>>",
                            arg.name, device_struct.name
                        ));
                        launch_args.push(format!("{}.as_ptr()", arg.name));
                        buffer_arg_names.push((arg.name.clone(), false));
                        indirect_scalar_buffer_arg_names.push(arg.name.clone());
                        keep_alive_arg_names.push(arg.name.clone());
                    } else {
                        if device_struct
                            .fields
                            .iter()
                            .any(|field| is_raw_pointer_type(&field.ty))
                        {
                            operation_supported = false;
                        }
                        params.push(format!("{}: {}", arg.name, arg.ty));
                        operation_params.push(format!("{}: {}", arg.name, arg.ty));
                        for field in &device_struct.layout.fields {
                            launch_args.push(format!("{}.{}", arg.name, field.name));
                        }
                    }
                } else if primitive_abi_size(&arg.ty).is_some() {
                    params.push(format!("{}: {}", arg.name, arg.ty));
                    operation_params.push(format!("{}: {}", arg.name, arg.ty));
                    launch_args.push(arg.name.clone());
                } else {
                    return Err(format!(
                        "{}: unsupported by-value kernel argument `{}` with type `{}`; use a primitive scalar, a layout-proven device struct, or pass the payload through a DeviceSlice",
                        kernel.span, arg.name, arg.ty
                    ));
                }
            }
        }
    }

    let mut out = String::new();
    let field_name = kernel_field_name(&kernel.name);
    let method_name = kernel_method_name(&kernel.name);
    out.push_str(&format!(
        "    pub unsafe fn {}(&self, {}) -> rocm_oxide::Result<()> {{\n",
        method_name,
        params.join(", ")
    ));
    out.push_str(&generate_kernel_validation_lines(
        kernel,
        &buffer_arg_names,
        &indirect_scalar_buffer_arg_names,
        has_len_arg,
        has_block_x_arg,
        false,
    ));
    out.push_str(&generate_kernel_param_setup(&launch_args, "        "));
    out.push_str("        unsafe {\n");
    out.push_str(&format!(
        "            self.{field_name}.launch_raw(config, &mut __params)\n"
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push('\n');
    let mut stream_params = vec!["stream: &rocm_oxide::Stream".to_string()];
    stream_params.extend(params.clone());
    out.push_str(&format!(
        "    pub unsafe fn {}_on_stream(&self, {}) -> rocm_oxide::Result<()> {{\n",
        method_name,
        stream_params.join(", ")
    ));
    out.push_str(&generate_kernel_validation_lines(
        kernel,
        &buffer_arg_names,
        &indirect_scalar_buffer_arg_names,
        has_len_arg,
        has_block_x_arg,
        false,
    ));
    out.push_str(&generate_kernel_param_setup(&launch_args, "        "));
    out.push_str("        unsafe {\n");
    out.push_str(&format!(
        "            self.{field_name}.launch_raw_on_stream(stream, config, &mut __params)\n"
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push('\n');
    out.push_str("    /// Launches without generated buffer, alias, or launch validation.\n");
    out.push_str("    ///\n");
    out.push_str("    /// # Safety\n");
    out.push_str("    /// The caller must prevalidate the launch config, buffer lengths, aliasing,\n");
    out.push_str("    /// argument ABI, and all pointer lifetimes for the launched work.\n");
    out.push_str(&format!(
        "    pub unsafe fn {}_unchecked(&self, {}) -> rocm_oxide::Result<()> {{\n",
        method_name,
        params.join(", ")
    ));
    out.push_str(&generate_kernel_param_setup(&launch_args, "        "));
    out.push_str("        unsafe {\n");
    out.push_str(&format!(
        "            self.{field_name}.launch_raw_unchecked(config, &mut __params)\n"
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push('\n');
    out.push_str("    /// Launches on a stream without generated buffer, alias, or launch validation.\n");
    out.push_str("    ///\n");
    out.push_str("    /// # Safety\n");
    out.push_str("    /// The caller must prevalidate the launch config, buffer lengths, aliasing,\n");
    out.push_str("    /// argument ABI, pointer lifetimes, and stream/device association.\n");
    out.push_str(&format!(
        "    pub unsafe fn {}_on_stream_unchecked(&self, {}) -> rocm_oxide::Result<()> {{\n",
        method_name,
        stream_params.join(", ")
    ));
    out.push_str(&generate_kernel_param_setup(&launch_args, "        "));
    out.push_str("        unsafe {\n");
    out.push_str(&format!(
        "            self.{field_name}.launch_raw_on_stream_unchecked(stream, config, &mut __params)\n"
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push('\n');
    let mut graph_params = vec![
        "graph: &rocm_oxide::hip::Graph".to_string(),
        "dependencies: &[rocm_oxide::hip::GraphNode]".to_string(),
    ];
    graph_params.extend(params.clone());
    out.push_str("    /// Adds this kernel launch as a node in an explicit HIP graph.\n");
    out.push_str("    ///\n");
    out.push_str("    /// # Safety\n");
    out.push_str("    /// The caller must keep `self`, all buffers, and all argument-owned data\n");
    out.push_str("    /// alive until graph execution using the returned node has completed.\n");
    out.push_str(&format!(
        "    pub unsafe fn {}_graph_node(&self, {}) -> rocm_oxide::Result<rocm_oxide::hip::GraphNode> {{\n",
        method_name,
        graph_params.join(", ")
    ));
    out.push_str(&generate_kernel_validation_lines(
        kernel,
        &buffer_arg_names,
        &indirect_scalar_buffer_arg_names,
        has_len_arg,
        has_block_x_arg,
        false,
    ));
    out.push_str(&generate_kernel_param_setup(&launch_args, "        "));
    out.push_str("        unsafe {\n");
    out.push_str(&format!(
        "            self.{field_name}.add_graph_node_raw(graph, dependencies, config, &mut __params)\n"
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push('\n');
    if operation_supported {
        out.push_str(&format!(
            "    pub unsafe fn {}_operation(&self, {}) -> rocm_oxide::Result<impl rocm_oxide::DeviceOperation<Output = rocm_oxide::KernelLaunchCompletion> + 'static> {{\n",
            method_name,
            operation_params.join(", ")
        ));
        out.push_str(&generate_kernel_validation_lines(
            kernel,
            &buffer_arg_names,
            &indirect_scalar_buffer_arg_names,
            has_len_arg,
            has_block_x_arg,
            true,
        ));
        out.push_str("        let module = std::sync::Arc::clone(&self.module);\n");
        out.push_str(&format!(
            "        let kernel = std::sync::Arc::clone(&self.{field_name});\n"
        ));
        out.push_str(
            "        Ok(move |context: &rocm_oxide::ExecutionContext| -> rocm_oxide::Result<rocm_oxide::KernelLaunchCompletion> {\n",
        );
        out.push_str(&generate_kernel_param_setup(&launch_args, "            "));
        out.push_str("            unsafe {\n");
        out.push_str(
            "                kernel.launch_raw_on_stream(context.stream(), config, &mut __params)?;\n",
        );
        out.push_str("            }\n");
        out.push_str("            let mut __completion = rocm_oxide::KernelLaunchCompletion::new();\n");
        out.push_str("            __completion.keep_alive(module);\n");
        out.push_str("            __completion.keep_alive(kernel);\n");
        for arg_name in &keep_alive_arg_names {
            out.push_str(&format!(
                "            __completion.keep_alive({arg_name});\n"
            ));
        }
        out.push_str("            Ok(__completion)\n");
        out.push_str("        })\n");
        out.push_str("    }\n");
    } else {
        out.push_str(&format!(
            "    // {method_name}_operation is intentionally omitted because by-value raw pointer arguments require caller-managed lifetimes.\n"
        ));
    }
    Ok(out)
}

fn scalar_arg_is_indirect_global_buffer(
    metadata: Option<&KernelObjectMetadata>,
    arg_name: &str,
) -> bool {
    metadata
        .and_then(|metadata| metadata.args.get(arg_name))
        .map(|arg_metadata| {
            arg_metadata.value_kind.as_deref() == Some("global_buffer")
                && arg_metadata.address_space.as_deref() == Some("global")
        })
        .unwrap_or(false)
}

fn kernel_field_name(name: &str) -> String {
    format!("__kernel_{}", to_snake_case(name))
}

fn kernel_method_name(name: &str) -> String {
    to_snake_case(name)
}

fn generate_kernel_param_setup(launch_args: &[String], indent: &str) -> String {
    let mut out = String::new();
    if launch_args.is_empty() {
        out.push_str(indent);
        out.push_str("let mut __params: [*mut std::ffi::c_void; 0] = [];\n");
    } else {
        for (index, arg) in launch_args.iter().enumerate() {
            out.push_str(indent);
            out.push_str(&format!("let mut __arg{index} = {arg};\n"));
        }
        out.push_str(indent);
        out.push_str("let mut __params = [\n");
        for index in 0..launch_args.len() {
            out.push_str(indent);
            out.push_str(&format!(
                "    rocm_oxide::__private::arg_ptr(&mut __arg{index}),\n"
            ));
        }
        out.push_str(indent);
        out.push_str("];\n");
    }
    out
}

fn generated_kernel_metadata(metadata: Option<&KernelObjectMetadata>) -> String {
    let Some(metadata) = metadata else {
        return "rocm_oxide::KernelMetadata::default()".to_string();
    };
    format!(
        "rocm_oxide::KernelMetadata {{ max_flat_workgroup_size: {}, static_shared_mem_bytes: {}, uses_dynamic_shared_mem: {}, wavefront_size: {} }}",
        generated_option_u32(metadata.max_flat_workgroup_size),
        metadata.group_segment_fixed_size.unwrap_or(0),
        metadata.uses_dynamic_shared_mem(),
        generated_option_u32(metadata.wavefront_size),
    )
}

fn generated_option_u32(value: Option<u32>) -> String {
    match value {
        Some(value) => format!("Some({value})"),
        None => "None".to_string(),
    }
}

fn generated_option_bool(value: Option<bool>) -> String {
    match value {
        Some(value) => format!("Some({value})"),
        None => "None".to_string(),
    }
}

fn generate_kernel_validation_lines(
    kernel: &KernelDecl,
    buffer_arg_names: &[(String, bool)],
    indirect_scalar_buffer_arg_names: &[String],
    has_len_arg: bool,
    has_block_x_arg: bool,
    operation_args: bool,
) -> String {
    let mut out = String::new();
    out.push_str("        rocm_oxide::validate_launch_config(config)?;\n");
    let length_buffer_arg_names = buffer_arg_names
        .iter()
        .filter(|(name, _)| {
            !indirect_scalar_buffer_arg_names
                .iter()
                .any(|indirect_name| indirect_name == name)
        })
        .collect::<Vec<_>>();
    if kernel.contracts.is_empty() && has_len_arg {
        for (arg_name, _) in length_buffer_arg_names {
            out.push_str(&format!(
                "        rocm_oxide::validate_buffer_len(\"{arg_name}\", {arg_name}.len(), n)?;\n"
            ));
        }
    } else if kernel.contracts.is_empty() && length_buffer_arg_names.len() > 1 {
        let (reference, _) = length_buffer_arg_names[0];
        for (arg_name, _) in length_buffer_arg_names.iter().skip(1) {
            out.push_str(&format!(
                "        rocm_oxide::validate_buffer_len(\"{arg_name}\", {arg_name}.len(), {reference}.len())?;\n"
            ));
        }
    }
    for contract in &kernel.contracts {
        out.push_str(&format!(
            "        rocm_oxide::validate_buffer_len(\"{}\", {}.len(), {})?;\n",
            contract.buffer,
            contract.buffer,
            contract.required_len.as_rust()
        ));
    }
    for arg_name in indirect_scalar_buffer_arg_names {
        out.push_str(&format!(
            "        rocm_oxide::validate_buffer_len(\"{arg_name}\", {arg_name}.len(), 1)?;\n"
        ));
    }
    if has_block_x_arg {
        out.push_str("        rocm_oxide::validate_block_x(config, block_x)?;\n");
    }
    for left_index in 0..buffer_arg_names.len() {
        for right_index in (left_index + 1)..buffer_arg_names.len() {
            let (left_name, left_mut) = &buffer_arg_names[left_index];
            let (right_name, right_mut) = &buffer_arg_names[right_index];
            if *left_mut || *right_mut {
                let left_arg = if operation_args {
                    format!("{left_name}.as_ref()")
                } else {
                    left_name.clone()
                };
                let right_arg = if operation_args {
                    format!("{right_name}.as_ref()")
                } else {
                    right_name.clone()
                };
                out.push_str(&format!(
                    "        rocm_oxide::validate_device_buffers_disjoint(\"{left_name}\", {left_arg}, \"{right_name}\", {right_arg})?;\n"
                ));
            }
        }
    }
    out
}

impl ArgKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::MutPtr(_) => "mut_ptr",
            Self::ConstPtr(_) => "const_ptr",
            Self::MutSlice(_) => "mut_slice",
            Self::ConstSlice(_) => "const_slice",
            Self::Scalar => "scalar",
        }
    }

    fn is_buffer(&self) -> bool {
        matches!(
            self,
            Self::MutPtr(_) | Self::ConstPtr(_) | Self::MutSlice(_) | Self::ConstSlice(_)
        )
    }
}

impl DeviceGlobalKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Constant => "constant",
            Self::Shared => "shared",
        }
    }

    fn address_space(self) -> &'static str {
        match self {
            Self::Global => "1",
            Self::Constant => "4",
            Self::Shared => "3",
        }
    }

    fn has_host_binding(self) -> bool {
        !matches!(self, Self::Shared)
    }
}

impl LenExpr {
    fn identifiers(&self) -> Vec<String> {
        tokenize_len_expr(&self.source)
            .unwrap_or_default()
            .into_iter()
            .filter(|token| is_identifier(token))
            .collect()
    }

    fn as_rust(&self) -> String {
        self.source.clone()
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn run_command(command: &mut Command, label: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|err| format!("failed to {label}: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "failed to {label}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn compiler_step<T, F>(label: &str, f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            Err(format!(
                "internal compiler panic while trying to {label}: {message}\n\
                 this is a rocm-oxide-build bug; rerun with the generated .ll file preserved and report the kernel source span above if present"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArgKind, CodeObjectMetadata, DeviceGlobalKind, KernelArgObjectMetadata,
        KernelObjectMetadata,
        annotate_dynamic_shared_mem_from_ir, compiler_step, discover_device_crate_bundle,
        discover_device_globals_in_source, discover_device_structs_in_source,
        discover_kernels_in_source, generate_device_global_binding, generate_device_struct_binding,
        generate_kernel_binding, generate_kernel_resource_binding, parse_inline_path_dependency,
        parse_kernel_resource_rows, parse_package_name, transform_ir,
        validate_code_object_metadata, verify_lds_ir, verify_lds_isa, verify_scoped_atomic_ir,
        verify_scoped_atomic_isa,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn kernel_map(source: &str) -> BTreeMap<String, super::KernelDecl> {
        discover_kernels_in_source(source)
            .expect("source should parse")
            .into_iter()
            .map(|kernel| (kernel.name.clone(), kernel))
            .collect()
    }

    #[test]
    fn discovers_marked_kernel_names() {
        let input = r#"
use rocm_oxide_kernel::kernel;

#[kernel]
pub unsafe extern "C" fn vector_add() {}

pub unsafe extern "C" fn helper() {}
"#;
        let kernels = discover_kernels_in_source(input).expect("source should parse");
        let names = kernels
            .into_iter()
            .map(|kernel| kernel.name)
            .collect::<BTreeSet<_>>();
        assert_eq!(names, BTreeSet::from(["vector_add".to_string()]));
    }

    #[test]
    fn rocm_tool_candidates_cover_common_rocm_layouts() {
        let llvm_paths = super::rocm_tool_paths(Path::new("/rocm"), "llc", super::ToolLayout::Llvm);
        assert_eq!(
            llvm_paths,
            vec![
                PathBuf::from("/rocm/lib/llvm/bin/llc"),
                PathBuf::from("/rocm/llvm/bin/llc"),
                PathBuf::from("/rocm/bin/llc"),
            ]
        );

        let bin_paths =
            super::rocm_tool_paths(Path::new("/rocm"), "rocminfo", super::ToolLayout::Bin);
        assert_eq!(
            bin_paths,
            vec![
                PathBuf::from("/rocm/bin/rocminfo"),
                PathBuf::from("/rocm/rocminfo"),
            ]
        );
    }

    #[test]
    fn rejects_non_gfx_architecture_with_actionable_hint() {
        let err = super::validate_gpu_arch("sm_90")
            .expect_err("CUDA architecture should not be accepted as ROCm gfx");
        assert!(err.contains("unsupported GPU architecture `sm_90`"));
        assert!(err.contains("--arch gfx"));
        assert!(err.contains("ROCM_OXIDE_ARCH=gfx"));
    }

    #[test]
    fn missing_rocm_tool_diagnostic_lists_checked_candidates() {
        let candidates = vec![
            super::ToolPath {
                path: PathBuf::from("/rocm/lib/llvm/bin/llc"),
                source: "ROCM_PATH".to_string(),
            },
            super::ToolPath {
                path: PathBuf::from("llc"),
                source: "PATH".to_string(),
            },
        ];
        let message = super::format_missing_rocm_tool("ROCM_OXIDE_LLC", "llc", &candidates);
        assert!(message.contains("could not find `llc`"));
        assert!(message.contains("[ROCM_PATH] /rocm/lib/llvm/bin/llc"));
        assert!(message.contains("[PATH] llc"));
        assert!(message.contains("ROCM_OXIDE_LLC=/path/to/llc"));
    }

    #[test]
    fn rocm_root_candidates_are_deduplicated() {
        let mut roots = Vec::new();
        super::push_rocm_root(&mut roots, "first", PathBuf::from("/rocm"));
        super::push_rocm_root(&mut roots, "second", PathBuf::from("/rocm"));
        assert_eq!(roots, vec![("first".to_string(), PathBuf::from("/rocm"))]);
    }

    #[test]
    fn device_debug_env_flag_accepts_common_truthy_and_falsey_values() {
        assert!(!super::env_flag_enabled(None));
        assert!(!super::env_flag_enabled(Some(std::ffi::OsStr::new(""))));
        assert!(!super::env_flag_enabled(Some(std::ffi::OsStr::new("0"))));
        assert!(!super::env_flag_enabled(Some(std::ffi::OsStr::new("false"))));
        assert!(!super::env_flag_enabled(Some(std::ffi::OsStr::new("OFF"))));
        assert!(super::env_flag_enabled(Some(std::ffi::OsStr::new("1"))));
        assert!(super::env_flag_enabled(Some(std::ffi::OsStr::new("true"))));
        assert!(super::env_flag_enabled(Some(std::ffi::OsStr::new("debug"))));
    }

    #[test]
    fn device_rustflags_keep_build_std_dependencies_on_target_cpu_only() {
        assert_eq!(super::device_rustflags("gfx1201"), "-C target-cpu=gfx1201");
        assert_eq!(super::device_debug_rustc_args(false), &[] as &[&str]);
        assert_eq!(
            super::device_debug_rustc_args(true),
            &["-C", "debuginfo=2"] as &[&str]
        );
    }

    #[test]
    fn strips_rocm_llc_unsupported_dwarf_address_space_metadata() {
        let input = r#"!54 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut f32", baseType: !4, size: 64, align: 64, dwarfAddressSpace: 0)
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut u32", baseType: !5, size: 64, dwarfAddressSpace: 1, flags: DIFlagArtificial)
"#;
        let output = super::strip_rocm_llc_unsupported_debug_metadata(input);
        assert!(!output.contains("dwarfAddressSpace"));
        assert!(output.contains("align: 64)"));
        assert!(output.contains("size: 64, flags: DIFlagArtificial)"));
    }

    #[test]
    fn parses_per_kernel_resource_rows_from_metadata_json() {
        let input = r#"{
  "kernels": [
    {
      "name": "vector_add",
      "args": [],
      "contracts": [],
      "code_object": {
        "kernarg_segment_size": 296,
        "max_flat_workgroup_size": 1024,
        "group_segment_fixed_size": 0,
        "private_segment_fixed_size": 4,
        "sgpr_count": 11,
        "vgpr_count": 4,
        "sgpr_spill_count": 0,
        "vgpr_spill_count": 1,
        "wavefront_size": 32,
        "uses_dynamic_stack": false
      }
    }
  ],
  "globals": [
    {
      "name": "ADD_ONE_DELTA",
      "type": "f32",
      "kind": "global",
      "mutable": true,
      "address_space": "1"
    }
  ]
}"#;
        let rows = parse_kernel_resource_rows(input);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "vector_add");
        assert_eq!(rows[0].vgpr_count, Some(4));
        assert_eq!(rows[0].sgpr_count, Some(11));
        assert_eq!(rows[0].vgpr_spill_count, Some(1));
        assert_eq!(rows[0].uses_dynamic_stack, Some(false));
    }

    #[test]
    fn validates_linked_code_object_metadata_for_all_kernels() {
        let mut metadata = CodeObjectMetadata::default();
        metadata
            .kernels
            .insert("present".to_string(), KernelObjectMetadata::default());
        let expected = ["present".to_string(), "missing".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();

        let err = validate_code_object_metadata(&metadata, &expected)
            .expect_err("missing linked metadata should fail");
        assert!(err.contains("missing"));

        metadata
            .kernels
            .insert("missing".to_string(), KernelObjectMetadata::default());
        validate_code_object_metadata(&metadata, &expected)
            .expect("complete linked metadata should pass");
    }

    #[test]
    fn annotates_dynamic_shared_mem_from_kernel_ir() {
        let input = r#"
@scratch = external local_unnamed_addr addrspace(3) global [0 x i8], align 4

define protected amdgpu_kernel void @lds_block_sum(ptr addrspace(1) %out) {
start:
  %slot = getelementptr inbounds float, ptr addrspace(3) @scratch, i32 0
  store float 1.0, ptr addrspace(3) %slot, align 4
  ret void
}
"#;
        let path =
            std::env::temp_dir().join(format!("rocm-oxide-dynamic-lds-{}.ll", std::process::id()));
        fs::write(&path, input).expect("temp IR should be writable");
        let mut metadata = CodeObjectMetadata::default();
        metadata
            .kernels
            .insert("lds_block_sum".to_string(), KernelObjectMetadata::default());

        annotate_dynamic_shared_mem_from_ir(&mut metadata, &path)
            .expect("IR annotation should succeed");
        let _ = fs::remove_file(&path);

        assert!(
            metadata
                .kernels
                .get("lds_block_sum")
                .expect("kernel metadata exists")
                .uses_dynamic_shared_mem()
        );
    }

    #[test]
    fn rewrites_marked_function_into_kernel() {
        let input = r#"; ModuleID = 'sample'
target triple = "amdgcn-amd-amdhsa"

define void @vector_add(ptr noundef writeonly %out, ptr noundef readonly %input, i64 noundef %n) unnamed_addr #0 {
start:
  %idx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %i = zext i32 %idx to i64
  %dst = getelementptr inbounds float, ptr %out, i64 %i
  %src = getelementptr inbounds float, ptr %input, i64 %i
  %value = load float, ptr %src, align 4
  store float %value, ptr %dst, align 4
  ret void
}

attributes #0 = { nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) "target-cpu"="gfx1201" }
"#;

        let decls = kernel_map(
            r#"
#[kernel]
pub unsafe extern "C" fn vector_add(out: *mut f32, input: *const f32, n: usize) {}
"#,
        );
        let kernels = decls.keys().cloned().collect::<BTreeSet<_>>();
        let output = transform_ir(input, &kernels, &decls, &BTreeMap::new())
            .expect("transform should succeed");
        assert!(output.contains("define protected amdgpu_kernel void @vector_add"));
        assert!(output.contains("ptr addrspace(1) noundef writeonly %out"));
        assert!(output.contains("ptr addrspace(1) noundef readonly %input"));
        assert!(output.contains("load float, ptr addrspace(1) %src"));
        assert!(output.contains("store float %value, ptr addrspace(1) %dst"));
        assert!(output.contains("\"amdgpu-flat-work-group-size\"=\"1,1024\""));
        assert!(output.contains("\"amdgpu-no-hostcall-ptr\""));
        assert!(!output.contains("target_mem"));
    }

    #[test]
    fn rejects_modules_without_exported_kernels() {
        let decls = kernel_map(
            r#"
#[kernel]
pub unsafe extern "C" fn vector_add(out: *mut f32) {}
"#,
        );
        let kernels = decls.keys().cloned().collect::<BTreeSet<_>>();
        let err = transform_ir(
            "define void @helper() {\n  ret void\n}\n",
            &kernels,
            &decls,
            &BTreeMap::new(),
        )
        .expect_err("module without kernels should fail");
        assert!(err.contains("none of the marked kernels"));
        assert!(err.contains("<source>:2"));
    }

    #[test]
    fn generates_typed_host_binding() {
        let kernels = discover_kernels_in_source(
            r#"
#[kernel]
pub unsafe extern "C" fn vector_add(
    out: *mut f32,
    a: *const f32,
    n: usize,
) {}
"#,
        )
        .expect("source should parse");

        assert_eq!(kernels[0].args[0].kind, ArgKind::MutPtr("f32".to_string()));
        assert_eq!(
            kernels[0].args[1].kind,
            ArgKind::ConstPtr("f32".to_string())
        );
        assert_eq!(kernels[0].args[2].kind, ArgKind::Scalar);

        let binding = generate_kernel_binding(&kernels[0], &BTreeMap::new(), None)
            .expect("binding should generate");
        assert!(binding.contains("out: &rocm_oxide::DeviceBuffer<f32>"));
        assert!(binding.contains("a: &rocm_oxide::DeviceBuffer<f32>"));
        assert!(binding.contains("n: usize"));
        assert!(binding.contains("validate_launch_config(config)?"));
        assert!(binding.contains("validate_buffer_len(\"out\", out.len(), n)?"));
        assert!(binding.contains("validate_buffer_len(\"a\", a.len(), n)?"));
        assert!(binding.contains("out.as_mut_ptr()"));
        assert!(binding.contains("a.as_ptr()"));
        assert!(binding.contains("pub unsafe fn vector_add_operation"));
        assert!(binding.contains("pub unsafe fn vector_add_graph_node"));
        assert!(binding.contains("graph: &rocm_oxide::hip::Graph"));
        assert!(binding.contains("dependencies: &[rocm_oxide::hip::GraphNode]"));
        assert!(binding.contains(
            "add_graph_node_raw(graph, dependencies, config, &mut __params)"
        ));
        assert!(binding.contains("out: std::sync::Arc<rocm_oxide::DeviceBuffer<f32>>"));
        assert!(binding.contains("a: std::sync::Arc<rocm_oxide::DeviceBuffer<f32>>"));
        assert!(binding.contains("Output = rocm_oxide::KernelLaunchCompletion"));
        assert!(binding.contains("launch_raw_on_stream(context.stream(), config, &mut __params)?"));
        assert!(binding.contains("__completion.keep_alive(module);"));
        assert!(binding.contains("__completion.keep_alive(kernel);"));
        assert!(binding.contains("__completion.keep_alive(out);"));
        assert!(binding.contains("__completion.keep_alive(a);"));
    }

    #[test]
    fn generates_device_slice_host_binding() {
        let kernels = discover_kernels_in_source(
            r#"
#[kernel]
pub unsafe extern "C" fn vector_add(
    out: gpu::DeviceSliceMut<f32>,
    a: gpu::DeviceSlice<f32>,
    b: rocm_oxide_device::DeviceSlice<f32>,
) {}
"#,
        )
        .expect("source should parse");

        assert_eq!(
            kernels[0].args[0].kind,
            ArgKind::MutSlice("f32".to_string())
        );
        assert_eq!(
            kernels[0].args[1].kind,
            ArgKind::ConstSlice("f32".to_string())
        );
        assert_eq!(
            kernels[0].args[2].kind,
            ArgKind::ConstSlice("f32".to_string())
        );

        let binding = generate_kernel_binding(&kernels[0], &BTreeMap::new(), None)
            .expect("binding should generate");
        assert!(binding.contains("out: &rocm_oxide::DeviceBuffer<f32>"));
        assert!(binding.contains("a: &rocm_oxide::DeviceBuffer<f32>"));
        assert!(binding.contains("b: &rocm_oxide::DeviceBuffer<f32>"));
        assert!(binding.contains("validate_buffer_len(\"a\", a.len(), out.len())?"));
        assert!(binding.contains("validate_buffer_len(\"b\", b.len(), out.len())?"));
        assert!(binding.contains("validate_device_buffers_disjoint(\"out\", out, \"a\", a)?"));
        assert!(binding.contains("validate_device_buffers_disjoint(\"out\", out, \"b\", b)?"));
        assert!(binding.contains(
            "validate_device_buffers_disjoint(\"out\", out.as_ref(), \"a\", a.as_ref())?"
        ));
        assert!(binding.contains(
            "validate_device_buffers_disjoint(\"out\", out.as_ref(), \"b\", b.as_ref())?"
        ));
        assert!(binding.contains("out.as_mut_ptr()"));
        assert!(binding.contains("out.len()"));
        assert!(binding.contains("a.as_ptr()"));
        assert!(binding.contains("a.len()"));
    }

    #[test]
    fn generates_kernel_resource_binding() {
        let kernels = discover_kernels_in_source(
            r#"
#[kernel]
pub unsafe extern "C" fn lds_block_sum(out: *mut f32) {}
"#,
        )
        .expect("source should parse");
        let metadata = KernelObjectMetadata {
            kernarg_segment_size: Some(312),
            kernarg_segment_align: Some(8),
            max_flat_workgroup_size: Some(1024),
            group_segment_fixed_size: Some(0),
            private_segment_fixed_size: Some(0),
            sgpr_count: Some(14),
            vgpr_count: Some(6),
            sgpr_spill_count: Some(0),
            vgpr_spill_count: Some(0),
            wavefront_size: Some(32),
            uses_dynamic_shared_mem: true,
            uses_dynamic_stack: Some(false),
            args: BTreeMap::new(),
        };
        let binding = generate_kernel_resource_binding(&kernels[0], Some(&metadata));
        assert!(binding.contains("name: \"lds_block_sum\""));
        assert!(binding.contains("kernarg_segment_size: Some(312)"));
        assert!(binding.contains("uses_dynamic_shared_mem: true"));
        assert!(binding.contains("uses_dynamic_stack: Some(false)"));
    }

    #[test]
    fn discovers_marked_device_globals() {
        let globals = discover_device_globals_in_source(
            r#"
use rocm_oxide_kernel::{constant, device_global, shared};

#[device_global]
pub static mut ADD_ONE_DELTA: f32 = 1.0;

#[constant]
pub static LUT: [u32; 4] = [1, 2, 3, 4];

#[shared]
pub static mut STATIC_LDS_U32: [u32; 256] = [0; 256];
"#,
        )
        .expect("source should parse");

        assert_eq!(globals.len(), 3);
        assert_eq!(globals[0].name, "ADD_ONE_DELTA");
        assert_eq!(globals[0].ty, "f32");
        assert_eq!(globals[0].kind, DeviceGlobalKind::Global);
        assert!(globals[0].mutable);
        assert_eq!(globals[1].name, "LUT");
        assert_eq!(globals[1].ty, "[u32; 4]");
        assert_eq!(globals[1].kind, DeviceGlobalKind::Constant);
        assert!(!globals[1].mutable);
        assert_eq!(globals[2].name, "STATIC_LDS_U32");
        assert_eq!(globals[2].ty, "[u32; 256]");
        assert_eq!(globals[2].kind, DeviceGlobalKind::Shared);
        assert!(globals[2].mutable);

        let binding = generate_device_global_binding(&globals[0]);
        assert!(binding.contains("pub fn global_add_one_delta"));
        assert!(binding.contains("rocm_oxide::Global<f32>"));
        assert!(binding.contains("self.module.global(c\"ADD_ONE_DELTA\")"));
    }

    #[test]
    fn rewrites_marked_device_globals_with_address_spaces() {
        let input = r#"; ModuleID = 'sample'
target triple = "amdgcn-amd-amdhsa"

@ADD_ONE_DELTA = local_unnamed_addr global float 1.0
@LUT = local_unnamed_addr constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
@STATIC_LDS_U32 = local_unnamed_addr global [1024 x i8] zeroinitializer

define void @use_globals(ptr noundef %out) unnamed_addr #0 {
start:
  %delta = load float, ptr @ADD_ONE_DELTA, align 4
  %slot = getelementptr inbounds [4 x i32], ptr @LUT, i64 0, i64 1
  %value = load i32, ptr %slot, align 4
  %scratch = getelementptr inbounds i32, ptr @STATIC_LDS_U32, i64 2
  store i32 %value, ptr %scratch, align 4
  %scratch_value = load i32, ptr %scratch, align 4
  store float %delta, ptr %out, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="gfx1201" }
"#;
        let decls = kernel_map(
            r#"
#[kernel]
pub unsafe extern "C" fn use_globals(out: *mut f32) {}
"#,
        );
        let globals = discover_device_globals_in_source(
            r#"
#[device_global]
pub static mut ADD_ONE_DELTA: f32 = 1.0;
#[constant]
pub static LUT: [u32; 4] = [1, 2, 3, 4];
#[shared]
pub static mut STATIC_LDS_U32: [u32; 256] = [0; 256];
"#,
        )
        .expect("globals should parse")
        .into_iter()
        .map(|global| (global.name.clone(), global))
        .collect::<BTreeMap<_, _>>();
        let kernels = decls.keys().cloned().collect::<BTreeSet<_>>();
        let output =
            transform_ir(input, &kernels, &decls, &globals).expect("transform should succeed");

        assert!(
            output.contains("@ADD_ONE_DELTA = local_unnamed_addr addrspace(1) global float 1.0")
        );
        assert!(output.contains("@LUT = local_unnamed_addr addrspace(4) constant [4 x i32]"));
        assert!(output.contains(
            "@STATIC_LDS_U32 = local_unnamed_addr addrspace(3) global [1024 x i8] undef"
        ));
        assert!(output.contains("load float, ptr addrspace(1) @ADD_ONE_DELTA"));
        assert!(output.contains("getelementptr inbounds [4 x i32], ptr addrspace(4) @LUT"));
        assert!(output.contains("load i32, ptr addrspace(4) %slot"));
        assert!(output.contains("getelementptr inbounds i32, ptr addrspace(3) @STATIC_LDS_U32"));
        assert!(output.contains("store i32 %value, ptr addrspace(3) %scratch"));
        assert!(output.contains("load i32, ptr addrspace(3) %scratch"));
    }

    #[test]
    fn parses_length_contracts_into_generated_validation() {
        let kernels = discover_kernels_in_source(
            r#"
// rocm-oxide: len(frame)=pixel_count
// rocm-oxide: len(color)=pixel_count/4
// rocm-oxide: len(aux)=pixel_count/4*3
#[kernel]
pub unsafe extern "C" fn temporal(
    frame: *mut u32,
    color: *const u32,
    aux: *const f32,
    pixel_count: usize,
) {}
"#,
        )
        .expect("source should parse");

        let binding = generate_kernel_binding(&kernels[0], &BTreeMap::new(), None)
            .expect("binding should generate");
        assert!(binding.contains("validate_buffer_len(\"frame\", frame.len(), pixel_count)?"));
        assert!(binding.contains("validate_buffer_len(\"color\", color.len(), pixel_count/4)?"));
        assert!(binding.contains("validate_buffer_len(\"aux\", aux.len(), pixel_count/4*3)?"));
    }

    #[test]
    fn rejects_contracts_that_reference_non_scalar_args() {
        let err = discover_kernels_in_source(
            r#"
// rocm-oxide: len(out)=input
#[kernel]
pub unsafe extern "C" fn bad(out: *mut f32, input: *const f32) {}
"#,
        )
        .expect_err("contract should fail");

        assert!(err.contains("references non-scalar"));
    }

    #[test]
    fn propagates_global_pointer_address_space_through_more_ir_ops() {
        let input = r#"; ModuleID = 'sample'
target triple = "amdgcn-amd-amdhsa"

define void @pointer_ops(ptr noundef %out, ptr noundef %fallback, i1 noundef %cond) unnamed_addr #0 {
start:
  %gep = getelementptr inbounds i32, ptr %out, i64 1
  %selected = select i1 %cond, ptr %gep, ptr %fallback
  %phi = phi ptr [ %selected, %start ], [ %gep, %start ]
  store i32 7, ptr %phi, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="gfx1201" }
"#;
        let decls = kernel_map(
            r#"
#[kernel]
pub unsafe extern "C" fn pointer_ops(out: *mut u32, fallback: *mut u32, cond: bool) {}
"#,
        );
        let kernels = decls.keys().cloned().collect::<BTreeSet<_>>();
        let output = transform_ir(input, &kernels, &decls, &BTreeMap::new())
            .expect("transform should succeed");
        assert!(output.contains(
            "%selected = select i1 %cond, ptr addrspace(1) %gep, ptr addrspace(1) %fallback"
        ));
        assert!(
            output.contains("%phi = phi ptr addrspace(1) [ %selected, %start ], [ %gep, %start ]")
        );
        assert!(output.contains("store i32 7, ptr addrspace(1) %phi"));
    }

    #[test]
    fn rejects_unsupported_pointer_integer_casts_with_source_span() {
        let input = r#"; ModuleID = 'sample'
target triple = "amdgcn-amd-amdhsa"

define void @bad_cast(ptr noundef %out, i64 noundef %addr) unnamed_addr #0 {
start:
  %raw = inttoptr i64 %addr to ptr
  %roundtrip = ptrtoint ptr %out to i64
  store i32 7, ptr %raw, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="gfx1201" }
"#;
        let decls = kernel_map(
            r#"
#[kernel]
pub unsafe extern "C" fn bad_cast(out: *mut u32, addr: usize) {}
"#,
        );
        let kernels = decls.keys().cloned().collect::<BTreeSet<_>>();
        let err = transform_ir(input, &kernels, &decls, &BTreeMap::new())
            .expect_err("pointer/integer casts should be rejected");
        assert!(err.contains("<source>:2"));
        assert!(err.contains("bad_cast"));
        assert!(err.contains("unsupported pointer/integer cast `inttoptr`"));
        assert!(err.contains("%raw = inttoptr i64 %addr to ptr"));
    }

    #[test]
    fn rewrites_atomic_scope_markers_to_llvm_syncscopes() {
        let input = r#"; ModuleID = 'sample'
target triple = "amdgcn-amd-amdhsa"

define void @scoped(ptr noundef %counters) unnamed_addr #0 {
start:
  call void @__rocm_oxide_atomic_scope_workgroup(ptr %counters)
  %wg = atomicrmw add ptr %counters, i32 1 monotonic, align 4
  call void @__rocm_oxide_atomic_scope_device(ptr %counters)
  %dev = atomicrmw add ptr %counters, i32 1 monotonic, align 4
  call void @__rocm_oxide_atomic_scope_device(ptr %counters)
  %cas = cmpxchg ptr %counters, i32 0, i32 1 monotonic monotonic, align 4
  call void @__rocm_oxide_atomic_scope_workgroup(ptr %counters)
  store atomic i32 7, ptr %counters release, align 4
  call void @__rocm_oxide_atomic_scope_system(ptr %counters)
  %sys = load atomic i32, ptr %counters acquire, align 4
  ret void
}

declare void @__rocm_oxide_atomic_scope_workgroup(ptr)
declare void @__rocm_oxide_atomic_scope_device(ptr)
declare void @__rocm_oxide_atomic_scope_system(ptr)

attributes #0 = { nounwind "target-cpu"="gfx1201" }
"#;
        let decls = kernel_map(
            r#"
#[kernel]
pub unsafe extern "C" fn scoped(counters: *mut u32) {}
"#,
        );
        let kernels = decls.keys().cloned().collect::<BTreeSet<_>>();
        let output = transform_ir(input, &kernels, &decls, &BTreeMap::new())
            .expect("transform should succeed");
        assert!(output.contains(
            "%wg = atomicrmw add ptr addrspace(1) %counters, i32 1 syncscope(\"workgroup\") monotonic, align 4"
        ));
        assert!(output.contains(
            "%dev = atomicrmw add ptr addrspace(1) %counters, i32 1 syncscope(\"agent\") monotonic, align 4"
        ));
        assert!(output.contains(
            "%cas = cmpxchg ptr addrspace(1) %counters, i32 0, i32 1 syncscope(\"agent\") monotonic monotonic, align 4"
        ));
        assert!(
            output.contains("store atomic i32 7, ptr addrspace(1) %counters syncscope(\"workgroup\") release, align 4")
        );
        assert!(
            output.contains("%sys = load atomic i32, ptr addrspace(1) %counters acquire, align 4")
        );
        assert!(!output.contains("__rocm_oxide_atomic_scope_"));
    }

    #[test]
    fn verifies_lds_ir_for_dynamic_and_static_cases() {
        let ir = r#"
@anon.dynamic = external local_unnamed_addr addrspace(3) global [0 x i8], align 4
@STATIC_LDS_U32 = addrspace(3) global [1024 x i8] undef, align 4

define protected amdgpu_kernel void @lds_block_sum(ptr addrspace(1) %out) {
start:
  %scratch = getelementptr inbounds i8, ptr addrspace(3) @anon.dynamic, i64 0
  store i32 1, ptr addrspace(3) %scratch, align 4
  %value = load i32, ptr addrspace(3) %scratch, align 4
  ret void
}

define protected amdgpu_kernel void @static_lds_reverse(ptr addrspace(1) %out) {
start:
  %scratch = getelementptr inbounds i32, ptr addrspace(3) @STATIC_LDS_U32, i64 1
  store i32 7, ptr addrspace(3) %scratch, align 4
  %value = load i32, ptr addrspace(3) %scratch, align 4
  ret void
}
"#;

        verify_lds_ir(ir).expect("LDS IR should verify");
    }

    #[test]
    fn rejects_static_lds_ir_without_addrspace_three() {
        let ir = r#"
@STATIC_LDS_U32 = addrspace(1) global [1024 x i8] zeroinitializer, align 4

define protected amdgpu_kernel void @static_lds_reverse(ptr addrspace(1) %out) {
start:
  %scratch = getelementptr inbounds i32, ptr addrspace(1) @STATIC_LDS_U32, i64 1
  store i32 7, ptr addrspace(1) %scratch, align 4
  %value = load i32, ptr addrspace(1) %scratch, align 4
  ret void
}
"#;

        let err = verify_lds_ir(ir).expect_err("static LDS must live in addrspace(3)");
        assert!(err.contains("static_lds_reverse IR did not preserve static LDS"));
        assert!(err.contains("shared definition: false"));
    }

    #[test]
    fn verifies_lds_isa_for_dynamic_and_static_cases() {
        let disassembly = r#"
0000000000002400 <lds_block_sum>:
  ds_store_b32 v4, v5
  ds_load_b32 v3, v4

000000000000a100 <static_lds_reverse>:
  ds_store_b32 v2, v3
  ds_load_b32 v2, v2 offset:1020
"#;

        verify_lds_isa(disassembly).expect("LDS ISA should verify");
    }

    #[test]
    fn rejects_lds_isa_without_static_load() {
        let disassembly = r#"
000000000000a100 <static_lds_reverse>:
  ds_store_b32 v2, v3
"#;

        let err = verify_lds_isa(disassembly).expect_err("static LDS should load from LDS");
        assert!(err.contains("static_lds_reverse ISA did not contain expected LDS"));
        assert!(err.contains("LDS load: false"));
    }

    #[test]
    fn verifies_scoped_atomic_ir_mapping() {
        let ir = r#"
define protected amdgpu_kernel void @scoped_atomics(ptr addrspace(1) %counters) {
start:
  %wg = atomicrmw add ptr addrspace(1) %counters, i32 1 syncscope("workgroup") monotonic, align 4
  %dev_ptr = getelementptr inbounds i8, ptr addrspace(1) %counters, i64 4
  %dev = atomicrmw add ptr addrspace(1) %dev_ptr, i32 1 syncscope("agent") monotonic, align 4
  %sys_ptr = getelementptr inbounds i8, ptr addrspace(1) %counters, i64 8
  %sys = atomicrmw add ptr addrspace(1) %sys_ptr, i32 1 monotonic, align 4
  ret void
}
"#;

        verify_scoped_atomic_ir(ir).expect("scoped atomic IR should verify");
    }

    #[test]
    fn rejects_scoped_atomic_ir_without_system_default() {
        let ir = r#"
define protected amdgpu_kernel void @scoped_atomics(ptr addrspace(1) %counters) {
start:
  %wg = atomicrmw add ptr addrspace(1) %counters, i32 1 syncscope("workgroup") monotonic, align 4
  %dev = atomicrmw add ptr addrspace(1) %counters, i32 1 syncscope("agent") monotonic, align 4
  %sys = atomicrmw add ptr addrspace(1) %counters, i32 1 syncscope("agent") monotonic, align 4
  ret void
}
"#;

        let err =
            verify_scoped_atomic_ir(ir).expect_err("system scope should stay backend default");
        assert!(err.contains("system backend default: false"));
    }

    #[test]
    fn verifies_scoped_atomic_isa_mapping() {
        let disassembly = r#"
0000000000009f00 <scoped_atomics>:
  global_atomic_add_u32 v2, v3, s[0:1] scope:SCOPE_SE
  global_atomic_add_u32 v2, v3, s[0:1] offset:4 scope:SCOPE_DEV
  global_atomic_add_u32 v2, v3, s[0:1] offset:8 scope:SCOPE_SYS

000000000000a100 <other>:
  s_endpgm
"#;

        verify_scoped_atomic_isa(disassembly).expect("scoped atomic ISA should verify");
    }

    #[test]
    fn verifies_scoped_atomic_isa_when_objdump_omits_scope_annotations() {
        let disassembly = r#"
0000000000009f00 <scoped_atomics>:
  global_atomic_add_u32 v2, v3, s[0:1]
  global_atomic_add_u32 v2, v3, s[0:1] offset:4
  global_atomic_add_u32 v2, v3, s[0:1] offset:8

000000000000a100 <other>:
  s_endpgm
"#;

        verify_scoped_atomic_isa(disassembly)
            .expect("unannotated scoped atomic ISA should still verify");
    }

    #[test]
    fn rejects_scoped_atomic_isa_without_system_scope() {
        let disassembly = r#"
0000000000009f00 <scoped_atomics>:
  global_atomic_add_u32 v2, v3, s[0:1] scope:SCOPE_SE
  global_atomic_add_u32 v2, v3, s[0:1] offset:4 scope:SCOPE_DEV

000000000000a100 <other>:
  s_endpgm
"#;

        let err = verify_scoped_atomic_isa(disassembly)
            .expect_err("ISA should contain the system-scope atomic");
        assert!(err.contains("system/SCOPE_SYS: false"));
    }

    #[test]
    fn generic_kernel_diagnostic_points_to_source_span() {
        let err = discover_kernels_in_source(
            r#"
#[kernel]
pub unsafe extern "C" fn copy_generic<T>(out: *mut T, input: *const T, n: usize) {}
"#,
        )
        .expect_err("generic kernels should get an actionable diagnostic");
        assert!(err.contains("<source>:2"));
        assert!(err.contains("generic #[kernel] functions require"));
        assert!(err.contains("monomorphize"));
    }

    #[test]
    fn discovers_monomorphized_generic_kernels() {
        let kernels = discover_kernels_in_source(
            r#"
// rocm-oxide: len(out)=n
// rocm-oxide: len(input)=n
#[kernel(monomorphize(f32), monomorphize(u32))]
pub unsafe extern "C" fn copy_generic<T: Copy>(out: *mut T, input: *const T, n: usize) {}
"#,
        )
        .expect("generic kernel specializations should parse");
        let names = kernels
            .iter()
            .map(|kernel| kernel.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["copy_generic_f32", "copy_generic_u32"]);
        assert_eq!(kernels[0].args[0].kind, ArgKind::MutPtr("f32".to_string()));
        assert_eq!(
            kernels[0].args[1].kind,
            ArgKind::ConstPtr("f32".to_string())
        );
        assert_eq!(kernels[1].args[0].kind, ArgKind::MutPtr("u32".to_string()));
        assert_eq!(kernels[0].contracts.len(), 2);
    }

    #[test]
    fn discovers_generic_kernels_with_closure_trait_bounds() {
        let kernels = discover_kernels_in_source(
            r#"
#[kernel(monomorphize(ClosureEnv))]
pub unsafe extern "C" fn apply_closure<F: FnOnce(u32) -> u32>(
    out: *mut u32,
    input: *const u32,
    f: F,
) {}
"#,
        )
        .expect("generic closure-bound kernel should parse");

        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "apply_closure_ClosureEnv");
        assert_eq!(kernels[0].args[2].ty, "ClosureEnv");
        assert_eq!(kernels[0].args[2].kind, ArgKind::Scalar);
    }

    #[test]
    fn lowers_indirect_host_to_device_closure_argument_envs() {
        let source = r#"
#[derive(Clone, Copy)]
pub struct HostAffineClosure {
    pub base: u32,
    pub stride: u32,
    pub xor_mask: u32,
}

#[kernel(monomorphize(HostAffineClosure))]
pub unsafe extern "C" fn apply_closure<F: FnOnce(u32) -> u32 + Copy>(
    out: gpu::DeviceSliceMut<u32>,
    input: gpu::DeviceSlice<u32>,
    f: F,
    n: usize,
) {}
"#;
        let kernels = discover_kernels_in_source(source).expect("source should parse");
        let device_structs = discover_device_structs_in_source(source)
            .expect("closure environment struct should parse")
            .into_iter()
            .map(|device_struct| (device_struct.name.clone(), device_struct))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(kernels[0].name, "apply_closure_HostAffineClosure");
        assert_eq!(kernels[0].args[2].ty, "HostAffineClosure");
        let mut metadata = KernelObjectMetadata::default();
        metadata.args.insert(
            "f".to_string(),
            KernelArgObjectMetadata {
                address_space: Some("global".to_string()),
                offset: Some(32),
                size: Some(8),
                value_kind: Some("global_buffer".to_string()),
            },
        );
        let binding = generate_kernel_binding(&kernels[0], &device_structs, Some(&metadata))
            .expect("binding should generate");
        assert!(binding.contains("pub unsafe fn apply_closure_host_affine_closure"));
        assert!(binding.contains("f: &rocm_oxide::DeviceBuffer<HostAffineClosure>"));
        assert!(binding.contains("rocm_oxide::validate_buffer_len(\"f\", f.len(), 1)?;"));
        assert!(binding.contains("let mut __arg4 = f.as_ptr();"));
        assert!(binding.contains("let mut __arg5 = n;"));
        assert!(!binding.contains("f.base"));
    }

    #[test]
    fn generic_helpers_can_be_wrapped_by_monomorphic_kernels() {
        let kernels = discover_kernels_in_source(
            r#"
unsafe fn copy_generic<T: Copy>(out: *mut T, input: *const T, i: usize) {
    unsafe { *out.add(i) = *input.add(i); }
}

#[kernel]
pub unsafe extern "C" fn copy_u32(out: *mut u32, input: *const u32, n: usize) {}
"#,
        )
        .expect("monomorphic wrapper should parse");
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "copy_u32");
    }

    #[test]
    fn emits_repr_c_device_structs_for_captured_environment_abi() {
        let structs = discover_device_structs_in_source(
            r#"
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AffineParams {
    pub scale: f32,
    pub bias: f32,
}
"#,
        )
        .expect("repr C struct should parse");
        assert_eq!(structs.len(), 1);
        let binding = generate_device_struct_binding(&structs[0]);
        assert!(binding.contains("#[repr(C)]"));
        assert!(binding.contains("pub struct AffineParams"));
        assert!(binding.contains("pub scale: f32"));
        assert!(binding.contains("pub bias: f32"));
        assert!(binding.contains("std::mem::size_of::<AffineParams>() == 8"));
        assert!(binding.contains("std::mem::offset_of!(AffineParams, scale) == 0"));
        assert!(binding.contains("std::mem::offset_of!(AffineParams, bias) == 4"));
    }

    #[test]
    fn emits_default_repr_rust_device_struct_layout_assertions() {
        let structs = discover_device_structs_in_source(
            r#"
#[derive(Clone, Copy)]
pub struct RustLayoutParams {
    pub base: u32,
    pub stride: u32,
}
"#,
        )
        .expect("default repr Rust struct should parse");
        assert_eq!(structs.len(), 1);
        assert_eq!(structs[0].repr, super::DeviceStructRepr::Rust);
        let binding = generate_device_struct_binding(&structs[0]);
        assert!(!binding.contains("#[repr(C)]"));
        assert!(binding.contains("pub struct RustLayoutParams"));
        assert!(binding.contains("std::mem::size_of::<RustLayoutParams>() == 8"));
        assert!(binding.contains("std::mem::align_of::<RustLayoutParams>() == 4"));
        assert!(binding.contains(
            "std::mem::offset_of!(RustLayoutParams, base) == 0"
        ));
        assert!(binding.contains(
            "std::mem::offset_of!(RustLayoutParams, stride) == 4"
        ));
    }

    #[test]
    fn emits_pointer_fields_for_host_visible_reference_closures() {
        let structs = discover_device_structs_in_source(
            r#"
#[derive(Clone, Copy)]
pub struct HostReferenceClosure {
    pub bias: *const u32,
    pub scale: u32,
}
"#,
        )
        .expect("pointer-bearing closure environment should parse");
        assert_eq!(structs.len(), 1);
        let binding = generate_device_struct_binding(&structs[0]);
        assert!(binding.contains("pub struct HostReferenceClosure"));
        assert!(binding.contains("pub bias: *const u32"));
        assert!(binding.contains("std::mem::size_of::<HostReferenceClosure>() == 16"));
        assert!(binding.contains("std::mem::align_of::<HostReferenceClosure>() == 8"));
        assert!(binding.contains(
            "std::mem::offset_of!(HostReferenceClosure, bias) == 0"
        ));
        assert!(binding.contains(
            "std::mem::offset_of!(HostReferenceClosure, scale) == 8"
        ));
        assert!(!binding.contains("unsafe impl Send for HostReferenceClosure"));
        assert!(!binding.contains("unsafe impl Sync for HostReferenceClosure"));
    }

    #[test]
    fn omits_operations_for_by_value_raw_pointer_kernel_args() {
        let source = r#"
#[derive(Clone, Copy)]
pub struct HostReferenceClosure {
    pub bias: *const u32,
    pub scale: u32,
}

#[kernel]
pub unsafe extern "C" fn reference_probe(
    out: gpu::DeviceSliceMut<u32>,
    f: HostReferenceClosure,
    n: usize,
) {}
"#;
        let kernels = discover_kernels_in_source(source).expect("source should parse");
        let device_structs = discover_device_structs_in_source(source)
            .expect("pointer-bearing struct should parse")
            .into_iter()
            .map(|device_struct| (device_struct.name.clone(), device_struct))
            .collect::<BTreeMap<_, _>>();
        let binding = generate_kernel_binding(&kernels[0], &device_structs, None)
            .expect("binding should generate");
        assert!(binding.contains("pub unsafe fn reference_probe"));
        assert!(binding.contains("f: HostReferenceClosure"));
        assert!(binding.contains("let mut __arg2 = f.bias;"));
        assert!(binding.contains("reference_probe_operation is intentionally omitted"));
        assert!(!binding.contains("pub unsafe fn reference_probe_operation"));
    }

    #[test]
    fn rejects_unsupported_device_struct_repr_attributes() {
        let err = discover_device_structs_in_source(
            r#"
#[repr(C, align(16))]
pub struct OverAligned {
    pub value: u32,
}
"#,
        )
        .expect_err("unsupported repr attributes should be rejected");
        assert!(err.contains("unsupported repr(C, align(16))"));
    }

    #[test]
    fn parses_rustc_layout_offsets_and_padding() {
        let names = BTreeSet::from(["RustLayoutParams".to_string()]);
        let layouts = super::parse_rustc_type_size_layouts(
            r#"
print-type-size type: `RustLayoutParams`: 12 bytes, alignment: 4 bytes
print-type-size     field `.base`: 4 bytes
print-type-size     padding: 4 bytes
print-type-size     field `.stride`: 4 bytes
"#,
            &names,
        );
        let layout = layouts
            .get("RustLayoutParams")
            .expect("target layout should parse");
        assert_eq!(layout.size, 12);
        assert_eq!(layout.align, 4);
        assert_eq!(layout.fields[0].name, "base");
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].name, "stride");
        assert_eq!(layout.fields[1].offset, 8);
        assert_eq!(layout.padding[0].offset, 4);
        assert_eq!(layout.padding[0].size, 4);
    }

    #[test]
    fn scalarizes_known_repr_c_struct_launch_args() {
        let source = r#"
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ControlParams {
    pub seed: u32,
    pub scale: i32,
}

#[kernel]
pub unsafe extern "C" fn probe(
    out: gpu::DeviceSliceMut<u32>,
    params: ControlParams,
    n: usize,
) {}
"#;
        let kernels = discover_kernels_in_source(source).expect("source should parse");
        let device_structs = discover_device_structs_in_source(source)
            .expect("repr C struct should parse")
            .into_iter()
            .map(|device_struct| (device_struct.name.clone(), device_struct))
            .collect::<BTreeMap<_, _>>();

        let binding = generate_kernel_binding(&kernels[0], &device_structs, None)
            .expect("binding should generate");
        assert!(binding.contains("params: ControlParams"));
        assert!(binding.contains("let mut __arg2 = params.seed;"));
        assert!(binding.contains("let mut __arg3 = params.scale;"));
        assert!(binding.contains("let mut __arg4 = n;"));
        assert!(!binding.contains("let mut __arg2 = params;"));
    }

    #[test]
    fn monomorphized_struct_kernel_methods_are_snake_case() {
        let source = r#"
#[derive(Clone, Copy)]
pub struct RustLayoutParams {
    pub base: u32,
    pub stride: u32,
}

#[kernel(monomorphize(RustLayoutParams))]
pub unsafe extern "C" fn probe<P>(
    out: gpu::DeviceSliceMut<u32>,
    params: P,
    n: usize,
) {}
"#;
        let kernels = discover_kernels_in_source(source).expect("source should parse");
        let device_structs = discover_device_structs_in_source(source)
            .expect("device struct should parse")
            .into_iter()
            .map(|device_struct| (device_struct.name.clone(), device_struct))
            .collect::<BTreeMap<_, _>>();

        let binding = generate_kernel_binding(&kernels[0], &device_structs, None)
            .expect("binding should generate");
        assert!(binding.contains("pub unsafe fn probe_rust_layout_params"));
        assert!(!binding.contains("probe_RustLayoutParams"));
    }

    #[test]
    fn rejects_unknown_by_value_struct_launch_args() {
        let source = r#"
#[kernel]
pub unsafe extern "C" fn probe(
    out: gpu::DeviceSliceMut<u32>,
    params: MissingLayout,
    n: usize,
) {}
"#;
        let kernels = discover_kernels_in_source(source).expect("source should parse");
        let err = generate_kernel_binding(&kernels[0], &BTreeMap::new(), None)
            .expect_err("unknown by-value struct should be rejected");
        assert!(err.contains("unsupported by-value kernel argument `params`"));
        assert!(err.contains("MissingLayout"));
    }

    #[test]
    fn catches_internal_compiler_panics() {
        let err = compiler_step::<(), _>("rewrite test IR", || panic!("boom"))
            .expect_err("panic should be converted into a diagnostic");
        assert!(err.contains("internal compiler panic"));
        assert!(err.contains("rewrite test IR"));
        assert!(err.contains("boom"));
    }

    #[test]
    fn parses_inline_path_dependencies() {
        let dep = parse_inline_path_dependency(
            r#"shared-kernels = { path = "../shared-kernels", version = "0.1" }"#,
        )
        .expect("path dependency should parse");
        assert_eq!(dep, Path::new("../shared-kernels"));
        assert!(parse_package_name("[package]\nname = \"gpu-kernels\"\n").is_some());
    }

    #[test]
    fn discovers_cross_crate_kernel_bundle_members() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be valid")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rocm-oxide-build-test-{suffix}"));
        let app = root.join("app");
        let shared = root.join("shared");
        fs::create_dir_all(app.join("src")).expect("create app src");
        fs::create_dir_all(shared.join("src")).expect("create shared src");
        fs::write(
            app.join("Cargo.toml"),
            r#"[package]
name = "app"
version = "0.1.0"

[dependencies]
shared = { path = "../shared" }
"#,
        )
        .expect("write app manifest");
        fs::write(app.join("src/lib.rs"), "#![no_std]\n").expect("write app source");
        fs::write(
            shared.join("Cargo.toml"),
            r#"[package]
name = "shared"
version = "0.1.0"
"#,
        )
        .expect("write shared manifest");
        fs::write(
            shared.join("src/lib.rs"),
            r#"
#[kernel]
pub unsafe extern "C" fn shared_kernel(out: *mut u32, n: usize) {}
"#,
        )
        .expect("write shared source");

        let bundle = discover_device_crate_bundle(&app).expect("bundle discovery should work");
        assert_eq!(bundle.len(), 2);
        assert!(bundle.iter().any(|path| path.ends_with("app")));
        assert!(bundle.iter().any(|path| path.ends_with("shared")));

        let _ = fs::remove_dir_all(root);
    }
}
