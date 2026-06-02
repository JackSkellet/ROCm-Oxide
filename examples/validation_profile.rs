use rocm_oxide::{
    Device, DeviceProperties, LibraryAvailability, MatrixIntegrationReport, RocmLibraryReport,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> rocm_oxide::Result<()> {
    let args = Args::parse().map_err(rocm_oxide::Error::InvalidLaunch)?;
    let device = Device::first()?;
    let device_count = Device::count()?;
    let properties = device.properties()?;
    let libraries = RocmLibraryReport::query();
    let matrix = MatrixIntegrationReport::query();
    let rocminfo = RocminfoSummary::query();
    let skipped_tests = skipped_tests(device_count, properties, &libraries, &matrix);
    let known_profile = KnownProfileReport::for_current(device.arch(), properties);

    println!("ROCm-Oxide validation profile on {}", device.arch());
    println!("devices: {device_count}");
    println!(
        "known profile: {} ({})",
        known_profile.label,
        if known_profile.deviations.is_empty() {
            "matched"
        } else {
            "deviations recorded"
        }
    );
    println!("skip reasons recorded: {}", skipped_tests.len());

    if let Some(path) = args.json {
        write_json_report(
            &path,
            device.arch(),
            device_count,
            properties,
            &libraries,
            &matrix,
            &rocminfo,
            &known_profile,
            &skipped_tests,
        )
        .map_err(rocm_oxide::Error::Io)?;
        println!("wrote {}", path.display());
    }

    Ok(())
}

#[derive(Debug, Default)]
struct Args {
    json: Option<PathBuf>,
}

impl Args {
    fn parse() -> std::result::Result<Self, String> {
        let mut args = Self::default();
        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            if arg == "--json" {
                args.json = Some(
                    iter.next()
                        .map(PathBuf::from)
                        .ok_or_else(|| "--json requires an output path".to_string())?,
                );
            } else if let Some(path) = arg.strip_prefix("--json=") {
                args.json = Some(PathBuf::from(path));
            } else if arg == "--help" || arg == "-h" {
                println!(
                    "Usage: cargo run --example validation_profile -- [--json target/validation_profile.json]"
                );
                std::process::exit(0);
            } else {
                return Err(format!("unknown argument: {arg}"));
            }
        }
        Ok(args)
    }
}

#[derive(Debug, Clone)]
struct SkippedTest {
    name: &'static str,
    reason: String,
}

#[derive(Debug, Clone)]
struct KnownProfileReport {
    arch: String,
    label: String,
    known: bool,
    deviations: Vec<String>,
}

impl KnownProfileReport {
    fn for_current(arch: &str, properties: DeviceProperties) -> Self {
        let Some(profile) = KnownProfile::for_arch(arch) else {
            return Self {
                arch: arch.to_string(),
                label: "unrecognized validation profile".to_string(),
                known: false,
                deviations: vec![format!(
                    "no checked ROCm-Oxide baseline has been recorded for architecture `{arch}`"
                )],
            };
        };

        let mut deviations = Vec::new();
        for expected in profile.expected_flags {
            if property_flag(properties, expected.name) != expected.value {
                deviations.push(format!(
                    "{} expected {}, observed {}",
                    expected.name,
                    expected.value,
                    property_flag(properties, expected.name)
                ));
            }
        }
        if properties.warp_size != profile.expected_wavefront_size {
            deviations.push(format!(
                "warp_size expected {}, observed {}",
                profile.expected_wavefront_size, properties.warp_size
            ));
        }

        Self {
            arch: arch.to_string(),
            label: profile.label.to_string(),
            known: true,
            deviations,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct KnownProfile {
    label: &'static str,
    expected_wavefront_size: u32,
    expected_flags: &'static [ExpectedFlag],
}

#[derive(Debug, Clone, Copy)]
struct ExpectedFlag {
    name: &'static str,
    value: bool,
}

impl KnownProfile {
    fn for_arch(arch: &str) -> Option<Self> {
        const GFX1201: &[ExpectedFlag] = &[
            ExpectedFlag {
                name: "managed_memory",
                value: true,
            },
            ExpectedFlag {
                name: "concurrent_managed_access",
                value: true,
            },
            ExpectedFlag {
                name: "can_map_host_memory",
                value: true,
            },
            ExpectedFlag {
                name: "host_register_supported",
                value: true,
            },
            ExpectedFlag {
                name: "memory_pools_supported",
                value: true,
            },
            ExpectedFlag {
                name: "unified_addressing",
                value: true,
            },
            ExpectedFlag {
                name: "host_native_atomic_supported",
                value: true,
            },
            ExpectedFlag {
                name: "can_use_host_pointer_for_registered_mem",
                value: false,
            },
        ];
        const GFX1100: &[ExpectedFlag] = &[
            ExpectedFlag {
                name: "managed_memory",
                value: true,
            },
            ExpectedFlag {
                name: "concurrent_managed_access",
                value: true,
            },
            ExpectedFlag {
                name: "can_map_host_memory",
                value: true,
            },
            ExpectedFlag {
                name: "host_register_supported",
                value: true,
            },
            ExpectedFlag {
                name: "memory_pools_supported",
                value: true,
            },
            ExpectedFlag {
                name: "unified_addressing",
                value: true,
            },
            ExpectedFlag {
                name: "host_native_atomic_supported",
                value: false,
            },
            ExpectedFlag {
                name: "direct_managed_mem_access_from_host",
                value: false,
            },
            ExpectedFlag {
                name: "can_use_host_pointer_for_registered_mem",
                value: false,
            },
            ExpectedFlag {
                name: "pageable_memory_access",
                value: false,
            },
        ];

        match arch {
            "gfx1201" => Some(Self {
                label: "RX 9070 XT validation baseline",
                expected_wavefront_size: 32,
                expected_flags: GFX1201,
            }),
            "gfx1100" => Some(Self {
                label: "RX 7900 XT validation baseline",
                expected_wavefront_size: 32,
                expected_flags: GFX1100,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct RocminfoSummary {
    tool: Option<String>,
    runtime_version: Option<String>,
    detected_arch: Option<String>,
}

impl RocminfoSummary {
    fn query() -> Self {
        let Some(path) = rocminfo_path() else {
            return Self {
                tool: None,
                runtime_version: None,
                detected_arch: None,
            };
        };
        let Ok(output) = Command::new(&path).output() else {
            return Self {
                tool: Some(path.display().to_string()),
                runtime_version: None,
                detected_arch: None,
            };
        };
        if !output.status.success() {
            return Self {
                tool: Some(path.display().to_string()),
                runtime_version: None,
                detected_arch: None,
            };
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        Self {
            tool: Some(path.display().to_string()),
            runtime_version: stdout.lines().find_map(|line| {
                let (_, value) = line.split_once("Runtime Version:")?;
                Some(value.trim().to_string())
            }),
            detected_arch: stdout.lines().find_map(|line| {
                let (_, value) = line.split_once("Name:")?;
                let value = value.trim();
                (value.starts_with("gfx") && !value.contains('-')).then(|| value.to_string())
            }),
        }
    }
}

fn rocminfo_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("ROCMINFO").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(path));
    }
    if let Some(root) = env::var_os("ROCM_PATH").filter(|value| !value.is_empty()) {
        let path = PathBuf::from(root).join("bin/rocminfo");
        if path.is_file() {
            return Some(path);
        }
    }
    let path = PathBuf::from("/opt/rocm/bin/rocminfo");
    if path.is_file() {
        return Some(path);
    }
    Some(PathBuf::from("rocminfo"))
}

fn skipped_tests(
    device_count: i32,
    properties: DeviceProperties,
    libraries: &RocmLibraryReport,
    matrix: &MatrixIntegrationReport,
) -> Vec<SkippedTest> {
    let mut skipped = Vec::new();
    if !(properties.can_map_host_memory && properties.host_native_atomic_supported) {
        skipped.push(SkippedTest {
            name: "mapped_host_visible_system_scope_atomics",
            reason: "requires mapped host memory and host-native atomics".to_string(),
        });
    }
    if !(properties.managed_memory
        && properties.concurrent_managed_access
        && properties.host_native_atomic_supported)
    {
        skipped.push(SkippedTest {
            name: "managed_fine_grain_concurrent_system_scope_atomics",
            reason: "requires managed memory, concurrent managed access, and host-native atomics"
                .to_string(),
        });
    }
    if !properties.direct_managed_mem_access_from_host {
        skipped.push(SkippedTest {
            name: "direct_host_access_to_managed_device_memory",
            reason: "HIP did not report direct managed-memory access from host".to_string(),
        });
    }
    if !properties.can_use_host_pointer_for_registered_mem {
        skipped.push(SkippedTest {
            name: "registered_host_pointer_reuse",
            reason: "HIP did not report registered host-pointer reuse".to_string(),
        });
    }
    if !properties.pageable_memory_access {
        skipped.push(SkippedTest {
            name: "pageable_memory_access",
            reason: "HIP did not report pageable-memory access".to_string(),
        });
    }
    if !properties.memory_pools_supported {
        skipped.push(SkippedTest {
            name: "stream_ordered_memory_pool_paths",
            reason: "HIP did not report memory-pool support".to_string(),
        });
    }
    if device_count < 2 || !properties.cooperative_multi_device_launch {
        skipped.push(SkippedTest {
            name: "cooperative_multi_device_launch",
            reason: format!(
                "requires at least two visible devices and cooperative multi-device support; devices={device_count}, support={}",
                properties.cooperative_multi_device_launch
            ),
        });
    }
    push_library_skip(&mut skipped, "rocblas_interop", &libraries.rocblas);
    push_library_skip(&mut skipped, "rocfft_interop", &libraries.rocfft);
    push_library_skip(&mut skipped, "rocprim_hipcub_interop", &libraries.rocprim);
    push_library_skip(&mut skipped, "comgr_compile_backend", &libraries.comgr);
    push_library_skip(
        &mut skipped,
        "hipblaslt_matrix_heuristics",
        &matrix.hipblaslt,
    );
    push_library_skip(
        &mut skipped,
        "composable_kernel_matrix_backend",
        &matrix.composable_kernel,
    );
    push_library_skip(&mut skipped, "rocwmma_matrix_backend", &matrix.rocwmma);
    skipped
}

fn push_library_skip(
    skipped: &mut Vec<SkippedTest>,
    name: &'static str,
    lib: &LibraryAvailability,
) {
    if !lib.available {
        skipped.push(SkippedTest {
            name,
            reason: lib.detail.clone(),
        });
    }
}

fn property_flag(properties: DeviceProperties, name: &str) -> bool {
    match name {
        "managed_memory" => properties.managed_memory,
        "concurrent_managed_access" => properties.concurrent_managed_access,
        "cooperative_launch" => properties.cooperative_launch,
        "cooperative_multi_device_launch" => properties.cooperative_multi_device_launch,
        "direct_managed_mem_access_from_host" => properties.direct_managed_mem_access_from_host,
        "can_map_host_memory" => properties.can_map_host_memory,
        "can_use_host_pointer_for_registered_mem" => {
            properties.can_use_host_pointer_for_registered_mem
        }
        "host_native_atomic_supported" => properties.host_native_atomic_supported,
        "pageable_memory_access" => properties.pageable_memory_access,
        "pageable_memory_access_uses_host_page_tables" => {
            properties.pageable_memory_access_uses_host_page_tables
        }
        "memory_pools_supported" => properties.memory_pools_supported,
        "unified_addressing" => properties.unified_addressing,
        "host_register_supported" => properties.host_register_supported,
        _ => false,
    }
}

fn write_json_report(
    path: &Path,
    arch: &str,
    device_count: i32,
    properties: DeviceProperties,
    libraries: &RocmLibraryReport,
    matrix: &MatrixIntegrationReport,
    rocminfo: &RocminfoSummary,
    known_profile: &KnownProfileReport,
    skipped_tests: &[SkippedTest],
) -> std::io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"format\": \"rocm-oxide-validation-profile-v1\",\n");
    out.push_str(&format!(
        "  \"selected_arch\": \"{}\",\n",
        json_escape(arch)
    ));
    out.push_str(&format!("  \"device_count\": {device_count},\n"));
    write_rocminfo_json(&mut out, rocminfo);
    out.push_str(",\n");
    write_properties_json(&mut out, properties);
    out.push_str(",\n");
    write_known_profile_json(&mut out, known_profile);
    out.push_str(",\n");
    write_library_report_json(&mut out, "libraries", libraries);
    out.push_str(",\n");
    write_matrix_report_json(&mut out, "matrix_integrations", matrix);
    out.push_str(",\n  \"skipped_tests\": [\n");
    for (index, skip) in skipped_tests.iter().enumerate() {
        if index > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "    {{ \"name\": \"{}\", \"reason\": \"{}\" }}",
            json_escape(skip.name),
            json_escape(&skip.reason)
        ));
    }
    out.push_str("\n  ]\n");
    out.push_str("}\n");
    fs::write(path, out)
}

fn write_rocminfo_json(out: &mut String, rocminfo: &RocminfoSummary) {
    out.push_str("  \"rocminfo\": {\n");
    write_json_opt_str(out, "tool", rocminfo.tool.as_deref(), true);
    write_json_opt_str(
        out,
        "runtime_version",
        rocminfo.runtime_version.as_deref(),
        false,
    );
    write_json_opt_str(
        out,
        "detected_arch",
        rocminfo.detected_arch.as_deref(),
        false,
    );
    out.push_str("\n  }");
}

fn write_properties_json(out: &mut String, properties: DeviceProperties) {
    out.push_str("  \"properties\": {\n");
    out.push_str(&format!("    \"ordinal\": {},\n", properties.ordinal));
    write_bool(out, "managed_memory", properties.managed_memory, true);
    write_bool(
        out,
        "concurrent_managed_access",
        properties.concurrent_managed_access,
        false,
    );
    write_bool(
        out,
        "cooperative_launch",
        properties.cooperative_launch,
        false,
    );
    write_bool(
        out,
        "cooperative_multi_device_launch",
        properties.cooperative_multi_device_launch,
        false,
    );
    write_bool(
        out,
        "direct_managed_mem_access_from_host",
        properties.direct_managed_mem_access_from_host,
        false,
    );
    write_bool(
        out,
        "can_map_host_memory",
        properties.can_map_host_memory,
        false,
    );
    write_bool(
        out,
        "can_use_host_pointer_for_registered_mem",
        properties.can_use_host_pointer_for_registered_mem,
        false,
    );
    write_bool(
        out,
        "host_native_atomic_supported",
        properties.host_native_atomic_supported,
        false,
    );
    write_bool(
        out,
        "pageable_memory_access",
        properties.pageable_memory_access,
        false,
    );
    write_bool(
        out,
        "pageable_memory_access_uses_host_page_tables",
        properties.pageable_memory_access_uses_host_page_tables,
        false,
    );
    write_bool(
        out,
        "memory_pools_supported",
        properties.memory_pools_supported,
        false,
    );
    write_bool(
        out,
        "unified_addressing",
        properties.unified_addressing,
        false,
    );
    write_bool(
        out,
        "host_register_supported",
        properties.host_register_supported,
        false,
    );
    out.push_str(",\n");
    out.push_str(&format!(
        "    \"async_engine_count\": {},\n",
        properties.async_engine_count
    ));
    out.push_str(&format!(
        "    \"multiprocessor_count\": {},\n",
        properties.multiprocessor_count
    ));
    out.push_str(&format!("    \"warp_size\": {},\n", properties.warp_size));
    out.push_str(&format!(
        "    \"clock_instruction_rate_khz\": {},\n",
        properties.clock_instruction_rate_khz
    ));
    out.push_str(&format!(
        "    \"wall_clock_rate_khz\": {}\n",
        properties.wall_clock_rate_khz
    ));
    out.push_str("  }");
}

fn write_known_profile_json(out: &mut String, profile: &KnownProfileReport) {
    out.push_str("  \"known_profile\": {\n");
    out.push_str(&format!(
        "    \"arch\": \"{}\",\n",
        json_escape(&profile.arch)
    ));
    out.push_str(&format!(
        "    \"label\": \"{}\",\n",
        json_escape(&profile.label)
    ));
    out.push_str(&format!("    \"known\": {},\n", profile.known));
    out.push_str("    \"deviations\": ");
    write_string_array(out, &profile.deviations);
    out.push_str("\n  }");
}

fn write_library_report_json(out: &mut String, key: &str, report: &RocmLibraryReport) {
    out.push_str(&format!("  \"{key}\": {{\n"));
    write_availability(out, "rocblas", &report.rocblas, true);
    write_availability(out, "rocfft", &report.rocfft, false);
    write_availability(out, "hipblaslt", &report.hipblaslt, false);
    write_availability(out, "comgr", &report.comgr, false);
    write_availability(out, "rocprim", &report.rocprim, false);
    out.push_str("\n  }");
}

fn write_matrix_report_json(out: &mut String, key: &str, report: &MatrixIntegrationReport) {
    out.push_str(&format!("  \"{key}\": {{\n"));
    write_availability(out, "hipblaslt", &report.hipblaslt, true);
    write_availability(out, "composable_kernel", &report.composable_kernel, false);
    write_availability(out, "rocwmma", &report.rocwmma, false);
    out.push_str("\n  }");
}

fn write_availability(out: &mut String, key: &str, lib: &LibraryAvailability, first: bool) {
    if !first {
        out.push_str(",\n");
    }
    out.push_str(&format!(
        "    \"{key}\": {{ \"available\": {}, \"detail\": \"{}\" }}",
        lib.available,
        json_escape(&lib.detail)
    ));
}

fn write_bool(out: &mut String, key: &str, value: bool, first: bool) {
    if !first {
        out.push_str(",\n");
    }
    out.push_str(&format!("    \"{key}\": {value}"));
}

fn write_json_opt_str(out: &mut String, key: &str, value: Option<&str>, first: bool) {
    if !first {
        out.push_str(",\n");
    }
    match value {
        Some(value) => out.push_str(&format!("    \"{key}\": \"{}\"", json_escape(value))),
        None => out.push_str(&format!("    \"{key}\": null")),
    }
}

fn write_string_array(out: &mut String, values: &[String]) {
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("\"{}\"", json_escape(value)));
    }
    out.push(']');
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
