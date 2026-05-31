use rocm_oxide::{Device, DeviceBuffer, Event, LaunchConfig, Result, Stream};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const BLOCK_X: u32 = 256;
const VECTOR_N: usize = 1 << 20;
const WIDTH: usize = 1024;
const HEIGHT: usize = 576;
const PIXELS: usize = WIDTH * HEIGHT;

fn main() -> Result<()> {
    let args = Args::parse().map_err(rocm_oxide::Error::InvalidLaunch)?;
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let resources = KernelResources::load_embedded();
    let mut samples = Vec::new();

    println!("ROCm-Oxide GPU performance probe on {}", device.arch());
    println!(
        "{:<30} {:>10} {:>10} {:>12}",
        "kernel", "gpu ms", "est FPS", "notes"
    );
    println!("{:-<68}", "");

    probe_vector_add(&kernels, &resources, &mut samples)?;
    probe_affine_transform(&kernels, &resources, &mut samples)?;
    probe_stress_pattern(&kernels, &resources, &mut samples)?;
    probe_stress_3d(&kernels, &resources, &mut samples)?;
    probe_raytrace_world(&kernels, &resources, &mut samples)?;

    if let Some(path) = args.json {
        write_json_report(&path, device.arch(), &samples).map_err(rocm_oxide::Error::Io)?;
        println!();
        println!("wrote {}", path.display());
    }

    Ok(())
}

fn probe_vector_add(
    kernels: &generated::DeviceKernels,
    resources: &KernelResources,
    samples: &mut Vec<ProbeSample>,
) -> Result<()> {
    let a = (0..VECTOR_N).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..VECTOR_N)
        .map(|i| (VECTOR_N - i) as f32)
        .collect::<Vec<_>>();
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(VECTOR_N)?;
    let config = LaunchConfig::for_num_elems_with_block_size(VECTOR_N, BLOCK_X);

    let ms = time_gpu_ms(64, || unsafe {
        kernels.vector_add(config, &d_out, &d_a, &d_b)
    })?;
    record_sample(
        samples,
        resources,
        "vector_add",
        "vector_add 1M f32",
        ms,
        "typed binding",
    );
    Ok(())
}

fn probe_affine_transform(
    kernels: &generated::DeviceKernels,
    resources: &KernelResources,
    samples: &mut Vec<ProbeSample>,
) -> Result<()> {
    let input = (0..VECTOR_N).map(|i| i as f32 * 0.25).collect::<Vec<_>>();
    let d_input = DeviceBuffer::from_slice(&input)?;
    let d_out = DeviceBuffer::<f32>::new(VECTOR_N)?;
    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 1.618,
        bias: -0.125,
    }])?;
    let config = LaunchConfig::for_num_elems_with_block_size(VECTOR_N, BLOCK_X);

    let ms = time_gpu_ms(64, || unsafe {
        kernels.affine_transform(config, &d_out, &d_input, &params)
    })?;
    record_sample(
        samples,
        resources,
        "affine_transform",
        "affine repr(C) 1M",
        ms,
        "env struct",
    );
    Ok(())
}

fn probe_stress_3d(
    kernels: &generated::DeviceKernels,
    resources: &KernelResources,
    samples: &mut Vec<ProbeSample>,
) -> Result<()> {
    let frame = DeviceBuffer::<u32>::new(PIXELS)?;
    let config = LaunchConfig::for_num_elems_with_block_size(PIXELS, BLOCK_X);

    for steps in [32u32, 96, 256, 1024, 3000] {
        let ms = time_gpu_ms(iterations_for_ms_probe(steps), || unsafe {
            kernels.stress_3d(config, &frame, PIXELS, 17, 2, steps)
        })?;
        record_sample(
            samples,
            resources,
            "stress_3d",
            &format!("stress_3d steps={steps}"),
            ms,
            if steps >= 1024 {
                "heavy ALU"
            } else {
                "3D stress"
            },
        );
    }
    Ok(())
}

fn probe_stress_pattern(
    kernels: &generated::DeviceKernels,
    resources: &KernelResources,
    samples: &mut Vec<ProbeSample>,
) -> Result<()> {
    let frame = DeviceBuffer::<u32>::new(PIXELS)?;
    let config = LaunchConfig::for_num_elems_with_block_size(PIXELS, BLOCK_X);

    for steps in [32u32, 96, 256, 1024, 3000] {
        let ms = time_gpu_ms(iterations_for_ms_probe(steps), || unsafe {
            kernels.stress_pattern(config, &frame, PIXELS, 17, 5, steps)
        })?;
        record_sample(
            samples,
            resources,
            "stress_pattern",
            "stress_pattern",
            ms,
            &format!("steps={steps}"),
        );
    }
    Ok(())
}

fn probe_raytrace_world(
    kernels: &generated::DeviceKernels,
    resources: &KernelResources,
    samples: &mut Vec<ProbeSample>,
) -> Result<()> {
    let frame = DeviceBuffer::<u32>::new(PIXELS)?;
    let camera = DeviceBuffer::from_slice(&[
        0.0f32, 0.28, -1.6, // position
        1.0, 0.0, 0.0, // right
        0.0, 1.0, 0.0, // up
        0.0, 0.0, 1.0, // forward
        3.0, // shadows + reflections
    ])?;
    let config = LaunchConfig::for_num_elems_with_block_size(PIXELS, BLOCK_X);

    let ms = time_gpu_ms(24, || unsafe {
        kernels.raytrace_world(config, &frame, &camera, PIXELS, 17)
    })?;
    record_sample(
        samples,
        resources,
        "raytrace_world",
        "raytrace_world 1024x576",
        ms,
        "camera scene",
    );
    Ok(())
}

fn time_gpu_ms<F>(iterations: usize, mut launch: F) -> Result<f32>
where
    F: FnMut() -> Result<()>,
{
    let stream = Stream::null();
    rocm_oxide::hip::synchronize()?;
    launch()?;
    rocm_oxide::hip::synchronize()?;

    let start = Event::new()?;
    let stop = Event::new()?;
    start.record(&stream)?;
    for _ in 0..iterations {
        launch()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;

    Ok(start.elapsed_ms_until(&stop)? / iterations as f32)
}

fn iterations_for_ms_probe(steps: u32) -> usize {
    match steps {
        0..=128 => 48,
        129..=512 => 24,
        513..=1500 => 12,
        _ => 6,
    }
}

fn print_row(name: &str, ms: f32, notes: &str) {
    println!(
        "{:<30} {:>10.3} {:>10.1} {:>12}",
        name,
        ms,
        1000.0 / ms.max(0.001),
        notes
    );
}

fn record_sample(
    samples: &mut Vec<ProbeSample>,
    resources: &KernelResources,
    kernel: &str,
    label: &str,
    ms: f32,
    notes: &str,
) {
    print_row(label, ms, notes);
    samples.push(ProbeSample {
        kernel: kernel.to_string(),
        label: label.to_string(),
        gpu_ms: ms,
        est_fps: 1000.0 / ms.max(0.001),
        notes: notes.to_string(),
        resources: resources.by_kernel.get(kernel).cloned(),
    });
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
                    "Usage: cargo run --example performance_probe -- [--json target/perf.json]"
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
struct ProbeSample {
    kernel: String,
    label: String,
    gpu_ms: f32,
    est_fps: f32,
    notes: String,
    resources: Option<KernelResource>,
}

#[derive(Debug, Default)]
struct KernelResources {
    by_kernel: BTreeMap<String, KernelResource>,
}

impl KernelResources {
    fn load_embedded() -> Self {
        let text = fs::read_to_string(env!("ROCM_OXIDE_DEVICE_METADATA")).unwrap_or_default();
        Self {
            by_kernel: parse_kernel_resources(&text),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct KernelResource {
    kernarg_segment_size: Option<u32>,
    max_flat_workgroup_size: Option<u32>,
    group_segment_fixed_size: Option<u32>,
    private_segment_fixed_size: Option<u32>,
    sgpr_count: Option<u32>,
    vgpr_count: Option<u32>,
    sgpr_spill_count: Option<u32>,
    vgpr_spill_count: Option<u32>,
    wavefront_size: Option<u32>,
    uses_dynamic_stack: Option<bool>,
}

fn parse_kernel_resources(text: &str) -> BTreeMap<String, KernelResource> {
    let mut resources = BTreeMap::new();
    let mut name: Option<String> = None;
    let mut current = KernelResource::default();
    let mut in_code_object = false;

    for line in text.lines() {
        if line.starts_with("      \"name\":") {
            if let Some(name) = name.take() {
                resources.insert(name, std::mem::take(&mut current));
            }
            name = find_json_string(line, "name");
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

        parse_resource_field(&mut current, trimmed);
    }

    if let Some(name) = name {
        resources.insert(name, current);
    }
    resources
}

fn parse_resource_field(resource: &mut KernelResource, line: &str) {
    if let Some(value) = json_u32_field(line, "kernarg_segment_size") {
        resource.kernarg_segment_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "max_flat_workgroup_size") {
        resource.max_flat_workgroup_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "group_segment_fixed_size") {
        resource.group_segment_fixed_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "private_segment_fixed_size") {
        resource.private_segment_fixed_size = Some(value);
    } else if let Some(value) = json_u32_field(line, "sgpr_count") {
        resource.sgpr_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "vgpr_count") {
        resource.vgpr_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "sgpr_spill_count") {
        resource.sgpr_spill_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "vgpr_spill_count") {
        resource.vgpr_spill_count = Some(value);
    } else if let Some(value) = json_u32_field(line, "wavefront_size") {
        resource.wavefront_size = Some(value);
    } else if let Some(value) = json_bool_field(line, "uses_dynamic_stack") {
        resource.uses_dynamic_stack = Some(value);
    }
}

fn write_json_report(path: &PathBuf, arch: &str, samples: &[ProbeSample]) -> std::io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"format\": \"rocm-oxide-performance-probe-v1\",\n");
    out.push_str(&format!("  \"arch\": \"{}\",\n", json_escape(arch)));
    out.push_str(&format!(
        "  \"metadata\": \"{}\",\n",
        json_escape(env!("ROCM_OXIDE_DEVICE_METADATA"))
    ));
    out.push_str(&format!("  \"width\": {WIDTH},\n"));
    out.push_str(&format!("  \"height\": {HEIGHT},\n"));
    out.push_str(&format!("  \"vector_n\": {VECTOR_N},\n"));
    out.push_str("  \"samples\": [\n");
    for (index, sample) in samples.iter().enumerate() {
        if index > 0 {
            out.push_str(",\n");
        }
        write_sample_json(&mut out, sample);
    }
    out.push_str("\n  ]\n");
    out.push_str("}\n");
    fs::write(path, out)
}

fn write_sample_json(out: &mut String, sample: &ProbeSample) {
    out.push_str("    {\n");
    out.push_str(&format!(
        "      \"kernel\": \"{}\",\n",
        json_escape(&sample.kernel)
    ));
    out.push_str(&format!(
        "      \"label\": \"{}\",\n",
        json_escape(&sample.label)
    ));
    out.push_str(&format!("      \"gpu_ms\": {:.6},\n", sample.gpu_ms));
    out.push_str(&format!("      \"est_fps\": {:.3},\n", sample.est_fps));
    out.push_str(&format!(
        "      \"notes\": \"{}\",\n",
        json_escape(&sample.notes)
    ));
    out.push_str("      \"resources\": ");
    if let Some(resources) = &sample.resources {
        write_resource_json(out, resources);
        out.push('\n');
    } else {
        out.push_str("null\n");
    }
    out.push_str("    }");
}

fn write_resource_json(out: &mut String, resources: &KernelResource) {
    out.push_str("{\n");
    write_json_u32(
        out,
        "kernarg_segment_size",
        resources.kernarg_segment_size,
        true,
    );
    write_json_u32(
        out,
        "max_flat_workgroup_size",
        resources.max_flat_workgroup_size,
        false,
    );
    write_json_u32(
        out,
        "group_segment_fixed_size",
        resources.group_segment_fixed_size,
        false,
    );
    write_json_u32(
        out,
        "private_segment_fixed_size",
        resources.private_segment_fixed_size,
        false,
    );
    write_json_u32(out, "sgpr_count", resources.sgpr_count, false);
    write_json_u32(out, "vgpr_count", resources.vgpr_count, false);
    write_json_u32(out, "sgpr_spill_count", resources.sgpr_spill_count, false);
    write_json_u32(out, "vgpr_spill_count", resources.vgpr_spill_count, false);
    write_json_u32(out, "wavefront_size", resources.wavefront_size, false);
    if let Some(value) = resources.uses_dynamic_stack {
        out.push_str(&format!(",\n        \"uses_dynamic_stack\": {value}"));
    }
    out.push_str("\n      }");
}

fn write_json_u32(out: &mut String, key: &str, value: Option<u32>, first: bool) {
    if let Some(value) = value {
        if !first {
            out.push_str(",\n");
        }
        out.push_str(&format!("        \"{key}\": {value}"));
    }
}

fn find_json_string(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\": \"");
    let start = text.find(&needle)? + needle.len();
    let rest = &text[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn json_u32_field(line: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\": ");
    line.trim()
        .strip_prefix(&needle)?
        .trim_end_matches(',')
        .parse::<u32>()
        .ok()
}

fn json_bool_field(line: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\": ");
    match line.trim().strip_prefix(&needle)?.trim_end_matches(',') {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
