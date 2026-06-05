use rocm_oxide::{Device, DeviceBuffer, Event, LaunchConfig, Result, Stream};
use std::collections::BTreeMap;
use std::env;
use std::ffi::CString;
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
const HIGH_VGPR_PRESSURE: u32 = 32;
const HIGH_SGPR_PRESSURE: u32 = 24;
const LOW_WAVES_PER_MULTIPROCESSOR: u32 = 16;

fn main() -> Result<()> {
    let args = Args::parse().map_err(rocm_oxide::Error::InvalidLaunch)?;
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let resources = KernelResources::load_embedded();
    let mut samples = Vec::new();

    println!("ROCm-Oxide GPU performance probe on {}", device.arch());
    println!(
        "{:<30} {:>10} {:>10} {:>14}  flags",
        "kernel", "gpu ms", "est FPS", "occupancy"
    );
    println!("{:-<92}", "");

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
        kernels,
        resources,
        "vector_add",
        "vector_add 1M f32",
        ms,
        "typed binding",
        config,
    )?;
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
        kernels,
        resources,
        "affine_transform",
        "affine repr(C) 1M",
        ms,
        "env struct",
        config,
    )?;
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
            kernels.stress_3d(config, &frame, 17, 2, steps)
        })?;
        record_sample(
            samples,
            kernels,
            resources,
            "stress_3d",
            &format!("stress_3d steps={steps}"),
            ms,
            if steps >= 1024 {
                "heavy ALU"
            } else {
                "3D stress"
            },
            config,
        )?;
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
            kernels.stress_pattern(config, &frame, 17, 5, steps)
        })?;
        record_sample(
            samples,
            kernels,
            resources,
            "stress_pattern",
            "stress_pattern",
            ms,
            &format!("steps={steps}"),
            config,
        )?;
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
        kernels.raytrace_world(config, &frame, &camera, 17)
    })?;
    record_sample(
        samples,
        kernels,
        resources,
        "raytrace_world",
        "raytrace_world 1024x576",
        ms,
        "camera scene",
        config,
    )?;
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

fn print_row(name: &str, ms: f32, occupancy: Option<&OccupancyReport>, flags: &[String]) {
    let occupancy = occupancy
        .map(OccupancyReport::summary)
        .unwrap_or_else(|| "-".to_string());
    let flags = if flags.is_empty() {
        "ok".to_string()
    } else {
        flags.join(",")
    };
    println!(
        "{:<30} {:>10.3} {:>10.1} {:>14}  {}",
        name,
        ms,
        1000.0 / ms.max(0.001),
        occupancy,
        flags
    );
}

fn record_sample(
    samples: &mut Vec<ProbeSample>,
    kernels: &generated::DeviceKernels,
    resources: &KernelResources,
    kernel: &str,
    label: &str,
    ms: f32,
    notes: &str,
    config: LaunchConfig,
) -> Result<()> {
    let resources = resources.by_kernel.get(kernel).copied();
    let occupancy = resources
        .map(|resource| query_occupancy(kernels, kernel, resource, config))
        .transpose()?;
    let limiters = detect_limiters(resources.as_ref(), occupancy.as_ref());
    print_row(label, ms, occupancy.as_ref(), &limiters);
    samples.push(ProbeSample {
        kernel: kernel.to_string(),
        label: label.to_string(),
        gpu_ms: ms,
        est_fps: 1000.0 / ms.max(0.001),
        notes: notes.to_string(),
        occupancy,
        limiters,
        resources,
    });
    Ok(())
}

fn query_occupancy(
    kernels: &generated::DeviceKernels,
    kernel: &str,
    resources: rocm_oxide::KernelResource,
    config: LaunchConfig,
) -> Result<OccupancyReport> {
    let name = CString::new(kernel).map_err(|_| {
        rocm_oxide::Error::InvalidLaunch(format!("kernel name `{kernel}` contains a NUL byte"))
    })?;
    let kernel = kernels
        .module()
        .kernel_with_metadata(name.as_c_str(), resources.launch_metadata())?;
    let active = kernel.occupancy_for_config(config)?;
    let potential = kernel.occupancy_max_potential_block_size(config.shared_mem_bytes, 0)?;
    let block_threads = config.block.x * config.block.y * config.block.z;
    let waves_per_block = resources
        .wavefront_size
        .filter(|wavefront| *wavefront > 0)
        .map(|wavefront| block_threads.div_ceil(wavefront));
    let waves_per_multiprocessor =
        waves_per_block.map(|waves| waves * active.blocks_per_multiprocessor);

    Ok(OccupancyReport {
        block_threads,
        dynamic_shared_mem_bytes: config.shared_mem_bytes,
        active_blocks_per_multiprocessor: active.blocks_per_multiprocessor,
        waves_per_block,
        waves_per_multiprocessor,
        suggested_min_grid_size: potential.min_grid_size,
        suggested_block_size: potential.block_size,
    })
}

fn detect_limiters(
    resources: Option<&rocm_oxide::KernelResource>,
    occupancy: Option<&OccupancyReport>,
) -> Vec<String> {
    let mut limiters = Vec::new();

    if let Some(resources) = resources {
        if resources.sgpr_spill_count.unwrap_or(0) > 0
            || resources.vgpr_spill_count.unwrap_or(0) > 0
        {
            limiters.push(format!(
                "spills:{}/{}",
                resources.sgpr_spill_count.unwrap_or(0),
                resources.vgpr_spill_count.unwrap_or(0)
            ));
        }
        if let Some(bytes) = resources
            .private_segment_fixed_size
            .filter(|bytes| *bytes > 0)
        {
            limiters.push(format!("private:{bytes}B"));
        }
        if let Some(bytes) = resources
            .group_segment_fixed_size
            .filter(|bytes| *bytes > 0)
        {
            limiters.push(format!("static-lds:{bytes}B"));
        }
        if resources.uses_dynamic_shared_mem {
            limiters.push("dynamic-lds".to_string());
        }
        if let Some(vgpr) = resources
            .vgpr_count
            .filter(|vgpr| *vgpr >= HIGH_VGPR_PRESSURE)
        {
            limiters.push(format!("vgpr:{vgpr}"));
        }
        if let Some(sgpr) = resources
            .sgpr_count
            .filter(|sgpr| *sgpr >= HIGH_SGPR_PRESSURE)
        {
            limiters.push(format!("sgpr:{sgpr}"));
        }
    }

    if let Some(occupancy) = occupancy {
        if occupancy.active_blocks_per_multiprocessor <= 1 {
            limiters.push(format!(
                "active-blocks:{}",
                occupancy.active_blocks_per_multiprocessor
            ));
        }
        if let Some(waves) = occupancy
            .waves_per_multiprocessor
            .filter(|waves| *waves < LOW_WAVES_PER_MULTIPROCESSOR)
        {
            limiters.push(format!("waves-per-mp:{waves}"));
        }
    }

    limiters
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
                    "Usage: cargo run --features device-spike --example performance_probe -- [--json target/perf.json]"
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
    occupancy: Option<OccupancyReport>,
    limiters: Vec<String>,
    resources: Option<rocm_oxide::KernelResource>,
}

#[derive(Debug, Clone)]
struct OccupancyReport {
    block_threads: u32,
    dynamic_shared_mem_bytes: u32,
    active_blocks_per_multiprocessor: u32,
    waves_per_block: Option<u32>,
    waves_per_multiprocessor: Option<u32>,
    suggested_min_grid_size: u32,
    suggested_block_size: u32,
}

impl OccupancyReport {
    fn summary(&self) -> String {
        match self.waves_per_multiprocessor {
            Some(waves) => format!("{}blk/{}wv", self.active_blocks_per_multiprocessor, waves),
            None => format!("{}blk", self.active_blocks_per_multiprocessor),
        }
    }
}

#[derive(Debug, Default)]
struct KernelResources {
    by_kernel: BTreeMap<&'static str, rocm_oxide::KernelResource>,
}

impl KernelResources {
    fn load_embedded() -> Self {
        Self {
            by_kernel: generated::DEVICE_KERNEL_RESOURCES
                .iter()
                .map(|resource| (resource.name, *resource))
                .collect(),
        }
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
    out.push_str("  \"format\": \"rocm-oxide-performance-probe-v2\",\n");
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
    out.push_str("      \"occupancy\": ");
    if let Some(occupancy) = &sample.occupancy {
        write_occupancy_json(out, occupancy);
        out.push_str(",\n");
    } else {
        out.push_str("null,\n");
    }
    out.push_str("      \"limiters\": ");
    write_json_string_array(out, &sample.limiters);
    out.push_str(",\n");
    out.push_str("      \"resources\": ");
    if let Some(resources) = &sample.resources {
        write_resource_json(out, resources);
        out.push('\n');
    } else {
        out.push_str("null\n");
    }
    out.push_str("    }");
}

fn write_occupancy_json(out: &mut String, occupancy: &OccupancyReport) {
    out.push_str("{\n");
    out.push_str(&format!(
        "        \"block_threads\": {}",
        occupancy.block_threads
    ));
    write_json_u32(
        out,
        "dynamic_shared_mem_bytes",
        Some(occupancy.dynamic_shared_mem_bytes),
        false,
    );
    write_json_u32(
        out,
        "active_blocks_per_multiprocessor",
        Some(occupancy.active_blocks_per_multiprocessor),
        false,
    );
    write_json_u32(out, "waves_per_block", occupancy.waves_per_block, false);
    write_json_u32(
        out,
        "waves_per_multiprocessor",
        occupancy.waves_per_multiprocessor,
        false,
    );
    write_json_u32(
        out,
        "suggested_min_grid_size",
        Some(occupancy.suggested_min_grid_size),
        false,
    );
    write_json_u32(
        out,
        "suggested_block_size",
        Some(occupancy.suggested_block_size),
        false,
    );
    out.push_str("\n      }");
}

fn write_resource_json(out: &mut String, resources: &rocm_oxide::KernelResource) {
    out.push_str("{\n");
    out.push_str(&format!(
        "        \"name\": \"{}\"",
        json_escape(resources.name)
    ));
    write_json_u32(
        out,
        "kernarg_segment_size",
        resources.kernarg_segment_size,
        false,
    );
    write_json_u32(
        out,
        "kernarg_segment_align",
        resources.kernarg_segment_align,
        false,
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
    out.push_str(&format!(
        ",\n        \"uses_dynamic_shared_mem\": {}",
        resources.uses_dynamic_shared_mem
    ));
    if let Some(value) = resources.uses_dynamic_stack {
        out.push_str(&format!(",\n        \"uses_dynamic_stack\": {value}"));
    }
    out.push_str("\n      }");
}

fn write_json_string_array(out: &mut String, values: &[String]) {
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("\"{}\"", json_escape(value)));
    }
    out.push(']');
}

fn write_json_u32(out: &mut String, key: &str, value: Option<u32>, first: bool) {
    if let Some(value) = value {
        if !first {
            out.push_str(",\n");
        }
        out.push_str(&format!("        \"{key}\": {value}"));
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
