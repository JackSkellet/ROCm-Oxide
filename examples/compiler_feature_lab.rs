use font8x8::{BASIC_FONTS, UnicodeFonts};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rocm_oxide::{
    Device, DeviceBuffer, DeviceOperation, Dim3, HostReferenceCaptureVisibility, LaunchConfig,
    ManagedBuffer, ManagedMemoryKind, MatrixIntegrationReport, PinnedHostBuffer, RocmLibraryReport,
    StreamPool,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1600;
const HEIGHT: usize = 900;
const SIDEBAR_W: usize = 430;
const PANEL_X: usize = SIDEBAR_W + 28;
const PANEL_W: usize = WIDTH - PANEL_X - 28;
const BG: u32 = 0x111111;
const PANEL: u32 = 0x1a1d21;
const PANEL_2: u32 = 0x242424;
const TEXT: u32 = 0xf3f1e8;
const MUTED: u32 = 0x9da7a6;
const CYAN: u32 = 0x42d9ff;
const GREEN: u32 = 0x80d25b;
const AMBER: u32 = 0xf1b447;
const RED: u32 = 0xe46a5e;
const MAGENTA: u32 = 0xd57cff;

#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Rect {
    const fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self { x, y, w, h }
    }

    fn contains(self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

#[derive(Clone, Copy)]
enum FeatureKind {
    Overview,
    RuntimeBasics,
    LaunchContracts,
    VisualKernels,
    LayoutClosure,
    MathIntrinsics,
    ReturnValue,
    CastMatrix,
    HostReference,
    SyncScope,
    LdsCollectives,
    DeviceApi,
    GraphOperations,
    RocmLibraries,
    DebugInfo,
}

struct FeatureProbe {
    kind: FeatureKind,
    title: String,
    status: String,
    detail: String,
    ok: bool,
    values: Vec<u64>,
}

struct AppState {
    probes: Vec<FeatureProbe>,
    selected: usize,
    signal_scale: f32,
    paused: bool,
    last_probe: Instant,
    message: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_frames = parse_max_frames();
    let mut state = AppState {
        probes: run_probes().unwrap_or_else(|err| {
            vec![FeatureProbe {
                kind: FeatureKind::DebugInfo,
                title: "Probe setup".to_string(),
                status: "GPU probes did not complete".to_string(),
                detail: err.to_string(),
                ok: false,
                values: vec![0, 1, 0, 1],
            }]
        }),
        selected: 0,
        signal_scale: 1.0,
        paused: false,
        last_probe: Instant::now(),
        message: "R reruns probes, Space pauses animation, number keys select panels".to_string(),
    };

    let mut window = Window::new(
        "ROCm-Oxide Compiler Feature Lab",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(60);

    let mut frame = vec![0u32; WIDTH * HEIGHT];
    let mut mouse_was_down = false;
    let mut rendered = 0u32;
    let start = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mouse = framebuffer_mouse_pos(&window);
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if mouse_down && let Some((mx, my)) = mouse {
            handle_mouse(&mut state, mx, my, !mouse_was_down);
        }
        mouse_was_down = mouse_down;

        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Key1 => state.selected = 0,
                Key::Key2 => state.selected = 1.min(state.probes.len().saturating_sub(1)),
                Key::Key3 => state.selected = 2.min(state.probes.len().saturating_sub(1)),
                Key::Key4 => state.selected = 3.min(state.probes.len().saturating_sub(1)),
                Key::Key5 => state.selected = 4.min(state.probes.len().saturating_sub(1)),
                Key::Key6 => state.selected = 5.min(state.probes.len().saturating_sub(1)),
                Key::Key7 => state.selected = 6.min(state.probes.len().saturating_sub(1)),
                Key::Key8 => state.selected = 7.min(state.probes.len().saturating_sub(1)),
                Key::Key9 => state.selected = 8.min(state.probes.len().saturating_sub(1)),
                Key::Key0 => state.selected = 9.min(state.probes.len().saturating_sub(1)),
                Key::Left => {
                    state.selected = state
                        .selected
                        .checked_sub(1)
                        .unwrap_or_else(|| state.probes.len().saturating_sub(1));
                }
                Key::Right => {
                    if !state.probes.is_empty() {
                        state.selected = (state.selected + 1) % state.probes.len();
                    }
                }
                Key::Up => state.signal_scale = (state.signal_scale + 0.1).min(2.5),
                Key::Down => state.signal_scale = (state.signal_scale - 0.1).max(0.4),
                Key::Space => state.paused = !state.paused,
                Key::R => rerun_probes(&mut state),
                _ => {}
            }
        }

        let elapsed = if state.paused {
            Duration::from_millis(0)
        } else {
            start.elapsed()
        };
        draw(&mut frame, &state, mouse, elapsed);
        window.update_with_buffer(&frame, WIDTH, HEIGHT)?;

        rendered += 1;
        if max_frames.is_some_and(|limit| rendered >= limit) {
            break;
        }
    }

    Ok(())
}

fn parse_max_frames() -> Option<u32> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--frames" {
            return args.next().and_then(|value| value.parse::<u32>().ok());
        }
    }
    std::env::var("ROCM_OXIDE_FEATURE_LAB_FRAMES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
}

fn rerun_probes(state: &mut AppState) {
    match run_probes() {
        Ok(probes) => {
            state.probes = probes;
            state.selected = state.selected.min(state.probes.len().saturating_sub(1));
            state.last_probe = Instant::now();
            state.message = "Probe refresh completed".to_string();
        }
        Err(err) => {
            state.message = format!("Probe refresh failed: {err}");
        }
    }
}

fn run_probes() -> Result<Vec<FeatureProbe>, Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let mut probes = vec![
        probe_or_error(
            FeatureKind::RuntimeBasics,
            "Runtime basics",
            run_runtime_basics_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::LaunchContracts,
            "Launch contracts",
            run_launch_contract_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::VisualKernels,
            "Visual kernels",
            run_visual_kernel_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::LayoutClosure,
            "Layout and closures",
            run_layout_closure_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::MathIntrinsics,
            "Math intrinsics",
            run_math_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::ReturnValue,
            "Return-by-value",
            run_return_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::CastMatrix,
            "Conversion matrix",
            run_cast_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::HostReference,
            "Host reference capture",
            run_host_reference_probe(&device, &kernels),
        ),
        probe_or_error(
            FeatureKind::SyncScope,
            "Atomic syncscope",
            run_syncscope_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::LdsCollectives,
            "LDS and collectives",
            run_lds_collectives_probe(&kernels),
        ),
        probe_or_error(
            FeatureKind::DeviceApi,
            "Wave and device API",
            run_device_api_probe(&device, &kernels),
        ),
        probe_or_error(
            FeatureKind::GraphOperations,
            "Graphs and operations",
            run_graph_operation_probe(&device, &kernels),
        ),
        probe_or_error(
            FeatureKind::RocmLibraries,
            "ROCm libraries",
            Ok(run_library_probe()),
        ),
        debug_info_probe(&device),
    ];
    let overview = overview_probe(&device, &probes);
    probes.insert(0, overview);
    Ok(probes)
}

fn probe_or_error(
    kind: FeatureKind,
    title: &str,
    result: Result<FeatureProbe, Box<dyn std::error::Error>>,
) -> FeatureProbe {
    match result {
        Ok(probe) => probe,
        Err(err) => FeatureProbe {
            kind,
            title: title.to_string(),
            status: "probe failed".to_string(),
            detail: err.to_string(),
            ok: false,
            values: vec![0, 1, 0, 1],
        },
    }
}

fn overview_probe(device: &Device, probes: &[FeatureProbe]) -> FeatureProbe {
    let ok = probes.iter().filter(|probe| probe.ok).count();
    let mut values = vec![
        probes.len() as u64,
        ok as u64,
        device.limits().max_threads_per_block as u64,
        device.limits().max_shared_mem_per_block as u64,
        device.limits().max_shared_mem_per_multiprocessor as u64,
    ];
    values.extend(
        probes
            .iter()
            .map(|probe| probe.values.iter().fold(0u64, |acc, value| acc ^ *value)),
    );
    FeatureProbe {
        kind: FeatureKind::Overview,
        title: "Whole stack overview".to_string(),
        status: format!("{ok}/{} live probes passed on {}", probes.len(), device.arch()),
        detail: "generated kernels, memory policy, atomics, LDS, wave APIs, graphs, libraries, debug artifacts".to_string(),
        ok: ok == probes.len(),
        values,
    }
}

fn run_runtime_basics_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let delta = kernels.global_add_one_delta()?;
    delta.set(2.0)?;
    let input = DeviceBuffer::from_slice(&[1.0f32, 5.5, -3.0, 0.25])?;
    let out = DeviceBuffer::<f32>::new(input.len())?;
    unsafe {
        kernels.add_one(
            LaunchConfig::for_num_elems_with_block_size(input.len(), 32),
            &out,
            &input,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let add = out.copy_to_vec()?;

    let a = DeviceBuffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0])?;
    let b = DeviceBuffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0])?;
    let sum = DeviceBuffer::<f32>::new(4)?;
    unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(4, 32),
            &sum,
            &a,
            &b,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let vector = sum.copy_to_vec()?;

    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 2.0,
        bias: 3.0,
    }])?;
    unsafe {
        kernels.affine_transform(
            LaunchConfig::for_num_elems_with_block_size(4, 32),
            &sum,
            &a,
            &params,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let affine = sum.copy_to_vec()?;
    let ok = add == vec![3.0, 7.5, -1.0, 2.25]
        && vector == vec![11.0, 22.0, 33.0, 44.0]
        && affine == vec![5.0, 7.0, 9.0, 11.0];
    let mut values = add
        .iter()
        .map(|value| value.to_bits() as u64)
        .collect::<Vec<_>>();
    values.extend(vector.iter().map(|value| value.to_bits() as u64));
    values.extend(affine.iter().map(|value| value.to_bits() as u64));
    Ok(FeatureProbe {
        kind: FeatureKind::RuntimeBasics,
        title: "Runtime basics".to_string(),
        status: if ok {
            "device globals, vector add, and affine kernels match".to_string()
        } else {
            "basic runtime output mismatch".to_string()
        },
        detail: "HIP module load, typed generated calls, device global access, and scalar buffers"
            .to_string(),
        ok,
        values,
    })
}

fn run_launch_contract_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let input = DeviceBuffer::from_slice(&[0u32, 1, 2, 3])?;
    let short_scores = DeviceBuffer::<u64>::new(2)?;
    let packets = DeviceBuffer::<generated::ReturnPacket>::new(4)?;
    let params = generated::ControlParams { seed: 1, scale: 2 };
    let rejected = unsafe {
        kernels.compiler_return_value_probe(
            LaunchConfig::for_num_elems_with_block_size(4, 32),
            &short_scores,
            &packets,
            &input,
            params,
            4,
        )
    };
    let ok = matches!(rejected, Err(rocm_oxide::Error::InvalidLaunch(_)));
    let detail = match rejected {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => message,
        Err(err) => format!("unexpected validation error: {err}"),
        Ok(()) => "short output buffer was not rejected".to_string(),
    };
    Ok(FeatureProbe {
        kind: FeatureKind::LaunchContracts,
        title: "Launch contracts".to_string(),
        status: if ok {
            "generated binding rejected an invalid launch before HIP".to_string()
        } else {
            "launch contract check did not reject as expected".to_string()
        },
        detail,
        ok,
        values: vec![4, 2, packets.len() as u64, ok as u64],
    })
}

fn run_visual_kernel_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let width = 1024usize;
    let height = 256usize;
    let pixel_count = width * height;
    let block_x = 256u32;
    let frame = DeviceBuffer::<u32>::new(pixel_count)?;
    unsafe {
        kernels.rainbow_geometry(
            LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
            &frame,
            width as u32,
            height as u32,
            17,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let rainbow = frame.copy_to_vec()?;

    unsafe {
        kernels.stress_pattern(
            LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
            &frame,
            17,
            5,
            32,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let stress = frame.copy_to_vec()?;

    let camera = DeviceBuffer::from_slice(&[
        0.0f32, 0.28, -1.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0,
    ])?;
    unsafe {
        kernels.raytrace_world(
            LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
            &frame,
            &camera,
            17,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let raytrace = frame.copy_to_vec()?;
    let samples = [
        pixel_count / 7,
        pixel_count / 3,
        pixel_count / 2,
        pixel_count - pixel_count / 5,
    ];
    let ok = samples.iter().any(|index| rainbow[*index] != 0)
        && samples
            .iter()
            .any(|index| stress[*index] != rainbow[*index])
        && samples
            .iter()
            .any(|index| raytrace[*index] != stress[*index]);
    let mut values = Vec::new();
    for index in samples {
        values.push(rainbow[index] as u64);
        values.push(stress[index] as u64);
        values.push(raytrace[index] as u64);
    }
    Ok(FeatureProbe {
        kind: FeatureKind::VisualKernels,
        title: "Visual kernels".to_string(),
        status: if ok {
            "rainbow, stress, and raytrace kernels produced distinct samples".to_string()
        } else {
            "visual kernel samples were not distinct".to_string()
        },
        detail: "samples the render kernels without embedding the older demo GUIs".to_string(),
        ok,
        values,
    })
}

fn run_layout_closure_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let layout_input = vec![2u32, 3, 5, 8, 13, 21, 34, 55];
    let layout_values = DeviceBuffer::from_slice(&layout_input)?;
    let layout_out = DeviceBuffer::<u32>::new(layout_input.len())?;
    let layout_params = generated::RustLayoutParams { base: 7, stride: 4 };
    unsafe {
        kernels.compiler_layout_probe(
            LaunchConfig::for_num_elems_with_block_size(layout_input.len(), 32),
            &layout_out,
            &layout_values,
            layout_params,
            layout_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let layout = layout_out.copy_to_vec()?;

    let closure_out = DeviceBuffer::<u32>::new(layout_input.len())?;
    unsafe {
        kernels.compiler_move_closure_probe_rust_layout_params(
            LaunchConfig::for_num_elems_with_block_size(layout_input.len(), 32),
            &closure_out,
            &layout_values,
            layout_params,
            layout_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let moved = closure_out.copy_to_vec()?;

    let host_out = DeviceBuffer::<u32>::new(layout_input.len())?;
    let host_closure = generated::HostAffineClosure {
        base: 19,
        stride: 3,
        xor_mask: 0x55aa,
    };
    let host_closure_arg = DeviceBuffer::from_slice(&[host_closure])?;
    unsafe {
        kernels.compiler_host_closure_arg_probe_host_affine_closure(
            LaunchConfig::for_num_elems_with_block_size(layout_input.len(), 32),
            &host_out,
            &layout_values,
            &host_closure_arg,
            layout_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let host = host_out.copy_to_vec()?;

    let layout_ok = layout_input.iter().enumerate().all(|(index, value)| {
        layout[index]
            == value
                .wrapping_mul(layout_params.stride)
                .wrapping_add(layout_params.base)
            && moved[index]
                == value
                    .wrapping_mul(layout_params.stride)
                    .wrapping_add(layout_params.base)
                    .wrapping_add((index as u32) & 1)
            && host[index]
                == ((value.wrapping_add((index as u32) & 3))
                    .wrapping_mul(host_closure.stride)
                    .wrapping_add(host_closure.base)
                    ^ host_closure.xor_mask)
    });
    let mut values = layout.iter().map(|value| *value as u64).collect::<Vec<_>>();
    values.extend(moved.iter().map(|value| *value as u64));
    values.extend(host.iter().map(|value| *value as u64));
    Ok(FeatureProbe {
        kind: FeatureKind::LayoutClosure,
        title: "Layout and closures".to_string(),
        status: if layout_ok {
            "repr(Rust) layout, moved closures, and host closure ABI match".to_string()
        } else {
            "layout or closure output mismatch".to_string()
        },
        detail: "rustc layout facts drive generated bindings for struct and closure payloads"
            .to_string(),
        ok: layout_ok,
        values,
    })
}

fn run_math_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let math_input = DeviceBuffer::from_slice(&[4.0f32, 0.0, 1.0, -1.0])?;
    let math_out = DeviceBuffer::<f32>::new(16)?;
    unsafe {
        kernels.math_intrinsics(LaunchConfig::for_num_elems(1), &math_out, &math_input)?;
    }
    rocm_oxide::hip::synchronize()?;
    let math = math_out.copy_to_vec()?;
    let ok = close_to(math[0], 2.0, 0.0001)
        && close_to(math[1], 0.5, 0.0001)
        && close_to(math[2], 0.0, 0.0001)
        && close_to(math[3], 1.0, 0.0001)
        && close_to(math[4], std::f32::consts::FRAC_PI_4, 0.002)
        && close_to(math[7], 2.0, 0.0001)
        && close_to(math[8], 0.5, 0.0001)
        && math[12] == 1.0
        && math[13] == 1.0
        && math[14] == 1.0;
    Ok(FeatureProbe {
        kind: FeatureKind::MathIntrinsics,
        title: "Math intrinsics".to_string(),
        status: if ok {
            "f32/f64 math helpers and NaN paths match expected values".to_string()
        } else {
            "math intrinsic output mismatch".to_string()
        },
        detail: "sqrt, rsqrt, sin, cos, atan, min/max, f64 mirrors, and NaN sentinels".to_string(),
        ok,
        values: math.iter().map(|value| value.to_bits() as u64).collect(),
    })
}

fn run_return_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let input = vec![0u32, 1, 2, 8, 17, 31, 64, 255];
    let params = generated::ControlParams {
        seed: 23,
        scale: -9,
    };
    let values = DeviceBuffer::from_slice(&input)?;
    let scores = DeviceBuffer::<u64>::new(input.len())?;
    let packets = DeviceBuffer::<generated::ReturnPacket>::new(input.len())?;
    unsafe {
        kernels.compiler_return_value_probe(
            LaunchConfig::for_num_elems_with_block_size(input.len(), 32),
            &scores,
            &packets,
            &values,
            params,
            input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let scores_host = scores.copy_to_vec()?;
    let packets_host = packets.copy_to_vec()?;
    let ok = input.iter().copied().enumerate().all(|(index, value)| {
        let expected = return_packet_host(value, params);
        packets_host[index].sum == expected.sum
            && packets_host[index].folded == expected.folded
            && packets_host[index].tag == expected.tag
            && scores_host[index] == return_packet_score_host(expected)
    });
    let mut visual = scores_host;
    visual.extend(packets_host.iter().map(|packet| packet.sum));
    Ok(FeatureProbe {
        kind: FeatureKind::ReturnValue,
        title: "Return-by-value".to_string(),
        status: if ok {
            "device struct returns match host mirror".to_string()
        } else {
            "mismatch in returned packet".to_string()
        },
        detail: format!(
            "{} lanes, internal Rust pair returned into repr(C) packet",
            input.len()
        ),
        ok,
        values: visual,
    })
}

fn run_cast_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let input = vec![0u32, 1, 3, 7, 19, 127, 1024, 65_535];
    let values = DeviceBuffer::from_slice(&input)?;
    let scores = DeviceBuffer::<u64>::new(input.len())?;
    let packets = DeviceBuffer::<generated::CastPacket>::new(input.len())?;
    unsafe {
        kernels.compiler_arithmetic_cast_probe(
            LaunchConfig::for_num_elems_with_block_size(input.len(), 32),
            &scores,
            &packets,
            &values,
            input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let scores_host = scores.copy_to_vec()?;
    let packets_host = packets.copy_to_vec()?;
    let ok = input.iter().copied().enumerate().all(|(index, value)| {
        let expected = cast_packet_host(value, index);
        packets_host[index].wide == expected.wide
            && packets_host[index].signed_bits == expected.signed_bits
            && packets_host[index].float_bits == expected.float_bits
            && packets_host[index].narrow == expected.narrow
            && scores_host[index] == cast_packet_score_host(expected)
    });
    let mut visual = scores_host;
    visual.extend(packets_host.iter().map(|packet| packet.wide));
    visual.extend(packets_host.iter().map(|packet| packet.signed_bits));
    Ok(FeatureProbe {
        kind: FeatureKind::CastMatrix,
        title: "Conversion matrix".to_string(),
        status: if ok {
            "64-bit arithmetic and bitcasts match".to_string()
        } else {
            "conversion matrix mismatch".to_string()
        },
        detail: "u32/i64/u64/f32/f64 bit patterns are checked per lane".to_string(),
        ok,
        values: visual,
    })
}

fn run_host_reference_probe(
    device: &Device,
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let input = vec![3u32, 5, 8, 13, 21, 34, 55, 89];
    let values = DeviceBuffer::from_slice(&input)?;
    let out = DeviceBuffer::<u32>::new(input.len())?;
    let properties = device.properties()?;
    let bias_value = 41u32;
    let scale = 7u32;

    let mut visibility = HostReferenceCaptureVisibility::DeviceOnly;
    let mut ran = false;
    if let Some(kind) = properties.mapped_host_reference_capture_kind() {
        visibility = kind.host_reference_capture_visibility();
        let mut bias = PinnedHostBuffer::<u32>::new_zeroed_mapped_coherent(1)?;
        bias.as_mut_slice()[0] = bias_value;
        let closure = generated::HostReferenceClosure {
            bias: bias.device_ptr()? as *const u32,
            scale,
        };
        unsafe {
            kernels.compiler_host_reference_closure_probe_host_reference_closure(
                LaunchConfig::for_num_elems_with_block_size(input.len(), 32),
                &out,
                &values,
                closure,
                input.len(),
            )?;
        }
        ran = true;
    } else if let Some(kind) =
        properties.managed_host_reference_capture_kind(ManagedMemoryKind::FineGrain)
    {
        visibility = kind.host_reference_capture_visibility();
        let bias = ManagedBuffer::from_slice(&[bias_value])?;
        let closure = generated::HostReferenceClosure {
            bias: bias.as_ptr(),
            scale,
        };
        unsafe {
            kernels.compiler_host_reference_closure_probe_host_reference_closure(
                LaunchConfig::for_num_elems_with_block_size(input.len(), 32),
                &out,
                &values,
                closure,
                input.len(),
            )?;
        }
        ran = true;
    }
    if ran {
        rocm_oxide::hip::synchronize()?;
    }
    let got = if ran { out.copy_to_vec()? } else { Vec::new() };
    let expected = expected_reference_closure(&input, bias_value, scale);
    let ok = ran && got == expected;
    let values = if ran {
        got.iter().map(|value| *value as u64).collect()
    } else {
        vec![
            0,
            properties.managed_memory as u64,
            properties.can_map_host_memory as u64,
        ]
    };
    Ok(FeatureProbe {
        kind: FeatureKind::HostReference,
        title: "Host reference capture".to_string(),
        status: if ok {
            "captured pointer read through validated memory policy".to_string()
        } else if ran {
            "captured pointer result mismatch".to_string()
        } else {
            "no valid mapped or managed capture path reported".to_string()
        },
        detail: format!(
            "visibility={visibility:?}, managed={}, mapped={}, native_atomics={}",
            properties.managed_memory,
            properties.can_map_host_memory,
            properties.host_native_atomic_supported
        ),
        ok,
        values,
    })
}

fn run_syncscope_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let out = DeviceBuffer::<u32>::new(4)?;
    let counters = DeviceBuffer::from_slice(&[0u32; 3])?;
    unsafe {
        kernels.scoped_atomics(LaunchConfig::new(Dim3::x(1), Dim3::x(256)), &out, &counters)?;
    }
    rocm_oxide::hip::synchronize()?;
    let markers = out.copy_to_vec()?;
    let counter_values = counters.copy_to_vec()?;
    let ok = markers == vec![0, 1, 2, 0] && counter_values == vec![256, 256, 256];
    let mut values = markers
        .iter()
        .map(|value| *value as u64)
        .collect::<Vec<_>>();
    values.extend(counter_values.iter().map(|value| *value as u64));
    Ok(FeatureProbe {
        kind: FeatureKind::SyncScope,
        title: "Atomic syncscope".to_string(),
        status: if ok {
            "workgroup/device/system markers survived lowering".to_string()
        } else {
            "scoped atomic runtime result mismatch".to_string()
        },
        detail: "IR rewrite covers atomicrmw, cmpxchg, load, and store markers".to_string(),
        ok,
        values,
    })
}

fn run_lds_collectives_probe(
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let reduce_n = 256usize;
    let block_x = 64u32;
    let partial_count = reduce_n.div_ceil(block_x as usize);
    let input = (0..reduce_n).map(|i| (i % 11) as f32).collect::<Vec<_>>();
    let expected = input
        .chunks(block_x as usize)
        .map(|chunk| chunk.iter().sum::<f32>())
        .collect::<Vec<_>>();
    let d_input = DeviceBuffer::from_slice(&input)?;
    let partials = DeviceBuffer::<f32>::new(partial_count)?;
    unsafe {
        kernels.lds_block_sum(
            LaunchConfig::for_num_elems_with_block_size(reduce_n, block_x)
                .try_with_dynamic_shared_mem::<f32>(block_x as usize)?,
            &partials,
            &d_input,
            reduce_n,
            partial_count,
            block_x,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let got_partials = partials.copy_to_vec()?;
    let lds_ok = got_partials
        .iter()
        .zip(expected.iter())
        .all(|(got, expected)| close_to(*got, *expected, 0.0001));

    let collective_block_x = 32u32;
    let collective_out = DeviceBuffer::<u32>::new(18)?;
    let scan = DeviceBuffer::<u32>::new(collective_block_x as usize)?;
    unsafe {
        kernels.block_collectives_probe(
            LaunchConfig::for_num_elems_with_block_size(
                collective_block_x as usize,
                collective_block_x,
            )
            .with_shared_mem_bytes(collective_block_x * 12),
            &collective_out,
            &scan,
            collective_block_x as usize,
            collective_block_x,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let collective = collective_out.copy_to_vec()?;
    let scan = scan.copy_to_vec()?;
    let collective_ok = collective[0] == 528
        && collective[3] == collective_block_x
        && scan.last().copied() == Some(528);
    let mut values = got_partials
        .iter()
        .map(|value| value.to_bits() as u64)
        .collect::<Vec<_>>();
    values.extend(collective.iter().map(|value| *value as u64));
    values.extend(scan.iter().take(8).map(|value| *value as u64));
    Ok(FeatureProbe {
        kind: FeatureKind::LdsCollectives,
        title: "LDS and collectives".to_string(),
        status: if lds_ok && collective_ok {
            "dynamic LDS reduction and block collectives match".to_string()
        } else {
            "LDS or collective output mismatch".to_string()
        },
        detail: "dynamic shared memory, barriers, reductions, scans, and block cooperative helpers"
            .to_string(),
        ok: lds_ok && collective_ok,
        values,
    })
}

fn run_device_api_probe(
    device: &Device,
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let cooperative_out = DeviceBuffer::<u32>::new(12)?;
    let block_x = 256u32;
    unsafe {
        kernels.cooperative_groups_probe(
            LaunchConfig::new(Dim3::x(1), Dim3::x(block_x)),
            &cooperative_out,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let cooperative = cooperative_out.copy_to_vec()?;

    let api_out = DeviceBuffer::<u32>::new(24)?;
    let i32_counter = DeviceBuffer::from_slice(&[0i32])?;
    let u64_counter = DeviceBuffer::from_slice(&[0u64])?;
    let i64_counter = DeviceBuffer::from_slice(&[0i64])?;
    unsafe {
        kernels.device_api_breadth_probe(
            LaunchConfig::new(Dim3::x(1), Dim3::x(32)),
            &api_out,
            &i32_counter,
            &u64_counter,
            &i64_counter,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let api = api_out.copy_to_vec()?;
    let wavefront_size = device.properties()?.warp_size;
    let active_lanes = 32u32.min(wavefront_size);
    let expected_sum = active_lanes * (active_lanes + 1) / 2;
    let ok = cooperative[0] == block_x
        && cooperative[3] == wavefront_size
        && cooperative[4] == 32
        && api[0] == 6
        && api[4] == expected_sum
        && u64_counter.copy_to_vec()? == vec![active_lanes as u64];
    let mut values = cooperative
        .iter()
        .map(|value| *value as u64)
        .collect::<Vec<_>>();
    values.extend(api.iter().map(|value| *value as u64));
    values.extend(
        i32_counter
            .copy_to_vec()?
            .iter()
            .map(|value| *value as u32 as u64),
    );
    values.extend(i64_counter.copy_to_vec()?.iter().map(|value| *value as u64));
    Ok(FeatureProbe {
        kind: FeatureKind::DeviceApi,
        title: "Wave and device API".to_string(),
        status: if ok {
            "wavefront shuffles, ballots, reductions, and cooperative groups match".to_string()
        } else {
            "device API output mismatch".to_string()
        },
        detail: format!("wavefront_size={wavefront_size}, active_lanes={active_lanes}"),
        ok,
        values,
    })
}

fn run_graph_operation_probe(
    device: &Device,
    kernels: &generated::DeviceKernels,
) -> Result<FeatureProbe, Box<dyn std::error::Error>> {
    let n = 8192usize;
    let block_x = 256u32;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (n - i) as f32).collect::<Vec<_>>();
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let graph_out = DeviceBuffer::<f32>::new(n)?;
    let graph = rocm_oxide::hip::Graph::new()?;
    unsafe {
        kernels.vector_add_graph_node(
            &graph,
            &[],
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &graph_out,
            &d_a,
            &d_b,
        )?;
    }
    let graph_exec = graph.instantiate()?;
    let graph_stream = rocm_oxide::Stream::new()?;
    graph_exec.launch(&graph_stream)?;
    graph_stream.synchronize()?;
    let graph_result = graph_out.copy_to_vec()?;

    let pool = StreamPool::new(device, 2)?;
    let lazy_a = Arc::new(DeviceBuffer::from_slice(&a)?);
    let lazy_b = Arc::new(DeviceBuffer::from_slice(&b)?);
    let lazy_out = Arc::new(DeviceBuffer::<f32>::new(n)?);
    let completion = unsafe {
        kernels.vector_add_operation(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            Arc::clone(&lazy_out),
            Arc::clone(&lazy_a),
            Arc::clone(&lazy_b),
        )?
    }
    .async_in(&pool)
    .wait()?;
    let lazy = lazy_out.copy_to_vec()?;
    let ok = graph_result[4096] == a[4096] + b[4096]
        && lazy[4096] == a[4096] + b[4096]
        && completion.retained_count() == 5;
    Ok(FeatureProbe {
        kind: FeatureKind::GraphOperations,
        title: "Graphs and operations".to_string(),
        status: if ok {
            "generated graph node and lazy DeviceOperation completed".to_string()
        } else {
            "graph or lazy operation result mismatch".to_string()
        },
        detail: "explicit HIP graph launch plus generated async operation on a two-stream pool"
            .to_string(),
        ok,
        values: vec![
            graph_result[0].to_bits() as u64,
            graph_result[4096].to_bits() as u64,
            lazy[4096].to_bits() as u64,
            completion.retained_count() as u64,
            n as u64,
            block_x as u64,
        ],
    })
}

fn run_library_probe() -> FeatureProbe {
    let report = RocmLibraryReport::query();
    let matrix = MatrixIntegrationReport::query();
    let libs = [
        &report.rocblas,
        &report.rocfft,
        &report.hipblaslt,
        &report.comgr,
        &report.rocprim,
        &matrix.hipblaslt,
        &matrix.composable_kernel,
        &matrix.rocwmma,
    ];
    let available = libs.iter().filter(|lib| lib.available).count();
    let mut values = libs
        .iter()
        .enumerate()
        .map(|(index, lib)| {
            ((lib.available as u64) << 32) | (lib.detail.len() as u64) | index as u64
        })
        .collect::<Vec<_>>();
    values.push(available as u64);
    FeatureProbe {
        kind: FeatureKind::RocmLibraries,
        title: "ROCm libraries".to_string(),
        status: format!(
            "{available}/{} library and matrix backends reported available",
            libs.len()
        ),
        detail: format!(
            "rocBLAS={}, rocFFT={}, hipBLASLt={}, COMGR={}, rocPRIM={}",
            report.rocblas.available,
            report.rocfft.available,
            report.hipblaslt.available,
            report.comgr.available,
            report.rocprim.available
        ),
        ok: available >= 4,
        values,
    }
}

fn debug_info_probe(device: &Device) -> FeatureProbe {
    let debug_enabled = env_flag_enabled(std::env::var("ROCM_OXIDE_DEVICE_DEBUG").ok().as_deref());
    let metadata = option_env!("ROCM_OXIDE_DEVICE_METADATA").unwrap_or("metadata unavailable");
    FeatureProbe {
        kind: FeatureKind::DebugInfo,
        title: "Debug-info pipeline".to_string(),
        status: if debug_enabled {
            "ROCM_OXIDE_DEVICE_DEBUG is enabled for this run".to_string()
        } else {
            "debug path available; enable ROCM_OXIDE_DEVICE_DEBUG=1".to_string()
        },
        detail: format!(
            "arch={}, artifacts keep LLVM IR, object, HSACO, and metadata: {}",
            device.arch(),
            metadata
        ),
        ok: true,
        values: vec![
            debug_enabled as u64,
            device.limits().max_threads_per_block as u64,
            device.limits().max_shared_mem_per_block as u64,
            device.limits().max_shared_mem_per_multiprocessor as u64,
        ],
    }
}

fn env_flag_enabled(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    !matches!(
        value.to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn close_to(got: f32, expected: f32, tolerance: f32) -> bool {
    (got - expected).abs() <= tolerance
}

fn handle_mouse(state: &mut AppState, x: usize, y: usize, first_press: bool) {
    for (index, rect) in feature_button_rects(state.probes.len())
        .into_iter()
        .enumerate()
    {
        if first_press && rect.contains(x, y) {
            state.selected = index;
            return;
        }
    }
    if first_press && Rect::new(26, HEIGHT - 142, 128, 38).contains(x, y) {
        rerun_probes(state);
        return;
    }
    if first_press && Rect::new(170, HEIGHT - 142, 128, 38).contains(x, y) {
        state.paused = !state.paused;
        return;
    }
    let slider = Rect::new(30, HEIGHT - 72, SIDEBAR_W - 60, 22);
    if slider.contains(x, y) {
        let t = ((x - slider.x) as f32 / slider.w as f32).clamp(0.0, 1.0);
        state.signal_scale = 0.4 + t * 2.1;
    }
}

fn draw(frame: &mut [u32], state: &AppState, mouse: Option<(usize, usize)>, elapsed: Duration) {
    clear_background(frame, elapsed);
    fill_rect(frame, Rect::new(0, 0, SIDEBAR_W, HEIGHT), 0x151719);
    fill_rect(frame, Rect::new(SIDEBAR_W, 0, 2, HEIGHT), 0x3b3f44);

    draw_text_fit(
        frame,
        Rect::new(26, 26, SIDEBAR_W - 52, 34),
        "Compiler Feature Lab",
        TEXT,
        2,
    );
    draw_text_fit(
        frame,
        Rect::new(28, 66, SIDEBAR_W - 56, 18),
        "ROCm-Oxide AMDGPU parity probes",
        CYAN,
        1,
    );
    draw_wrapped_text(
        frame,
        Rect::new(28, 94, SIDEBAR_W - 56, 42),
        "clicks use scaled framebuffer coordinates",
        MUTED,
        1,
        2,
    );

    for (index, rect) in feature_button_rects(state.probes.len())
        .into_iter()
        .enumerate()
    {
        let probe = &state.probes[index];
        let hover = mouse.is_some_and(|(mx, my)| rect.contains(mx, my));
        let selected = state.selected == index;
        let color = if selected {
            0x2d6f83
        } else if hover {
            0x30343b
        } else {
            PANEL
        };
        fill_rect(frame, rect, color);
        stroke_rect(
            frame,
            rect,
            if selected {
                accent_for(probe.kind)
            } else {
                0x51565d
            },
        );
        let badge = if probe.ok { "OK" } else { "CHECK" };
        draw_text_fit(
            frame,
            Rect::new(rect.x + 14, rect.y + 9, rect.w - 92, 20),
            &probe.title,
            TEXT,
            1,
        );
        draw_text_fit(
            frame,
            Rect::new(rect.x + rect.w - 72, rect.y + 9, 56, 20),
            badge,
            if probe.ok { GREEN } else { RED },
            1,
        );
    }

    draw_button(
        frame,
        Rect::new(26, HEIGHT - 142, 128, 38),
        "Rerun",
        mouse,
        AMBER,
    );
    draw_button(
        frame,
        Rect::new(170, HEIGHT - 142, 128, 38),
        if state.paused { "Resume" } else { "Pause" },
        mouse,
        MAGENTA,
    );
    draw_text_fit(
        frame,
        Rect::new(30, HEIGHT - 92, SIDEBAR_W - 60, 18),
        "Signal scale",
        TEXT,
        1,
    );
    let slider = Rect::new(30, HEIGHT - 72, SIDEBAR_W - 60, 22);
    stroke_rect(frame, slider, 0x626970);
    let fill_w = ((state.signal_scale - 0.4) / 2.1 * slider.w as f32) as usize;
    fill_rect(
        frame,
        Rect::new(slider.x + 2, slider.y + 2, fill_w, slider.h - 4),
        CYAN,
    );
    draw_text_fit(
        frame,
        Rect::new(30, HEIGHT - 38, SIDEBAR_W - 60, 18),
        &format!("{:.2}x  {}", state.signal_scale, state.message),
        MUTED,
        1,
    );

    if state.probes.is_empty() {
        return;
    }
    let selected = state.selected.min(state.probes.len() - 1);
    let probe = &state.probes[selected];
    draw_probe_panel(frame, probe, state.signal_scale, mouse, elapsed);
}

fn draw_probe_panel(
    frame: &mut [u32],
    probe: &FeatureProbe,
    scale: f32,
    mouse: Option<(usize, usize)>,
    elapsed: Duration,
) {
    let accent = accent_for(probe.kind);
    let header = Rect::new(PANEL_X, 30, PANEL_W, 136);
    fill_rect(frame, header, PANEL);
    stroke_rect(frame, header, accent);
    draw_text_fit(
        frame,
        Rect::new(header.x + 26, header.y + 22, header.w - 52, 34),
        &probe.title,
        TEXT,
        2,
    );
    draw_wrapped_text(
        frame,
        Rect::new(header.x + 26, header.y + 66, header.w - 52, 32),
        &format!("status: {}", probe.status),
        if probe.ok { GREEN } else { RED },
        1,
        2,
    );
    draw_wrapped_text(
        frame,
        Rect::new(header.x + 26, header.y + 102, header.w - 52, 28),
        &probe.detail,
        MUTED,
        1,
        2,
    );

    let plot = Rect::new(PANEL_X, 196, PANEL_W, 430);
    fill_rect(frame, plot, 0x141414);
    stroke_rect(frame, plot, 0x424850);
    draw_grid(frame, plot, elapsed);
    match probe.kind {
        FeatureKind::Overview => draw_overview_view(frame, plot, probe, scale, elapsed),
        FeatureKind::RuntimeBasics => draw_runtime_view(frame, plot, probe, scale),
        FeatureKind::LaunchContracts => draw_contract_view(frame, plot, probe, scale),
        FeatureKind::VisualKernels => draw_visual_kernel_view(frame, plot, probe, scale),
        FeatureKind::LayoutClosure => draw_layout_view(frame, plot, probe, scale),
        FeatureKind::MathIntrinsics => draw_math_view(frame, plot, probe, scale, elapsed),
        FeatureKind::ReturnValue => draw_return_view(frame, plot, probe, scale, elapsed),
        FeatureKind::CastMatrix => draw_cast_view(frame, plot, probe, scale, elapsed),
        FeatureKind::HostReference => draw_host_ref_view(frame, plot, probe, scale),
        FeatureKind::SyncScope => draw_syncscope_view(frame, plot, probe, scale),
        FeatureKind::LdsCollectives => draw_lds_view(frame, plot, probe, scale),
        FeatureKind::DeviceApi => draw_device_api_view(frame, plot, probe, scale, elapsed),
        FeatureKind::GraphOperations => draw_graph_view(frame, plot, probe, scale),
        FeatureKind::RocmLibraries => draw_library_view(frame, plot, probe, scale),
        FeatureKind::DebugInfo => draw_debug_view(frame, plot, probe, scale),
    }

    let inspector = Rect::new(PANEL_X, 658, PANEL_W, 204);
    fill_rect(frame, inspector, PANEL_2);
    stroke_rect(frame, inspector, 0x555b61);
    draw_text_fit(
        frame,
        Rect::new(inspector.x + 22, inspector.y + 18, 300, 18),
        "Feature signals",
        TEXT,
        1,
    );
    draw_wrapped_text(
        frame,
        Rect::new(inspector.x + 22, inspector.y + 42, inspector.w - 44, 36),
        "Each band is drawn from the GPU probe payload or build artifact state.",
        MUTED,
        1,
        2,
    );
    for (row, chunk) in probe.values.chunks(10).take(5).enumerate() {
        let y = inspector.y + 84 + row * 20;
        let mut text = String::new();
        for value in chunk {
            text.push_str(&format!("{:08x} ", (value & 0xffff_ffff) as u32));
        }
        draw_text_clipped(
            frame,
            inspector.x + 22,
            y,
            text.trim_end(),
            accent,
            1,
            Rect::new(
                inspector.x + 18,
                inspector.y + 80,
                inspector.w - 36,
                inspector.h - 92,
            ),
        );
    }
    if let Some((mx, my)) = mouse {
        draw_text_fit(
            frame,
            Rect::new(inspector.x + inspector.w - 220, inspector.y + 18, 198, 18),
            &format!("mouse fb {},{}", mx, my),
            MUTED,
            1,
        );
    }
}

fn draw_overview_view(
    frame: &mut [u32],
    rect: Rect,
    probe: &FeatureProbe,
    scale: f32,
    elapsed: Duration,
) {
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, GREEN);
    let labels = [
        "runtime",
        "contracts",
        "visuals",
        "layout",
        "math",
        "returns",
        "casts",
        "host mem",
        "atomics",
        "LDS",
        "wave API",
        "graphs",
        "libraries",
        "debug",
    ];
    let cols = 5;
    let cell_w = (rect.w - 88) / cols;
    let cell_h = 54;
    for (index, label) in labels.iter().enumerate() {
        let col = index % cols;
        let row = index / cols;
        let x = rect.x + 44 + col * cell_w;
        let y = rect.y + 62 + row * (cell_h + 20);
        let value = probe.values.get(index + 5).copied().unwrap_or(index as u64);
        let color = blend(CYAN, AMBER, ((value & 0xff) as f32) / 255.0);
        fill_rect(frame, Rect::new(x, y, cell_w - 22, cell_h), 0x20252a);
        stroke_rect(frame, Rect::new(x, y, cell_w - 22, cell_h), color);
        draw_text_fit(
            frame,
            Rect::new(x + 14, y + 13, cell_w - 50, 18),
            label,
            TEXT,
            1,
        );
        draw_text_fit(
            frame,
            Rect::new(x + 14, y + 34, cell_w - 50, 16),
            "live probe",
            MUTED,
            1,
        );
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 24, rect.w - 56, 32),
        "This panel is the stack map: every tile below has its own selectable proof panel.",
        TEXT,
        1,
        2,
    );
    let pulse = ((elapsed.as_secs_f32() * 3.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
    fill_rect(
        frame,
        Rect::new(rect.x + rect.w - 110, rect.y + 24, 62, 24),
        blend(GREEN, CYAN, pulse),
    );
}

fn draw_runtime_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, GREEN);
    let lanes = ["global", "add_one", "vector_add", "affine", "readback"];
    for (index, label) in lanes.iter().enumerate() {
        let x = rect.x + 58 + index * ((rect.w - 116) / lanes.len());
        let y = rect.y + 118 + (index % 2) * 72;
        fill_rect(frame, Rect::new(x, y, 136, 52), 0x1f3033);
        stroke_rect(frame, Rect::new(x, y, 136, 52), CYAN);
        draw_text_fit(frame, Rect::new(x + 12, y + 18, 112, 16), label, TEXT, 1);
        if index > 0 {
            draw_line(frame, x - 34, y + 26, x, y + 26, GREEN);
        }
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 24, rect.w - 56, 34),
        "Basic HIP module loading, generated calls, device globals, and buffer round trips.",
        TEXT,
        1,
        2,
    );
}

fn draw_contract_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, AMBER, RED);
    let labels = [
        "typed args",
        "length",
        "alias",
        "block",
        "LDS",
        "HIP launch",
    ];
    for (index, label) in labels.iter().enumerate() {
        let x = rect.x + 52 + index * ((rect.w - 104) / labels.len());
        let y = rect.y + 110 + (index % 2) * 72;
        fill_rect(
            frame,
            Rect::new(x, y, 118, 46),
            if index < 5 { 0x3a3120 } else { 0x213624 },
        );
        stroke_rect(
            frame,
            Rect::new(x, y, 118, 46),
            if index < 5 { AMBER } else { GREEN },
        );
        draw_text_fit(frame, Rect::new(x + 10, y + 15, 98, 16), label, TEXT, 1);
        if index > 0 {
            draw_line(frame, x - 34, y + 22, x, y + 22, MUTED);
        }
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 26, rect.w - 56, 34),
        "Generated bindings reject invalid launches before HIP sees them.",
        TEXT,
        1,
        2,
    );
}

fn draw_visual_kernel_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, MAGENTA, AMBER);
    let labels = ["rainbow", "stress", "raytrace"];
    let swatch_w = (rect.w - 116) / 12;
    for group in 0..4 {
        for (kernel, label) in labels.iter().enumerate() {
            let value = probe.values.get(group * 3 + kernel).copied().unwrap_or(0) as u32;
            let x = rect.x + 58 + group * (swatch_w * 3 + 28) + kernel * swatch_w;
            let y = rect.y + 118;
            fill_rect(
                frame,
                Rect::new(x, y, swatch_w.saturating_sub(4), 92),
                value & 0x00ff_ffff,
            );
            stroke_rect(
                frame,
                Rect::new(x, y, swatch_w.saturating_sub(4), 92),
                if value == 0 { RED } else { TEXT },
            );
            if group == 0 {
                draw_text_fit(
                    frame,
                    Rect::new(x, y + 106, swatch_w.saturating_sub(4), 16),
                    label,
                    MUTED,
                    1,
                );
            }
        }
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 24, rect.w - 56, 34),
        "Samples distinct pixels from rainbow, compute-stress, and raytrace kernels.",
        TEXT,
        1,
        2,
    );
}

fn draw_layout_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, GREEN, MAGENTA);
    let nodes = [
        "rustc layout",
        "repr(Rust)",
        "move closure",
        "host env",
        "device fn",
    ];
    for (index, node) in nodes.iter().enumerate() {
        let x = rect.x + 64 + index * 178;
        let y = rect.y + 92 + (index % 2) * 86;
        fill_rect(frame, Rect::new(x, y, 136, 54), 0x222b28);
        stroke_rect(frame, Rect::new(x, y, 136, 54), GREEN);
        draw_text_fit(frame, Rect::new(x + 12, y + 19, 112, 16), node, TEXT, 1);
        if index > 0 {
            draw_line(frame, x - 42, y + 27, x, y + 27, CYAN);
        }
    }
}

fn draw_math_view(
    frame: &mut [u32],
    rect: Rect,
    probe: &FeatureProbe,
    scale: f32,
    elapsed: Duration,
) {
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, MAGENTA);
    let mid = rect.y + rect.h / 2;
    let phase = elapsed.as_secs_f32();
    let mut last = None;
    for step in 0..rect.w.saturating_sub(70) {
        let x = rect.x + 35 + step;
        let t = step as f32 / 42.0 + phase * 2.0;
        let y = (mid as f32 + t.sin() * 82.0 + (t * 0.37).cos() * 28.0) as usize;
        if let Some((lx, ly)) = last {
            draw_line(frame, lx, ly, x, y, CYAN);
        }
        last = Some((x, y));
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 24, rect.w - 56, 34),
        "The probe checks f32/f64 intrinsics plus NaN sentinel paths.",
        TEXT,
        1,
        2,
    );
}

fn draw_return_view(
    frame: &mut [u32],
    rect: Rect,
    probe: &FeatureProbe,
    scale: f32,
    elapsed: Duration,
) {
    let pulse = (elapsed.as_secs_f32() * 2.0).sin() * 0.5 + 0.5;
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, AMBER);
    for (index, value) in probe.values.iter().take(8).enumerate() {
        let x = rect.x + 46 + index * ((rect.w - 92) / 8);
        let y = rect.y + 54 + (((*value as usize) ^ (index * 31)) % 54);
        let color = blend(CYAN, GREEN, pulse);
        fill_rect(frame, Rect::new(x, y, 34, 34), color);
        stroke_rect(frame, Rect::new(x, y, 34, 34), TEXT);
        draw_text_fit(frame, Rect::new(x + 5, y + 12, 24, 12), "RET", 0x111111, 1);
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 26, rect.w - 56, 28),
        "function returns -> packet fields",
        TEXT,
        1,
        2,
    );
}

fn draw_cast_view(
    frame: &mut [u32],
    rect: Rect,
    probe: &FeatureProbe,
    scale: f32,
    elapsed: Duration,
) {
    draw_value_bars(frame, rect, &probe.values, scale, MAGENTA, GREEN);
    let cell_w = ((rect.w - 72) / 32).max(8);
    let cell_h = 10;
    let shift = (elapsed.as_millis() / 90) as usize;
    for row in 0..12 {
        let value = probe
            .values
            .get(row % probe.values.len())
            .copied()
            .unwrap_or(0);
        for bit in 0..32 {
            let on = ((value >> ((bit + shift) & 31)) & 1) != 0;
            let color = if on { CYAN } else { 0x272b2d };
            fill_rect(
                frame,
                Rect::new(
                    rect.x + 36 + bit * cell_w,
                    rect.y + 42 + row * (cell_h + 4),
                    cell_w - 2,
                    cell_h,
                ),
                color,
            );
        }
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 26, rect.w - 56, 28),
        "integer, float, pointer-cast, and bitcast lanes",
        TEXT,
        1,
        2,
    );
}

fn draw_host_ref_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, GREEN, AMBER);
    let labels = ["host ptr", "policy", "closure", "device read", "sync"];
    for (index, label) in labels.iter().enumerate() {
        let x = rect.x + 58 + index * 150;
        let y = rect.y + 74 + (index % 2) * 70;
        fill_rect(
            frame,
            Rect::new(x, y, 110, 48),
            if probe.ok { 0x25412b } else { 0x44251f },
        );
        stroke_rect(frame, Rect::new(x, y, 110, 48), accent_for(probe.kind));
        draw_text_fit(frame, Rect::new(x + 10, y + 16, 90, 16), label, TEXT, 1);
        if index > 0 {
            draw_line(frame, x - 40, y + 24, x, y + 24, CYAN);
        }
    }
}

fn draw_syncscope_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, MAGENTA);
    let scopes = [
        ("workgroup", 0x65d46e),
        ("device", 0x42d9ff),
        ("system", 0xf1b447),
    ];
    for (index, (label, color)) in scopes.iter().enumerate() {
        let x = rect.x + 92 + index * 220;
        let h = 58 + index * 36;
        fill_rect(frame, Rect::new(x, rect.y + 210 - h, 138, h), *color);
        stroke_rect(frame, Rect::new(x, rect.y + 210 - h, 138, h), TEXT);
        draw_text_fit(
            frame,
            Rect::new(x + 12, rect.y + 226, 114, 16),
            label,
            TEXT,
            1,
        );
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 26, rect.w - 56, 28),
        "source markers lower to AMDGPU syncscope choices",
        TEXT,
        1,
        2,
    );
}

fn draw_lds_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, GREEN, CYAN);
    let tile_cols = 8;
    let tile_w = (rect.w - 92) / tile_cols;
    for row in 0..4 {
        for col in 0..tile_cols {
            let index = row * tile_cols + col;
            let value = probe.values.get(index).copied().unwrap_or(index as u64);
            let x = rect.x + 46 + col * tile_w;
            let y = rect.y + 62 + row * 54;
            let color = blend(0x244230, CYAN, ((value >> 7) & 0xff) as f32 / 255.0);
            fill_rect(frame, Rect::new(x, y, tile_w - 8, 38), color);
            stroke_rect(frame, Rect::new(x, y, tile_w - 8, 38), GREEN);
        }
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 24, rect.w - 56, 30),
        "LDS tiles, barriers, reductions, scans, and block-level collective helpers.",
        TEXT,
        1,
        2,
    );
}

fn draw_device_api_view(
    frame: &mut [u32],
    rect: Rect,
    probe: &FeatureProbe,
    scale: f32,
    elapsed: Duration,
) {
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, AMBER);
    let lanes = 32usize;
    let lane_w = ((rect.w - 86) / lanes).max(10);
    let wave_y = rect.y + 104;
    let phase = (elapsed.as_millis() / 80) as usize;
    for lane in 0..lanes {
        let value = probe
            .values
            .get(lane % probe.values.len())
            .copied()
            .unwrap_or(0);
        let active = ((value as usize + lane + phase) & 3) != 0;
        let x = rect.x + 42 + lane * lane_w;
        let h = if active { 72 + (lane % 5) * 10 } else { 34 };
        fill_rect(
            frame,
            Rect::new(x, wave_y + 132 - h, lane_w.saturating_sub(3), h),
            if active { CYAN } else { 0x293033 },
        );
    }
    draw_wrapped_text(
        frame,
        Rect::new(rect.x + 28, rect.y + 24, rect.w - 56, 34),
        "Wavefront shuffles, ballots, reductions, lane masks, and cooperative group facts.",
        TEXT,
        1,
        2,
    );
}

fn draw_graph_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, MAGENTA, CYAN);
    let nodes = [
        "buffers",
        "kernel node",
        "instantiate",
        "launch",
        "stream pool",
        "future wait",
    ];
    for (index, node) in nodes.iter().enumerate() {
        let x = rect.x + 46 + index * ((rect.w - 92) / nodes.len());
        let y = rect.y + 108 + (index % 2) * 72;
        fill_rect(frame, Rect::new(x, y, 128, 48), 0x29233a);
        stroke_rect(frame, Rect::new(x, y, 128, 48), MAGENTA);
        draw_text_fit(frame, Rect::new(x + 10, y + 16, 108, 16), node, TEXT, 1);
        if index > 0 {
            draw_line(frame, x - 26, y + 24, x, y + 24, CYAN);
        }
    }
}

fn draw_library_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, AMBER, GREEN);
    let labels = [
        "rocBLAS",
        "rocFFT",
        "hipBLASLt",
        "COMGR",
        "rocPRIM",
        "matrix",
        "CK",
        "rocWMMA",
    ];
    for (index, label) in labels.iter().enumerate() {
        let value = probe.values.get(index).copied().unwrap_or(0);
        let available = (value >> 32) != 0;
        let x = rect.x + 48 + (index % 4) * ((rect.w - 96) / 4);
        let y = rect.y + 78 + (index / 4) * 92;
        fill_rect(
            frame,
            Rect::new(x, y, 190, 58),
            if available { 0x243723 } else { 0x3a2424 },
        );
        stroke_rect(
            frame,
            Rect::new(x, y, 190, 58),
            if available { GREEN } else { RED },
        );
        draw_text_fit(frame, Rect::new(x + 12, y + 14, 150, 18), label, TEXT, 1);
        draw_text_fit(
            frame,
            Rect::new(x + 12, y + 34, 150, 16),
            if available { "available" } else { "missing" },
            if available { GREEN } else { RED },
            1,
        );
    }
}

fn draw_debug_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, AMBER, CYAN);
    let nodes = ["rustc -g", "LLVM IR", "metadata fix", "clang -g", "HSACO"];
    for (index, node) in nodes.iter().enumerate() {
        let x = rect.x + 42 + index * 150;
        let y = rect.y + 108 + (index % 2) * 42;
        fill_rect(frame, Rect::new(x, y, 116, 46), 0x302a1d);
        stroke_rect(frame, Rect::new(x, y, 116, 46), AMBER);
        draw_text_fit(frame, Rect::new(x + 8, y + 16, 100, 16), node, TEXT, 1);
        if index > 0 {
            draw_line(frame, x - 34, y + 22, x, y + 22, CYAN);
        }
    }
}

fn draw_value_bars(frame: &mut [u32], rect: Rect, values: &[u64], scale: f32, a: u32, b: u32) {
    if values.is_empty() {
        return;
    }
    let bar_area = Rect::new(rect.x + 26, rect.y + 210, rect.w - 52, rect.h - 242);
    let bar_w = (bar_area.w / values.len().max(1)).max(4);
    for (index, value) in values.iter().enumerate() {
        let t = ((*value ^ (*value >> 23)) & 0xff) as f32 / 255.0;
        let h = (((18.0 + t * bar_area.h as f32) * scale).min(bar_area.h as f32)) as usize;
        let x = bar_area.x + index * bar_w;
        let y = bar_area.y + bar_area.h - h;
        fill_rect(
            frame,
            Rect::new(x, y, bar_w.saturating_sub(2), h),
            blend(a, b, t),
        );
    }
}

fn feature_button_rects(count: usize) -> Vec<Rect> {
    (0..count)
        .map(|index| Rect::new(26, 146 + index * 44, SIDEBAR_W - 52, 34))
        .collect()
}

fn framebuffer_mouse_pos(window: &Window) -> Option<(usize, usize)> {
    let (mx, my) = window.get_unscaled_mouse_pos(MouseMode::Discard)?;
    let (win_w, win_h) = window.get_size();
    if win_w == 0 || win_h == 0 {
        return None;
    }
    let x = ((mx.max(0.0) as f64) * WIDTH as f64 / win_w as f64)
        .floor()
        .clamp(0.0, (WIDTH - 1) as f64) as usize;
    let y = ((my.max(0.0) as f64) * HEIGHT as f64 / win_h as f64)
        .floor()
        .clamp(0.0, (HEIGHT - 1) as f64) as usize;
    Some((x, y))
}

fn clear_background(frame: &mut [u32], elapsed: Duration) {
    let phase = (elapsed.as_millis() / 24) as usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let stripe = ((x / 18 + y / 21 + phase) & 1) as u32;
            let shade = 0x0f + (((x ^ y ^ phase) & 7) as u32);
            frame[y * WIDTH + x] = if stripe == 0 {
                (shade << 16) | (shade << 8) | shade
            } else {
                BG
            };
        }
    }
}

fn draw_grid(frame: &mut [u32], rect: Rect, elapsed: Duration) {
    let offset = (elapsed.as_millis() / 36) as usize % 32;
    for x in (rect.x + offset..rect.x + rect.w).step_by(32) {
        draw_line(frame, x, rect.y, x, rect.y + rect.h - 1, 0x262a2e);
    }
    for y in (rect.y..rect.y + rect.h).step_by(28) {
        draw_line(frame, rect.x, y, rect.x + rect.w - 1, y, 0x262a2e);
    }
}

fn draw_button(
    frame: &mut [u32],
    rect: Rect,
    label: &str,
    mouse: Option<(usize, usize)>,
    accent: u32,
) {
    let hover = mouse.is_some_and(|(x, y)| rect.contains(x, y));
    fill_rect(frame, rect, if hover { 0x34383d } else { PANEL });
    stroke_rect(frame, rect, accent);
    draw_text_fit(
        frame,
        Rect::new(rect.x + 18, rect.y + 11, rect.w - 36, 16),
        label,
        TEXT,
        1,
    );
}

fn fill_rect(frame: &mut [u32], rect: Rect, color: u32) {
    let x1 = rect.x.min(WIDTH);
    let y1 = rect.y.min(HEIGHT);
    let x2 = (rect.x + rect.w).min(WIDTH);
    let y2 = (rect.y + rect.h).min(HEIGHT);
    for y in y1..y2 {
        for x in x1..x2 {
            frame[y * WIDTH + x] = color;
        }
    }
}

fn stroke_rect(frame: &mut [u32], rect: Rect, color: u32) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    draw_line(frame, rect.x, rect.y, rect.x + rect.w - 1, rect.y, color);
    draw_line(frame, rect.x, rect.y, rect.x, rect.y + rect.h - 1, color);
    draw_line(
        frame,
        rect.x + rect.w - 1,
        rect.y,
        rect.x + rect.w - 1,
        rect.y + rect.h - 1,
        color,
    );
    draw_line(
        frame,
        rect.x,
        rect.y + rect.h - 1,
        rect.x + rect.w - 1,
        rect.y + rect.h - 1,
        color,
    );
}

fn draw_line(frame: &mut [u32], mut x0: usize, mut y0: usize, x1: usize, y1: usize, color: u32) {
    let mut x0i = x0 as isize;
    let mut y0i = y0 as isize;
    let x1i = x1 as isize;
    let y1i = y1 as isize;
    let dx = (x1i - x0i).abs();
    let sx = if x0i < x1i { 1 } else { -1 };
    let dy = -(y1i - y0i).abs();
    let sy = if y0i < y1i { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0i >= 0 && y0i >= 0 {
            x0 = x0i as usize;
            y0 = y0i as usize;
            if x0 < WIDTH && y0 < HEIGHT {
                frame[y0 * WIDTH + x0] = color;
            }
        }
        if x0i == x1i && y0i == y1i {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0i += sx;
        }
        if e2 <= dx {
            err += dx;
            y0i += sy;
        }
    }
}

fn draw_text_fit(frame: &mut [u32], rect: Rect, text: &str, color: u32, max_scale: usize) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let max_scale = max_scale.max(1);
    for scale in (1..=max_scale).rev() {
        if text_pixel_width(text, scale) <= rect.w && 8 * scale <= rect.h {
            draw_text_clipped(frame, rect.x, rect.y, text, color, scale, rect);
            return;
        }
    }

    let available_chars = (rect.w / 9).max(1);
    let mut clipped = String::new();
    for ch in text.chars().take(available_chars.saturating_sub(2)) {
        clipped.push(ch);
    }
    if text.chars().count() > clipped.chars().count() {
        clipped.push_str("..");
    }
    draw_text_clipped(frame, rect.x, rect.y, &clipped, color, 1, rect);
}

fn draw_wrapped_text(
    frame: &mut [u32],
    rect: Rect,
    text: &str,
    color: u32,
    scale: usize,
    max_lines: usize,
) {
    if rect.w == 0 || rect.h == 0 || max_lines == 0 {
        return;
    }
    let scale = scale.max(1);
    let line_h = 10 * scale;
    let max_lines = max_lines.min((rect.h / line_h).max(1));
    let mut line = String::new();
    let mut y = rect.y;
    let mut lines_drawn = 0usize;

    for word in text.split_whitespace() {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            format!("{line} {word}")
        };
        if text_pixel_width(&candidate, scale) <= rect.w {
            line = candidate;
            continue;
        }
        if !line.is_empty() {
            draw_text_fit(
                frame,
                Rect::new(rect.x, y, rect.w, line_h),
                &line,
                color,
                scale,
            );
            lines_drawn += 1;
            y += line_h;
            if lines_drawn >= max_lines {
                return;
            }
        }
        line = word.to_string();
    }

    if !line.is_empty() && lines_drawn < max_lines {
        draw_text_fit(
            frame,
            Rect::new(rect.x, y, rect.w, line_h),
            &line,
            color,
            scale,
        );
    }
}

fn draw_text_clipped(
    frame: &mut [u32],
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    scale: usize,
    clip: Rect,
) {
    let scale = scale.max(1);
    let mut cursor_x = x;
    for ch in text.chars() {
        if ch == '\n' {
            cursor_x = x;
            continue;
        }
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits >> col) & 1 == 1 {
                        let px = cursor_x + col * scale;
                        let py = y + row * scale;
                        fill_rect(
                            frame,
                            intersect_rect(Rect::new(px, py, scale, scale), clip),
                            color,
                        );
                    }
                }
            }
        }
        cursor_x += 8 * scale + scale;
        if cursor_x >= WIDTH.saturating_sub(10) {
            break;
        }
    }
}

fn text_pixel_width(text: &str, scale: usize) -> usize {
    let scale = scale.max(1);
    text.chars().count() * (8 * scale + scale)
}

fn intersect_rect(a: Rect, b: Rect) -> Rect {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    if x2 <= x1 || y2 <= y1 {
        Rect::new(0, 0, 0, 0)
    } else {
        Rect::new(x1, y1, x2 - x1, y2 - y1)
    }
}

fn accent_for(kind: FeatureKind) -> u32 {
    match kind {
        FeatureKind::Overview => 0xe3e6ea,
        FeatureKind::RuntimeBasics => CYAN,
        FeatureKind::LaunchContracts => AMBER,
        FeatureKind::VisualKernels => MAGENTA,
        FeatureKind::LayoutClosure => GREEN,
        FeatureKind::MathIntrinsics => CYAN,
        FeatureKind::ReturnValue => CYAN,
        FeatureKind::CastMatrix => MAGENTA,
        FeatureKind::HostReference => GREEN,
        FeatureKind::SyncScope => AMBER,
        FeatureKind::LdsCollectives => 0x65d46e,
        FeatureKind::DeviceApi => 0x42d9ff,
        FeatureKind::GraphOperations => MAGENTA,
        FeatureKind::RocmLibraries => 0xf1b447,
        FeatureKind::DebugInfo => 0xe3e6ea,
    }
}

fn blend(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ar = ((a >> 16) & 0xff) as f32;
    let ag = ((a >> 8) & 0xff) as f32;
    let ab = (a & 0xff) as f32;
    let br = ((b >> 16) & 0xff) as f32;
    let bg = ((b >> 8) & 0xff) as f32;
    let bb = (b & 0xff) as f32;
    let r = (ar + (br - ar) * t) as u32;
    let g = (ag + (bg - ag) * t) as u32;
    let blue = (ab + (bb - ab) * t) as u32;
    (r << 16) | (g << 8) | blue
}

fn expected_reference_closure(input: &[u32], bias: u32, scale: u32) -> Vec<u32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .wrapping_add((index as u32) & 1)
                .wrapping_mul(scale)
                .wrapping_add(bias)
        })
        .collect()
}

fn return_rust_pair_host(value: u32, params: generated::ControlParams) -> (u32, u64) {
    let scale = params.scale.unsigned_abs();
    let rotation = (value & 7).wrapping_add(1);
    let left = value.wrapping_add(params.seed).rotate_left(rotation);
    let right = ((value as u64) << 32)
        .wrapping_add(scale as u64)
        .wrapping_add((params.seed as u64).wrapping_mul(17));
    (left, right)
}

fn return_packet_host(value: u32, params: generated::ControlParams) -> generated::ReturnPacket {
    let (left, right) = return_rust_pair_host(value, params);
    let shift = (value & 3) * 8;
    let lane_mix = (right >> shift) as u32;
    generated::ReturnPacket {
        sum: right.wrapping_add(left as u64),
        folded: left ^ lane_mix.rotate_right(value & 15),
        tag: 0xc0de_0000u32 ^ (value & 0xff) ^ ((right >> 48) as u32),
    }
}

fn return_packet_score_host(packet: generated::ReturnPacket) -> u64 {
    packet.sum ^ ((packet.folded as u64) << 16) ^ packet.tag as u64
}

fn cast_packet_host(value: u32, index: usize) -> generated::CastPacket {
    let signed = (value as i64).wrapping_sub(0x1234).wrapping_mul(-33);
    let wide = ((value as u64) << 37)
        .wrapping_add((index as u64).wrapping_mul(0x1f1f_0101))
        .rotate_left(value & 31);
    let float_value = (signed as f32) * 0.125 + index as f32;
    let double_value = f64::from_bits(0x3ff0_0000_0000_0000u64 | (wide & 0x000f_ffff_ffff_ffff));
    let narrow = (wide as u32)
        .wrapping_add(signed as u32)
        .wrapping_add(float_value as i32 as u32);
    let float_bits = float_value.to_bits() ^ (double_value.to_bits() as u32).rotate_left(11);
    generated::CastPacket {
        wide,
        signed_bits: signed as u64,
        float_bits,
        narrow,
    }
}

fn cast_packet_score_host(packet: generated::CastPacket) -> u64 {
    packet
        .wide
        .wrapping_add(packet.signed_bits.rotate_left(7))
        .wrapping_add((packet.float_bits as u64) << 1)
        .wrapping_add(packet.narrow as u64)
}
