use font8x8::{BASIC_FONTS, UnicodeFonts};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rocm_oxide::{
    Device, DeviceBuffer, Dim3, HostReferenceCaptureVisibility, LaunchConfig, ManagedBuffer,
    ManagedMemoryKind, PinnedHostBuffer,
};
use std::time::{Duration, Instant};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1180;
const HEIGHT: usize = 720;
const SIDEBAR_W: usize = 330;
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
    ReturnValue,
    CastMatrix,
    HostReference,
    SyncScope,
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
                Key::Key1 => state.selected = 0.min(state.probes.len().saturating_sub(1)),
                Key::Key2 => state.selected = 1.min(state.probes.len().saturating_sub(1)),
                Key::Key3 => state.selected = 2.min(state.probes.len().saturating_sub(1)),
                Key::Key4 => state.selected = 3.min(state.probes.len().saturating_sub(1)),
                Key::Key5 => state.selected = 4.min(state.probes.len().saturating_sub(1)),
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
    let mut probes = Vec::new();
    probes.push(run_return_probe(&kernels)?);
    probes.push(run_cast_probe(&kernels)?);
    probes.push(run_host_reference_probe(&device, &kernels)?);
    probes.push(run_syncscope_probe(&kernels)?);
    probes.push(debug_info_probe(&device));
    Ok(probes)
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

    draw_text(frame, 26, 26, "Compiler Feature Lab", TEXT, 2);
    draw_text(frame, 28, 60, "ROCm-Oxide AMDGPU parity probes", CYAN, 1);
    draw_text(
        frame,
        28,
        84,
        "clicks use scaled framebuffer coordinates",
        MUTED,
        1,
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
        draw_text(frame, rect.x + 14, rect.y + 12, &probe.title, TEXT, 1);
        draw_text(
            frame,
            rect.x + rect.w - 62,
            rect.y + 13,
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
    draw_text(frame, 30, HEIGHT - 92, "Signal scale", TEXT, 1);
    let slider = Rect::new(30, HEIGHT - 72, SIDEBAR_W - 60, 22);
    stroke_rect(frame, slider, 0x626970);
    let fill_w = ((state.signal_scale - 0.4) / 2.1 * slider.w as f32) as usize;
    fill_rect(
        frame,
        Rect::new(slider.x + 2, slider.y + 2, fill_w, slider.h - 4),
        CYAN,
    );
    draw_text(
        frame,
        30,
        HEIGHT - 38,
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
    fill_rect(frame, Rect::new(PANEL_X, 28, PANEL_W, 116), PANEL);
    stroke_rect(frame, Rect::new(PANEL_X, 28, PANEL_W, 116), accent);
    draw_text(frame, PANEL_X + 24, 48, &probe.title, TEXT, 2);
    draw_text(
        frame,
        PANEL_X + 24,
        88,
        &format!("status: {}", probe.status),
        if probe.ok { GREEN } else { RED },
        1,
    );
    draw_text(frame, PANEL_X + 24, 112, &probe.detail, MUTED, 1);

    let plot = Rect::new(PANEL_X, 170, PANEL_W, 332);
    fill_rect(frame, plot, 0x141414);
    stroke_rect(frame, plot, 0x424850);
    draw_grid(frame, plot, elapsed);
    match probe.kind {
        FeatureKind::ReturnValue => draw_return_view(frame, plot, probe, scale, elapsed),
        FeatureKind::CastMatrix => draw_cast_view(frame, plot, probe, scale, elapsed),
        FeatureKind::HostReference => draw_host_ref_view(frame, plot, probe, scale),
        FeatureKind::SyncScope => draw_syncscope_view(frame, plot, probe, scale),
        FeatureKind::DebugInfo => draw_debug_view(frame, plot, probe, scale),
    }

    let inspector = Rect::new(PANEL_X, 532, PANEL_W, 150);
    fill_rect(frame, inspector, PANEL_2);
    stroke_rect(frame, inspector, 0x555b61);
    draw_text(
        frame,
        inspector.x + 22,
        inspector.y + 18,
        "Feature signals",
        TEXT,
        1,
    );
    draw_text(
        frame,
        inspector.x + 22,
        inspector.y + 42,
        "Each band is drawn from the GPU probe payload or build artifact state.",
        MUTED,
        1,
    );
    for (row, chunk) in probe.values.chunks(8).take(4).enumerate() {
        let y = inspector.y + 70 + row * 18;
        let mut text = String::new();
        for value in chunk {
            text.push_str(&format!("{:08x} ", (value & 0xffff_ffff) as u32));
        }
        draw_text(frame, inspector.x + 22, y, text.trim_end(), accent, 1);
    }
    if let Some((mx, my)) = mouse {
        draw_text(
            frame,
            inspector.x + inspector.w - 190,
            inspector.y + 18,
            &format!("mouse fb {},{}", mx, my),
            MUTED,
            1,
        );
    }
}

fn draw_return_view(
    frame: &mut [u32],
    rect: Rect,
    probe: &FeatureProbe,
    scale: f32,
    elapsed: Duration,
) {
    let pulse = ((elapsed.as_secs_f32() * 2.0).sin() * 0.5 + 0.5) as f32;
    draw_value_bars(frame, rect, &probe.values, scale, CYAN, AMBER);
    for (index, value) in probe.values.iter().take(8).enumerate() {
        let x = rect.x + 46 + index * ((rect.w - 92) / 8);
        let y = rect.y + 54 + ((*value as usize ^ index * 31) % 54);
        let color = blend(CYAN, GREEN, pulse);
        fill_rect(frame, Rect::new(x, y, 34, 34), color);
        stroke_rect(frame, Rect::new(x, y, 34, 34), TEXT);
        draw_text(frame, x + 6, y + 12, "RET", 0x111111, 1);
    }
    draw_text(
        frame,
        rect.x + 28,
        rect.y + 26,
        "function returns -> packet fields",
        TEXT,
        1,
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
    draw_text(
        frame,
        rect.x + 28,
        rect.y + 26,
        "integer, float, pointer-cast, and bitcast lanes",
        TEXT,
        1,
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
        draw_text(frame, x + 12, y + 16, label, TEXT, 1);
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
        draw_text(frame, x + 18, rect.y + 226, label, TEXT, 1);
    }
    draw_text(
        frame,
        rect.x + 28,
        rect.y + 26,
        "source markers lower to AMDGPU syncscope choices",
        TEXT,
        1,
    );
}

fn draw_debug_view(frame: &mut [u32], rect: Rect, probe: &FeatureProbe, scale: f32) {
    draw_value_bars(frame, rect, &probe.values, scale, AMBER, CYAN);
    let nodes = ["rustc -g", "LLVM IR", "llc -g", "clang -g", "HSACO"];
    for (index, node) in nodes.iter().enumerate() {
        let x = rect.x + 42 + index * 150;
        let y = rect.y + 108 + (index % 2) * 42;
        fill_rect(frame, Rect::new(x, y, 116, 46), 0x302a1d);
        stroke_rect(frame, Rect::new(x, y, 116, 46), AMBER);
        draw_text(frame, x + 10, y + 16, node, TEXT, 1);
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
        .map(|index| Rect::new(26, 128 + index * 62, SIDEBAR_W - 52, 46))
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
    draw_text(frame, rect.x + 20, rect.y + 12, label, TEXT, 1);
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

fn draw_text(frame: &mut [u32], x: usize, y: usize, text: &str, color: u32, scale: usize) {
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
                        fill_rect(
                            frame,
                            Rect::new(cursor_x + col * scale, y + row * scale, scale, scale),
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

fn accent_for(kind: FeatureKind) -> u32 {
    match kind {
        FeatureKind::ReturnValue => CYAN,
        FeatureKind::CastMatrix => MAGENTA,
        FeatureKind::HostReference => GREEN,
        FeatureKind::SyncScope => AMBER,
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
