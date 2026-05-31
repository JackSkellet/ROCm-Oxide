use font8x8::{BASIC_FONTS, UnicodeFonts};
use image::{Rgb, RgbImage};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rocm_oxide::{
    Device, DeviceBuffer, LaunchConfig, RocBlas, RocmLibraryReport, SgemmLayout,
    rocm_feature_parity_for_device,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const PANEL_W: usize = 318;
const BLOCK_X: u32 = 256;
const DEFAULT_OUTPUT: &str = "target/spectral_lattice.png";
const MODES: [&str; 4] = ["Core", "LDS", "Atomic", "Chain"];
const FPS_LIMITS: [usize; 7] = [30, 60, 90, 120, 144, 240, 0];
const RESOLUTION_PRESETS: [ResolutionPreset; 5] = [
    ResolutionPreset::new("540p", 960, 540),
    ResolutionPreset::new("720p", 1280, 720),
    ResolutionPreset::new("1080p", 1920, 1080),
    ResolutionPreset::new("1440p", 2560, 1440),
    ResolutionPreset::new("4K", 3840, 2160),
];

struct DemoArgs {
    frames: Option<u32>,
    output: PathBuf,
    mode: Option<usize>,
    size: RenderSize,
    fps_limit: usize,
}

struct PaletteSeed {
    values: [f32; 4],
    source: String,
}

struct DemoState {
    mode: usize,
    speed: f32,
    warp: f32,
    gain: f32,
    paused: bool,
    auto_cycle: bool,
    frame_index: u32,
    palette_seed: u32,
    palette: [f32; 4],
    palette_source: String,
    status: String,
    fps: f64,
    fps_limit: usize,
    render_size: RenderSize,
    save_requested: bool,
}

struct ResourceSnapshot {
    resource_line: String,
    launch_line: String,
    library_line: String,
    parity_line: String,
}

struct DemoBuffers {
    size: RenderSize,
    base: DeviceBuffer<u32>,
    post: DeviceBuffer<u32>,
    short: DeviceBuffer<u32>,
    tile_stats: DeviceBuffer<u32>,
    histogram: DeviceBuffer<u32>,
    histogram_zero: Vec<u32>,
    tile_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RenderSize {
    width: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug)]
struct ResolutionPreset {
    label: &'static str,
    size: RenderSize,
}

#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl RenderSize {
    const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    const fn pixel_count(self) -> usize {
        self.width * self.height
    }

    fn label(self) -> String {
        RESOLUTION_PRESETS
            .iter()
            .find(|preset| preset.size == self)
            .map_or_else(
                || format!("{}x{}", self.width, self.height),
                |preset| preset.label.to_string(),
            )
    }
}

impl ResolutionPreset {
    const fn new(label: &'static str, width: usize, height: usize) -> Self {
        Self {
            label,
            size: RenderSize::new(width, height),
        }
    }
}

impl Rect {
    const fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self { x, y, w, h }
    }

    fn contains(self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let mut buffers = DemoBuffers::new(args.size)?;
    let mut host_frame = vec![0u32; buffers.size.pixel_count()];
    let mut resources = ResourceSnapshot::new(&device, &kernels, buffers.size.pixel_count())?;
    let palette = derive_palette(0);
    let mut state = DemoState {
        mode: 0,
        speed: 1.0,
        warp: palette.values[3],
        gain: 1.0,
        paused: false,
        auto_cycle: false,
        frame_index: 0,
        palette_seed: 0,
        palette: palette.values,
        palette_source: palette.source,
        status: "ready: core + libs + contracts".into(),
        fps: 0.0,
        fps_limit: args.fps_limit,
        render_size: args.size,
        save_requested: false,
    };
    if let Some(mode) = args.mode {
        set_mode(&mut state, mode);
    }

    if let Some(frames) = args.frames {
        let frames = frames.max(1);
        let mut use_post = false;
        for frame in 0..frames {
            state.frame_index = ((frame as f32) * state.speed * 2.0) as u32;
            if state.auto_cycle {
                state.mode = ((state.frame_index / 180) as usize) % MODES.len();
            }
            use_post = render_frame(&kernels, &buffers, &state)?;
        }
        rocm_oxide::hip::synchronize()?;
        copy_display_frame(&buffers, use_post, &mut host_frame)?;
        draw_overlay(&mut host_frame, buffers.size, &state, &resources);
        save_png(&args.output, &host_frame, buffers.size)?;
        println!(
            "saved Spectral Lattice GUI preview after {frames} frame(s): {}",
            args.output.display()
        );
        return Ok(());
    }

    let mut window = create_window(buffers.size, state.fps_limit)?;

    let start = Instant::now();
    let mut last_fps = Instant::now();
    let mut frames_since_fps = 0u32;
    let mut mouse_was_down = false;
    let mut saved_once = false;
    let mut applied_fps_limit = state.fps_limit;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        handle_keyboard(&window, &mut state, &kernels, &buffers.short);
        handle_mouse(
            &window,
            &mut state,
            &kernels,
            &buffers.short,
            &mut mouse_was_down,
        );

        if state.render_size != buffers.size {
            buffers = DemoBuffers::new(state.render_size)?;
            host_frame.resize(buffers.size.pixel_count(), 0);
            resources = ResourceSnapshot::new(&device, &kernels, buffers.size.pixel_count())?;
            window = create_window(buffers.size, state.fps_limit)?;
            applied_fps_limit = state.fps_limit;
            saved_once = false;
            state.status = format!("resolution set to {}", buffers.size.label());
        } else if state.fps_limit != applied_fps_limit {
            window.set_target_fps(state.fps_limit);
            applied_fps_limit = state.fps_limit;
            state.status = format!("FPS limit set to {}", fps_label(state.fps_limit));
        }

        if !state.paused {
            state.frame_index = (start.elapsed().as_secs_f32() * 60.0 * state.speed) as u32;
            if state.auto_cycle {
                state.mode = ((state.frame_index / 180) as usize) % MODES.len();
            }
        }

        let use_post = render_frame(&kernels, &buffers, &state)?;
        rocm_oxide::hip::synchronize()?;
        copy_display_frame(&buffers, use_post, &mut host_frame)?;
        draw_overlay(&mut host_frame, buffers.size, &state, &resources);

        if !saved_once || state.save_requested {
            save_png(&args.output, &host_frame, buffers.size)?;
            state.status = format!("saved {}", args.output.display());
            state.save_requested = false;
            saved_once = true;
        }

        window.update_with_buffer(&host_frame, buffers.size.width, buffers.size.height)?;
        frames_since_fps = frames_since_fps.saturating_add(1);
        if last_fps.elapsed() >= Duration::from_millis(500) {
            state.fps = frames_since_fps as f64 / last_fps.elapsed().as_secs_f64();
            frames_since_fps = 0;
            last_fps = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide Spectral Lattice | {} | {:.1} FPS | limit {} | {}",
                buffers.size.label(),
                state.fps,
                fps_label(state.fps_limit),
                MODES[state.mode]
            ));
        }
    }

    Ok(())
}

fn create_window(size: RenderSize, fps_limit: usize) -> Result<Window, Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "ROCm-Oxide Spectral Lattice",
        size.width,
        size.height,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(fps_limit);
    Ok(window)
}

fn copy_display_frame(
    buffers: &DemoBuffers,
    use_post: bool,
    host_frame: &mut [u32],
) -> Result<(), Box<dyn std::error::Error>> {
    if use_post {
        buffers.post.copy_to_host(host_frame)?;
    } else {
        buffers.base.copy_to_host(host_frame)?;
    }
    Ok(())
}

impl ResourceSnapshot {
    fn new(
        device: &Device,
        kernels: &generated::DeviceKernels,
        pixel_count: usize,
    ) -> rocm_oxide::Result<Self> {
        let props = device.properties()?;
        let parity = rocm_feature_parity_for_device(props);
        let libraries = RocmLibraryReport::query();
        let resource = kernels.resource("spectral_lattice").ok_or_else(|| {
            rocm_oxide::Error::InvalidLaunch("missing spectral_lattice resource".into())
        })?;
        let recommendation = kernels.recommend_1d_launch("spectral_lattice", pixel_count, 0, 0)?;
        Ok(Self {
            resource_line: format!(
                "kernels:{} core:{}v/{}s/w{}",
                kernels.resources().len(),
                opt_u32(resource.vgpr_count),
                opt_u32(resource.sgpr_count),
                opt_u32(resource.wavefront_size),
            ),
            launch_line: format!(
                "launch: block{} active{} waves{}",
                recommendation.block_size,
                recommendation.active_blocks_per_multiprocessor,
                recommendation.waves_per_block.unwrap_or(0)
            ),
            library_line: format!(
                "libs: rocBLAS={} rocFFT={}",
                on_off(libraries.rocblas.available),
                on_off(libraries.rocfft.available)
            ),
            parity_line: format!(
                "{}",
                if parity.cluster_launch.requires_runtime_capability {
                    "parity: cooperative launch path"
                } else {
                    "parity: stream tiled path"
                }
            ),
        })
    }
}

impl DemoBuffers {
    fn new(size: RenderSize) -> rocm_oxide::Result<Self> {
        let pixel_count = size.pixel_count();
        let tile_count = pixel_count.div_ceil(BLOCK_X as usize);
        Ok(Self {
            size,
            base: DeviceBuffer::<u32>::new(pixel_count)?,
            post: DeviceBuffer::<u32>::new(pixel_count)?,
            short: DeviceBuffer::<u32>::new(pixel_count / 2)?,
            tile_stats: DeviceBuffer::<u32>::new(tile_count)?,
            histogram: DeviceBuffer::<u32>::new(256)?,
            histogram_zero: vec![0u32; 256],
            tile_count,
        })
    }
}

fn render_frame(
    kernels: &generated::DeviceKernels,
    buffers: &DemoBuffers,
    state: &DemoState,
) -> rocm_oxide::Result<bool> {
    unsafe {
        kernels.spectral_lattice(
            LaunchConfig::for_num_elems_with_block_size(buffers.size.pixel_count(), BLOCK_X),
            &buffers.base,
            buffers.size.width as u32,
            buffers.size.height as u32,
            buffers.size.pixel_count(),
            state.frame_index,
            state.mode as u32,
            state.palette[0],
            state.palette[1],
            state.palette[2],
            state.warp,
            state.gain,
        )?;
    }

    match state.mode {
        1 => {
            let config =
                LaunchConfig::for_num_elems_with_block_size(buffers.size.pixel_count(), BLOCK_X)
                    .try_with_dynamic_shared_mem::<u32>(BLOCK_X as usize)?;
            unsafe {
                kernels.spectral_lds_tiles(
                    config,
                    &buffers.post,
                    &buffers.base,
                    &buffers.tile_stats,
                    buffers.size.pixel_count(),
                    buffers.tile_count,
                    BLOCK_X,
                    state.mode as u32,
                )?;
            }
            Ok(true)
        }
        2 => {
            buffers.histogram.copy_from_host(&buffers.histogram_zero)?;
            unsafe {
                kernels.spectral_atomic_histogram(
                    LaunchConfig::for_num_elems_with_block_size(
                        buffers.size.pixel_count(),
                        BLOCK_X,
                    ),
                    &buffers.histogram,
                    &buffers.base,
                    buffers.size.pixel_count(),
                )?;
                kernels.spectral_histogram_overlay(
                    LaunchConfig::for_num_elems_with_block_size(
                        buffers.size.pixel_count(),
                        BLOCK_X,
                    ),
                    &buffers.post,
                    &buffers.base,
                    &buffers.histogram,
                    buffers.size.width as u32,
                    buffers.size.height as u32,
                    buffers.size.pixel_count(),
                    state.frame_index,
                )?;
            }
            Ok(true)
        }
        3 => {
            unsafe {
                kernels.spectral_post_fx(
                    LaunchConfig::for_num_elems_with_block_size(
                        buffers.size.pixel_count(),
                        BLOCK_X,
                    ),
                    &buffers.post,
                    &buffers.base,
                    buffers.size.width as u32,
                    buffers.size.height as u32,
                    buffers.size.pixel_count(),
                    state.frame_index,
                    state.mode as u32,
                    state.gain,
                )?;
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn handle_keyboard(
    window: &Window,
    state: &mut DemoState,
    kernels: &generated::DeviceKernels,
    short_frame: &DeviceBuffer<u32>,
) {
    for key in window.get_keys_pressed(KeyRepeat::No) {
        match key {
            Key::Key1 => set_mode(state, 0),
            Key::Key2 => set_mode(state, 1),
            Key::Key3 => set_mode(state, 2),
            Key::Key4 => set_mode(state, 3),
            Key::Left => set_mode(state, (state.mode + MODES.len() - 1) % MODES.len()),
            Key::Right => set_mode(state, (state.mode + 1) % MODES.len()),
            Key::Up => state.warp = clamp_f32(state.warp + 0.08, 0.05, 2.25),
            Key::Down => state.warp = clamp_f32(state.warp - 0.08, 0.05, 2.25),
            Key::PageUp => state.speed = clamp_f32(state.speed + 0.15, 0.1, 3.0),
            Key::PageDown => state.speed = clamp_f32(state.speed - 0.15, 0.1, 3.0),
            Key::Minus => step_fps_limit(state, -1),
            Key::Equal => step_fps_limit(state, 1),
            Key::Comma => cycle_resolution(state, -1),
            Key::Period => cycle_resolution(state, 1),
            Key::Space => state.paused = !state.paused,
            Key::A => state.auto_cycle = !state.auto_cycle,
            Key::R => reseed_palette(state),
            Key::C => run_contract_check(state, kernels, short_frame),
            Key::S => state.save_requested = true,
            _ => {}
        }
    }
}

fn handle_mouse(
    window: &Window,
    state: &mut DemoState,
    kernels: &generated::DeviceKernels,
    short_frame: &DeviceBuffer<u32>,
    mouse_was_down: &mut bool,
) {
    let mouse_down = window.get_mouse_down(MouseButton::Left);
    let mouse_clicked = mouse_down && !*mouse_was_down;
    if let Some((x, y)) = buffer_mouse_pos(window, state.render_size) {
        if mouse_down {
            handle_slider_drag(state, x, y);
        }
        if mouse_clicked {
            handle_click(state, kernels, short_frame, x, y);
        }
    }
    *mouse_was_down = mouse_down;
}

fn buffer_mouse_pos(window: &Window, size: RenderSize) -> Option<(usize, usize)> {
    let (mx, my) = window.get_unscaled_mouse_pos(MouseMode::Discard)?;
    let (win_w, win_h) = window.get_size();
    if win_w == 0 || win_h == 0 {
        return None;
    }

    let x = ((mx.max(0.0) as f64) * size.width as f64 / win_w as f64)
        .floor()
        .clamp(0.0, (size.width - 1) as f64) as usize;
    let y = ((my.max(0.0) as f64) * size.height as f64 / win_h as f64)
        .floor()
        .clamp(0.0, (size.height - 1) as f64) as usize;
    Some((x, y))
}

fn handle_click(
    state: &mut DemoState,
    kernels: &generated::DeviceKernels,
    short_frame: &DeviceBuffer<u32>,
    x: usize,
    y: usize,
) {
    let scale = ui_scale(state.render_size);
    for index in 0..MODES.len() {
        if scale_rect(mode_rect(index), scale).contains(x, y) {
            set_mode(state, index);
            state.auto_cycle = false;
            return;
        }
    }

    if scale_rect(button_rect(0), scale).contains(x, y) {
        reseed_palette(state);
    } else if scale_rect(button_rect(1), scale).contains(x, y) {
        run_contract_check(state, kernels, short_frame);
    } else if scale_rect(button_rect(2), scale).contains(x, y) {
        state.paused = !state.paused;
    } else if scale_rect(button_rect(3), scale).contains(x, y) {
        state.auto_cycle = !state.auto_cycle;
    } else if scale_rect(button_rect(4), scale).contains(x, y) {
        state.save_requested = true;
    } else if scale_rect(button_rect(5), scale).contains(x, y) {
        cycle_resolution(state, 1);
    }
}

fn set_mode(state: &mut DemoState, mode: usize) {
    state.mode = mode % MODES.len();
    state.status = mode_detail(state.mode).into();
}

fn handle_slider_drag(state: &mut DemoState, x: usize, y: usize) {
    let scale = ui_scale(state.render_size);
    if scale_rect(slider_rect(0), scale).contains(x, y) {
        state.warp = slider_value(x, scale_rect(slider_rect(0), scale), 0.05, 2.25);
    } else if scale_rect(slider_rect(1), scale).contains(x, y) {
        state.gain = slider_value(x, scale_rect(slider_rect(1), scale), 0.35, 1.8);
    } else if scale_rect(slider_rect(2), scale).contains(x, y) {
        state.speed = slider_value(x, scale_rect(slider_rect(2), scale), 0.1, 3.0);
    } else if scale_rect(slider_rect(3), scale).contains(x, y) {
        set_fps_limit_from_slider(state, x);
    }
}

fn cycle_resolution(state: &mut DemoState, delta: isize) {
    let current = resolution_preset_index(state.render_size).unwrap_or_else(|| {
        RESOLUTION_PRESETS
            .iter()
            .enumerate()
            .min_by_key(|(_, preset)| {
                preset
                    .size
                    .pixel_count()
                    .abs_diff(state.render_size.pixel_count())
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    });
    let len = RESOLUTION_PRESETS.len() as isize;
    let next = (current as isize + delta).rem_euclid(len) as usize;
    state.render_size = RESOLUTION_PRESETS[next].size;
}

fn resolution_preset_index(size: RenderSize) -> Option<usize> {
    RESOLUTION_PRESETS
        .iter()
        .position(|preset| preset.size == size)
}

fn set_fps_limit_from_slider(state: &mut DemoState, x: usize) {
    let rect = scale_rect(slider_rect(3), ui_scale(state.render_size));
    let t = ((x.saturating_sub(rect.x)) as f32 / rect.w.max(1) as f32).clamp(0.0, 1.0);
    let index = (t * (FPS_LIMITS.len() - 1) as f32).round() as usize;
    state.fps_limit = FPS_LIMITS[index.min(FPS_LIMITS.len() - 1)];
}

fn step_fps_limit(state: &mut DemoState, delta: isize) {
    let current = fps_limit_index(state.fps_limit);
    let len = FPS_LIMITS.len() as isize;
    let next = (current as isize + delta).rem_euclid(len) as usize;
    state.fps_limit = FPS_LIMITS[next];
}

fn fps_limit_index(limit: usize) -> usize {
    FPS_LIMITS
        .iter()
        .position(|value| *value == limit)
        .unwrap_or_else(|| {
            FPS_LIMITS
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0)
                .min_by_key(|(_, value)| value.abs_diff(limit))
                .map(|(index, _)| index)
                .unwrap_or(1)
        })
}

fn fps_label(limit: usize) -> String {
    if limit == 0 {
        "uncapped".into()
    } else {
        format!("{limit}")
    }
}

fn reseed_palette(state: &mut DemoState) {
    state.palette_seed = state.palette_seed.wrapping_add(1);
    let palette = derive_palette(state.palette_seed);
    state.palette = palette.values;
    state.warp = palette.values[3];
    state.palette_source = palette.source;
    state.status = format!("palette reseeded through {}", state.palette_source);
}

fn run_contract_check(
    state: &mut DemoState,
    kernels: &generated::DeviceKernels,
    short_frame: &DeviceBuffer<u32>,
) {
    let result = unsafe {
        kernels.spectral_lattice(
            LaunchConfig::for_num_elems_with_block_size(state.render_size.pixel_count(), BLOCK_X),
            short_frame,
            state.render_size.width as u32,
            state.render_size.height as u32,
            state.render_size.pixel_count(),
            state.frame_index,
            state.mode as u32,
            state.palette[0],
            state.palette[1],
            state.palette[2],
            state.warp,
            state.gain,
        )
    };
    state.status = match result {
        Err(rocm_oxide::Error::InvalidLaunch(err)) => {
            format!("contract guard passed: {err}")
        }
        Err(err) => format!("unexpected contract check error: {err}"),
        Ok(()) => "contract guard failed: short frame launched".into(),
    };
}

fn derive_palette(seed: u32) -> PaletteSeed {
    match derive_palette_with_rocblas(seed) {
        Ok(values) => PaletteSeed {
            values,
            source: "rocBLAS SGEMM".into(),
        },
        Err(err) => PaletteSeed {
            values: fallback_palette(seed),
            source: format!("fallback ({err})"),
        },
    }
}

fn derive_palette_with_rocblas(seed: u32) -> rocm_oxide::Result<[f32; 4]> {
    let phase_seed = seed as f32 * 0.137;
    let blas = RocBlas::open()?;
    let handle = blas.create_handle()?;
    let a = DeviceBuffer::from_slice(&[
        0.82f32 + phase_seed.sin() * 0.07,
        0.27 + phase_seed.cos() * 0.05,
        0.16 + (phase_seed * 1.7).sin() * 0.06,
        0.91 + (phase_seed * 1.3).cos() * 0.04,
    ])?;
    let b = DeviceBuffer::from_slice(&[
        1.03f32 + (phase_seed * 0.7).cos() * 0.05,
        0.21 + (phase_seed * 1.9).sin() * 0.04,
        0.34 + phase_seed.cos() * 0.08,
        0.79 + phase_seed.sin() * 0.06,
    ])?;
    let c = DeviceBuffer::<f32>::new(4)?;
    handle.sgemm_nn(SgemmLayout::column_major(2, 2, 2)?, 1.0, &a, &b, 0.0, &c)?;
    rocm_oxide::hip::synchronize()?;
    let out = c.copy_to_vec()?;
    Ok([
        phase(out[0] * 1.7 + phase_seed),
        phase(out[1] * 2.1 + 0.4),
        phase(out[2] * 2.6 + 1.2),
        0.7 + out[3].abs().fract() * 0.9,
    ])
}

fn fallback_palette(seed: u32) -> [f32; 4] {
    let seed = seed as f32 * 0.37;
    [
        phase(1.1 + seed),
        phase(2.8 + seed * 0.7),
        phase(4.6 + seed * 1.3),
        0.8 + (seed.sin() + 1.0) * 0.35,
    ]
}

fn draw_overlay(
    frame: &mut [u32],
    size: RenderSize,
    state: &DemoState,
    resources: &ResourceSnapshot,
) {
    let scale = ui_scale(size);
    let panel_w = PANEL_W * scale;
    blend_rect(frame, size, 0, 0, panel_w, size.height, 0x071018, 224);
    blend_rect(
        frame,
        size,
        12 * scale,
        12 * scale,
        (PANEL_W - 24) * scale,
        82 * scale,
        0x102030,
        190,
    );
    draw_text(
        frame,
        size,
        scale,
        24 * scale,
        24 * scale,
        "Spectral Lattice",
        0xffffff,
    );
    draw_text(
        frame,
        size,
        scale,
        24 * scale,
        44 * scale,
        "ROCm-Oxide Visual Workbench",
        0x93dcff,
    );
    draw_text(
        frame,
        size,
        scale,
        24 * scale,
        66 * scale,
        &format!(
            "mode={} fps={:.1} limit={}",
            MODES[state.mode],
            state.fps,
            fps_label(state.fps_limit)
        ),
        0xdce8f4,
    );

    for index in 0..MODES.len() {
        let rect = scale_rect(mode_rect(index), scale);
        let active = index == state.mode;
        draw_button(
            frame,
            size,
            scale,
            rect,
            MODES[index],
            if active { 0x2b8ee8 } else { 0x1a3045 },
            active,
        );
    }

    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        146 * scale,
        mode_detail(state.mode),
        0x9adfb1,
        (PANEL_W - 42) * scale,
    );

    draw_text(
        frame,
        size,
        scale,
        24 * scale,
        160 * scale,
        "Controls",
        0xffffff,
    );
    draw_button(
        frame,
        size,
        scale,
        scale_rect(button_rect(0), scale),
        "R BLAS Palette",
        0x244b3b,
        false,
    );
    draw_button(
        frame,
        size,
        scale,
        scale_rect(button_rect(1), scale),
        "C Contract",
        0x483842,
        false,
    );
    draw_button(
        frame,
        size,
        scale,
        scale_rect(button_rect(2), scale),
        if state.paused { "Resume" } else { "Pause" },
        0x3a3e58,
        state.paused,
    );
    draw_button(
        frame,
        size,
        scale,
        scale_rect(button_rect(3), scale),
        if state.auto_cycle {
            "Auto On"
        } else {
            "Auto Off"
        },
        0x344733,
        state.auto_cycle,
    );
    draw_button(
        frame,
        size,
        scale,
        scale_rect(button_rect(4), scale),
        "S Save",
        0x3f384f,
        false,
    );
    draw_button(
        frame,
        size,
        scale,
        scale_rect(button_rect(5), scale),
        &format!("Res {}", state.render_size.label()),
        0x34485f,
        false,
    );

    draw_slider(
        frame,
        size,
        scale,
        "Warp",
        scale_rect(slider_rect(0), scale),
        state.warp,
        0.05,
        2.25,
    );
    draw_slider(
        frame,
        size,
        scale,
        "Gain",
        scale_rect(slider_rect(1), scale),
        state.gain,
        0.35,
        1.8,
    );
    draw_slider(
        frame,
        size,
        scale,
        "Speed",
        scale_rect(slider_rect(2), scale),
        state.speed,
        0.1,
        3.0,
    );
    draw_fps_slider(
        frame,
        size,
        scale,
        scale_rect(slider_rect(3), scale),
        state.fps_limit,
    );

    draw_text(
        frame,
        size,
        scale,
        24 * scale,
        448 * scale,
        "Runtime",
        0xffffff,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        468 * scale,
        &resources.resource_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        486 * scale,
        &resources.launch_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        504 * scale,
        &resources.library_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        522 * scale,
        &resources.parity_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    if size.height >= 560 * scale {
        draw_text_clipped(
            frame,
            size,
            scale,
            24 * scale,
            540 * scale,
            &format!("palette: {}", state.palette_source),
            0x9adfb1,
            (PANEL_W - 42) * scale,
        );
        draw_text_clipped(
            frame,
            size,
            scale,
            24 * scale,
            558 * scale,
            &state.status,
            0xffcc8a,
            (PANEL_W - 42) * scale,
        );
    } else {
        draw_text_clipped(
            frame,
            size,
            scale,
            24 * scale,
            528 * scale,
            &state.status,
            0xffcc8a,
            (PANEL_W - 42) * scale,
        );
    }
}

fn mode_detail(mode: usize) -> &'static str {
    match mode {
        1 => "dynamic LDS tile reduction",
        2 => "device-scope atomic histogram",
        3 => "kernel chain: base -> post FX",
        _ => "typed Rust GPU kernel launch",
    }
}

fn draw_button(
    frame: &mut [u32],
    size: RenderSize,
    scale: usize,
    rect: Rect,
    label: &str,
    color: u32,
    active: bool,
) {
    blend_rect(
        frame,
        size,
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        color,
        if active { 242 } else { 205 },
    );
    draw_rect_outline(
        frame,
        size,
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        if active { 0xd7f6ff } else { 0x557083 },
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        rect.x + 8 * scale,
        rect.y + 9 * scale,
        label,
        0xf6fbff,
        rect.w - 14 * scale,
    );
}

fn draw_slider(
    frame: &mut [u32],
    size: RenderSize,
    scale: usize,
    label: &str,
    rect: Rect,
    value: f32,
    min: f32,
    max: f32,
) {
    draw_text(
        frame,
        size,
        scale,
        rect.x,
        rect.y.saturating_sub(18 * scale),
        &format!("{label} {:.2}", value),
        0xdce8f4,
    );
    blend_rect(frame, size, rect.x, rect.y, rect.w, rect.h, 0x142434, 210);
    let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
    let fill = ((rect.w - 4) as f32 * t) as usize;
    draw_rect(
        frame,
        size,
        rect.x + 2,
        rect.y + 2,
        fill,
        rect.h - 4,
        0x2b8ee8,
    );
    draw_rect_outline(frame, size, rect.x, rect.y, rect.w, rect.h, 0x66889e);
}

fn draw_fps_slider(
    frame: &mut [u32],
    size: RenderSize,
    scale: usize,
    rect: Rect,
    fps_limit: usize,
) {
    draw_text(
        frame,
        size,
        scale,
        rect.x,
        rect.y.saturating_sub(18 * scale),
        &format!("FPS Limit {}", fps_label(fps_limit)),
        0xdce8f4,
    );
    blend_rect(frame, size, rect.x, rect.y, rect.w, rect.h, 0x142434, 210);
    let index = fps_limit_index(fps_limit);
    let t = index as f32 / (FPS_LIMITS.len() - 1) as f32;
    let fill = ((rect.w - 4) as f32 * t) as usize;
    draw_rect(
        frame,
        size,
        rect.x + 2,
        rect.y + 2,
        fill,
        rect.h - 4,
        0x2b8ee8,
    );
    draw_rect_outline(frame, size, rect.x, rect.y, rect.w, rect.h, 0x66889e);
}

fn mode_rect(index: usize) -> Rect {
    Rect::new(18 + index * 72, 106, 66, 32)
}

fn button_rect(index: usize) -> Rect {
    let col = index % 2;
    let row = index / 2;
    Rect::new(24 + col * 136, 176 + row * 38, 124, 30)
}

fn slider_rect(index: usize) -> Rect {
    Rect::new(24, 306 + index * 36, PANEL_W - 48, 16)
}

fn ui_scale(size: RenderSize) -> usize {
    if size.height >= 1800 || size.width >= 3200 {
        3
    } else if size.height >= 1200 || size.width >= 2200 {
        2
    } else {
        1
    }
}

fn scale_rect(rect: Rect, scale: usize) -> Rect {
    Rect::new(
        rect.x * scale,
        rect.y * scale,
        rect.w * scale,
        rect.h * scale,
    )
}

fn slider_value(x: usize, rect: Rect, min: f32, max: f32) -> f32 {
    let t = ((x.saturating_sub(rect.x)) as f32 / rect.w.max(1) as f32).clamp(0.0, 1.0);
    min + (max - min) * t
}

fn save_png(
    path: &Path,
    frame: &[u32],
    size: RenderSize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut image = RgbImage::new(size.width as u32, size.height as u32);
    for (index, pixel) in frame.iter().copied().enumerate() {
        let x = (index % size.width) as u32;
        let y = (index / size.width) as u32;
        image.put_pixel(
            x,
            y,
            Rgb([
                ((pixel >> 16) & 255) as u8,
                ((pixel >> 8) & 255) as u8,
                (pixel & 255) as u8,
            ]),
        );
    }
    image.save(path)?;
    Ok(())
}

fn parse_args() -> Result<DemoArgs, Box<dyn std::error::Error>> {
    let mut frames = None;
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut mode = None;
    let mut size = RESOLUTION_PRESETS[0].size;
    let mut fps_limit = 60usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--frames requires a frame count".to_string())?;
                frames = Some(value.parse::<u32>()?);
            }
            "--output" => {
                output = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--output requires a path".to_string())?;
            }
            "--mode" => {
                let value = args.next().ok_or_else(|| {
                    "--mode requires Core, LDS, Atomic, Chain, or 1-4".to_string()
                })?;
                mode = Some(parse_mode(&value)?);
            }
            "--resolution" | "--res" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--resolution requires a preset or WIDTHxHEIGHT".to_string())?;
                size = parse_resolution(&value)?;
            }
            "--fps" | "--fps-limit" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--fps-limit requires a number or uncapped".to_string())?;
                fps_limit = parse_fps_limit(&value)?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --example spectral_lattice -- [--frames N] [--mode MODE] [--resolution 4k|WIDTHxHEIGHT] [--fps-limit FPS|uncapped] [--output PATH]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument `{arg}`").into()),
        }
    }
    Ok(DemoArgs {
        frames,
        output,
        mode,
        size,
        fps_limit,
    })
}

fn parse_mode(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    if let Ok(index) = value.parse::<usize>() {
        return if (1..=MODES.len()).contains(&index) {
            Ok(index - 1)
        } else if index < MODES.len() {
            Ok(index)
        } else {
            Err(format!(
                "mode index {index} is outside 0-{} or 1-{}",
                MODES.len() - 1,
                MODES.len()
            )
            .into())
        };
    }

    MODES
        .iter()
        .position(|mode| mode.eq_ignore_ascii_case(value))
        .ok_or_else(|| {
            format!("unknown mode `{value}`; expected Core, LDS, Atomic, or Chain").into()
        })
}

fn parse_resolution(value: &str) -> Result<RenderSize, Box<dyn std::error::Error>> {
    if let Some(preset) = RESOLUTION_PRESETS
        .iter()
        .find(|preset| preset.label.eq_ignore_ascii_case(value))
    {
        return Ok(preset.size);
    }

    match value.to_ascii_lowercase().as_str() {
        "540" | "540p" => return Ok(RESOLUTION_PRESETS[0].size),
        "720" | "720p" => return Ok(RESOLUTION_PRESETS[1].size),
        "1080" | "1080p" | "fhd" => return Ok(RESOLUTION_PRESETS[2].size),
        "1440" | "1440p" | "qhd" => return Ok(RESOLUTION_PRESETS[3].size),
        "2160" | "2160p" | "uhd" | "4k" => return Ok(RESOLUTION_PRESETS[4].size),
        _ => {}
    }

    let Some((width, height)) = value.split_once(['x', 'X']) else {
        return Err(format!(
            "unknown resolution `{value}`; expected 540p, 720p, 1080p, 1440p, 4k, or WIDTHxHEIGHT"
        )
        .into());
    };
    let width = width.parse::<usize>()?;
    let height = height.parse::<usize>()?;
    if width < 640 || height < 360 || width > 7680 || height > 4320 {
        return Err(format!(
            "resolution {width}x{height} is outside supported bounds 640x360..7680x4320"
        )
        .into());
    }
    width
        .checked_mul(height)
        .ok_or_else(|| format!("resolution {width}x{height} overflows pixel count"))?;
    Ok(RenderSize::new(width, height))
}

fn parse_fps_limit(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    if value.eq_ignore_ascii_case("uncapped") || value.eq_ignore_ascii_case("off") {
        return Ok(0);
    }
    let fps = value.parse::<usize>()?;
    if fps != 0 && !(15..=360).contains(&fps) {
        return Err(
            format!("FPS limit {fps} is outside supported bounds 15..360, or 0/uncapped").into(),
        );
    }
    Ok(fps)
}

fn blend_rect(
    frame: &mut [u32],
    size: RenderSize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: u32,
    alpha: u32,
) {
    let x_end = (x + w).min(size.width);
    let y_end = (y + h).min(size.height);
    for py in y.min(size.height)..y_end {
        let row = py * size.width;
        for px in x.min(size.width)..x_end {
            let index = row + px;
            frame[index] = blend(frame[index], color, alpha);
        }
    }
}

fn draw_rect(
    frame: &mut [u32],
    size: RenderSize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: u32,
) {
    let x_end = (x + w).min(size.width);
    let y_end = (y + h).min(size.height);
    for py in y.min(size.height)..y_end {
        let row = py * size.width;
        for px in x.min(size.width)..x_end {
            frame[row + px] = color;
        }
    }
}

fn draw_rect_outline(
    frame: &mut [u32],
    size: RenderSize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: u32,
) {
    draw_rect(frame, size, x, y, w, 1, color);
    draw_rect(frame, size, x, y + h.saturating_sub(1), w, 1, color);
    draw_rect(frame, size, x, y, 1, h, color);
    draw_rect(frame, size, x + w.saturating_sub(1), y, 1, h, color);
}

fn draw_text(
    frame: &mut [u32],
    size: RenderSize,
    scale: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
) {
    draw_text_clipped(frame, size, scale, x, y, text, color, size.width - x);
}

fn draw_text_clipped(
    frame: &mut [u32],
    size: RenderSize,
    scale: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    max_width: usize,
) {
    let mut cx = x;
    let max_x = x.saturating_add(max_width).min(size.width);
    for ch in text.chars() {
        if cx + 8 > max_x {
            break;
        }
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits >> col) & 1 == 1 {
                        let base_x = cx + col * scale;
                        let base_y = y + row * scale;
                        for oy in 0..scale {
                            for ox in 0..scale {
                                let px = base_x + ox;
                                let py = base_y + oy;
                                if px < size.width && py < size.height {
                                    frame[py * size.width + px] = color;
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += 8 * scale;
    }
}

fn blend(dst: u32, src: u32, alpha: u32) -> u32 {
    let inv = 255u32.saturating_sub(alpha);
    let dr = (dst >> 16) & 255;
    let dg = (dst >> 8) & 255;
    let db = dst & 255;
    let sr = (src >> 16) & 255;
    let sg = (src >> 8) & 255;
    let sb = src & 255;
    (((dr * inv + sr * alpha) / 255) << 16)
        | (((dg * inv + sg * alpha) / 255) << 8)
        | ((db * inv + sb * alpha) / 255)
}

fn opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn phase(value: f32) -> f32 {
    value.abs().fract() * std::f32::consts::TAU
}

fn clamp_f32(value: f32, lo: f32, hi: f32) -> f32 {
    value.max(lo).min(hi)
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}
