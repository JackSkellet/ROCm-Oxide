use ash::vk::Handle;
use ash::{Entry, vk};
use font8x8::{BASIC_FONTS, UnicodeFonts};
use gl::types::{GLchar, GLenum, GLint, GLsizei, GLuint};
use image::{Rgb, RgbImage};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rocm_oxide::{
    Device, DeviceBuffer, Event, LaunchConfig, PinnedHostBuffer, RocBlas, RocmLibraryReport,
    SgemmLayout, Stream, rocm_feature_parity_for_device,
};
use sdl2::event::Event as SdlEvent;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton as SdlMouseButton;
use std::ffi::{CStr, CString, c_int, c_uint, c_void};
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const PANEL_W: usize = 318;
const BLOCK_X: u32 = 256;
const DEFAULT_OUTPUT: &str = "target/spectral_lattice.png";
const MODES: [&str; 4] = ["Core", "LDS", "Atomic", "Chain"];
const FPS_LIMITS: [usize; 7] = [30, 60, 90, 120, 144, 240, 0];
const PRESENT_SCALES: [usize; 3] = [1, 2, 4];
const GPU_WORK_PRESETS: [usize; 11] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const DEFAULT_GPU_WORK: usize = 64;
const HIP_GL_DEVICE_LIST_ALL: c_uint = 1;
const HIP_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: c_uint = 2;
const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_int = 1;
const RESOLUTION_PRESETS: [ResolutionPreset; 5] = [
    ResolutionPreset::new("540p", 960, 540),
    ResolutionPreset::new("720p", 1280, 720),
    ResolutionPreset::new("1080p", 1920, 1080),
    ResolutionPreset::new("1440p", 2560, 1440),
    ResolutionPreset::new("4K", 3840, 2160),
];

type HipGraphicsResource = *mut c_void;
type HipExternalMemory = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct HipExternalWin32Handle {
    handle: *mut c_void,
    name: *const c_void,
}

#[repr(C)]
union HipExternalMemoryHandle {
    fd: c_int,
    win32: HipExternalWin32Handle,
    nv_sci_buf_object: *const c_void,
}

#[repr(C)]
struct HipExternalMemoryHandleDesc {
    handle_type: c_int,
    handle: HipExternalMemoryHandle,
    size: u64,
    flags: c_uint,
    reserved: [c_uint; 16],
}

#[repr(C)]
struct HipExternalMemoryBufferDesc {
    offset: u64,
    size: u64,
    flags: c_uint,
    reserved: [c_uint; 16],
}

unsafe extern "C" {
    fn hipGLGetDevices(
        device_count_out: *mut c_uint,
        devices: *mut i32,
        device_count: c_uint,
        device_list: c_uint,
    ) -> rocm_oxide::hip::HipError;
    fn hipGraphicsGLRegisterBuffer(
        resource: *mut HipGraphicsResource,
        buffer: c_uint,
        flags: c_uint,
    ) -> rocm_oxide::hip::HipError;
    fn hipGraphicsMapResources(
        count: i32,
        resources: *mut HipGraphicsResource,
        stream: rocm_oxide::hip::HipStream,
    ) -> rocm_oxide::hip::HipError;
    fn hipGraphicsResourceGetMappedPointer(
        dev_ptr: *mut *mut c_void,
        size: *mut usize,
        resource: HipGraphicsResource,
    ) -> rocm_oxide::hip::HipError;
    fn hipGraphicsUnmapResources(
        count: i32,
        resources: *mut HipGraphicsResource,
        stream: rocm_oxide::hip::HipStream,
    ) -> rocm_oxide::hip::HipError;
    fn hipGraphicsUnregisterResource(resource: HipGraphicsResource) -> rocm_oxide::hip::HipError;
    fn hipImportExternalMemory(
        ext_mem_out: *mut HipExternalMemory,
        mem_handle_desc: *const HipExternalMemoryHandleDesc,
    ) -> rocm_oxide::hip::HipError;
    fn hipExternalMemoryGetMappedBuffer(
        dev_ptr: *mut *mut c_void,
        ext_mem: HipExternalMemory,
        buffer_desc: *const HipExternalMemoryBufferDesc,
    ) -> rocm_oxide::hip::HipError;
    fn hipDestroyExternalMemory(ext_mem: HipExternalMemory) -> rocm_oxide::hip::HipError;
}

struct DemoArgs {
    frames: Option<u32>,
    output: PathBuf,
    mode: Option<usize>,
    size: RenderSize,
    fps_limit: usize,
    gpu_work: usize,
    present_scale: usize,
    present_backend: PresentBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresentBackend {
    Cpu,
    Gl,
    Vulkan,
}

struct PaletteSeed {
    values: [f32; 4],
    source: String,
}

#[derive(Clone)]
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
    gpu_ms: f32,
    copy_ms: f64,
    draw_ms: f64,
    present_ms: f64,
    frame_ms: f64,
    fps_limit: usize,
    gpu_work: usize,
    present_scale: usize,
    render_size: RenderSize,
    save_requested: bool,
}

#[derive(Clone)]
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
    tile_count: usize,
}

struct GlPresenter {
    window: sdl2::video::Window,
    _context: sdl2::video::GLContext,
    size: RenderSize,
    present_scale: usize,
    texture: GLuint,
    pbo: GLuint,
    overlay_texture: GLuint,
    vao: GLuint,
    program: GLuint,
    resource: HipGraphicsResource,
}

struct VulkanPresenter {
    window: sdl2::video::Window,
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    external_memory_fd_loader: ash::khr::external_memory_fd::Device,
    size: RenderSize,
    present_scale: usize,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_initialized: Vec<bool>,
    swapchain_extent: vk::Extent2D,
    swapchain_format: vk::Format,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    frame_image: vk::Image,
    frame_memory: vk::DeviceMemory,
    frame_image_initialized: bool,
    overlay_buffer: vk::Buffer,
    overlay_memory: vk::DeviceMemory,
    overlay_mapped_ptr: *mut u32,
    overlay_bytes: usize,
    shared_buffer: vk::Buffer,
    shared_memory: vk::DeviceMemory,
    hip_external_memory: HipExternalMemory,
    hip_mapped_ptr: *mut u32,
    shared_bytes: usize,
}

struct VulkanSharedMemory {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    hip_external_memory: HipExternalMemory,
    hip_mapped_ptr: *mut u32,
    bytes: usize,
}

struct VulkanOverlayMemory {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u32,
    bytes: usize,
}

struct MappedGlBuffer {
    ptr: *mut u32,
    len: usize,
}

struct OverlayFrame {
    size: RenderSize,
    pixels: Vec<u32>,
}

struct OverlayJob {
    generation: u64,
    frame: OverlayFrame,
    state: DemoState,
    resources: ResourceSnapshot,
}

struct OverlayResult {
    generation: u64,
    frame: OverlayFrame,
    draw_ms: f64,
}

struct AsyncOverlayRenderer {
    job_tx: SyncSender<OverlayJob>,
    result_rx: Receiver<OverlayResult>,
    current: OverlayFrame,
    spare: Option<OverlayFrame>,
    generation: u64,
    pending_generation: Option<u64>,
    last_draw_ms: f64,
}

struct QueuedGpuTiming {
    start: Event,
    stop: Event,
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

impl OverlayFrame {
    fn new(render_size: RenderSize) -> Self {
        let size = overlay_size_for(render_size);
        Self {
            size,
            pixels: vec![0; size.pixel_count()],
        }
    }

    fn resize(&mut self, render_size: RenderSize) {
        let size = overlay_size_for(render_size);
        if self.size != size {
            self.size = size;
            self.pixels.resize(size.pixel_count(), 0);
        }
    }

    fn draw(&mut self, state: &DemoState, resources: &ResourceSnapshot) {
        self.pixels.fill(overlay_panel_background_rgb());
        draw_overlay_contents(&mut self.pixels, self.size, state, resources);
        for pixel in &mut self.pixels {
            if *pixel != 0 {
                *pixel |= 0xff00_0000;
            }
        }
    }

    fn byte_len(&self) -> Result<usize, Box<dyn std::error::Error>> {
        self.pixels
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| other_error("overlay byte length overflows usize"))
    }
}

impl AsyncOverlayRenderer {
    fn new(
        render_size: RenderSize,
        state: &DemoState,
        resources: &ResourceSnapshot,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (job_tx, job_rx) = mpsc::sync_channel::<OverlayJob>(1);
        let (result_tx, result_rx) = mpsc::channel::<OverlayResult>();
        let _ = thread::Builder::new()
            .name("spectral-overlay".into())
            .spawn(move || {
                while let Ok(mut job) = job_rx.recv() {
                    let draw_start = Instant::now();
                    job.frame.resize(job.state.render_size);
                    job.frame.draw(&job.state, &job.resources);
                    let draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
                    if result_tx
                        .send(OverlayResult {
                            generation: job.generation,
                            frame: job.frame,
                            draw_ms,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;

        let mut current = OverlayFrame::new(render_size);
        let draw_start = Instant::now();
        current.draw(state, resources);
        let last_draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
        Ok(Self {
            job_tx,
            result_rx,
            current,
            spare: Some(OverlayFrame::new(render_size)),
            generation: 0,
            pending_generation: None,
            last_draw_ms,
        })
    }

    fn reset(&mut self, render_size: RenderSize, state: &DemoState, resources: &ResourceSnapshot) {
        self.generation = self.generation.wrapping_add(1);
        self.pending_generation = None;
        self.drain_stale_results();
        self.current.resize(render_size);
        if let Some(spare) = &mut self.spare {
            spare.resize(render_size);
        }
        let draw_start = Instant::now();
        self.current.draw(state, resources);
        self.last_draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
    }

    fn schedule(&mut self, state: &DemoState, resources: &ResourceSnapshot) {
        if self.pending_generation.is_some() {
            return;
        }
        let mut frame = self
            .spare
            .take()
            .unwrap_or_else(|| OverlayFrame::new(state.render_size));
        frame.resize(state.render_size);
        let generation = self.generation.wrapping_add(1);
        let job = OverlayJob {
            generation,
            frame,
            state: state.clone(),
            resources: resources.clone(),
        };
        match self.job_tx.try_send(job) {
            Ok(()) => {
                self.generation = generation;
                self.pending_generation = Some(generation);
            }
            Err(TrySendError::Full(job)) => {
                self.spare = Some(job.frame);
            }
            Err(TrySendError::Disconnected(job)) => {
                self.spare = Some(job.frame);
            }
        }
    }

    fn poll_ready(&mut self) {
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => self.accept_result(result),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn current(&self) -> &OverlayFrame {
        &self.current
    }

    fn last_draw_ms(&self) -> f64 {
        self.last_draw_ms
    }

    fn accept_result(&mut self, result: OverlayResult) {
        let is_pending = self.pending_generation == Some(result.generation);
        let is_current_generation = result.generation == self.generation;
        if is_pending || is_current_generation {
            let old = std::mem::replace(&mut self.current, result.frame);
            self.spare = Some(old);
            self.last_draw_ms = result.draw_ms;
            if is_pending {
                self.pending_generation = None;
            }
        } else if self.spare.is_none() {
            self.spare = Some(result.frame);
        }
    }

    fn drain_stale_results(&mut self) {
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    if self.spare.is_none() {
                        self.spare = Some(result.frame);
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }
}

impl QueuedGpuTiming {
    fn elapsed_ms(&self) -> Result<f32, Box<dyn std::error::Error>> {
        Ok(self.start.elapsed_ms_until(&self.stop)?)
    }
}

fn overlay_size_for(render_size: RenderSize) -> RenderSize {
    RenderSize::new(
        (PANEL_W * ui_scale(render_size)).min(render_size.width),
        render_size.height,
    )
}

fn prefer_x11_for_rocm_gl_interop(present_backend: PresentBackend) {
    if present_backend != PresentBackend::Gl || std::env::var_os("SDL_VIDEODRIVER").is_some() {
        return;
    }

    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("DISPLAY").is_some() {
            // ROCm 7.2's Mesa GL interop path recognizes this machine's GLX
            // context, while the Wayland/EGL path can abort inside rocclr.
            let _ = sdl2::hint::set_with_priority(
                "SDL_VIDEODRIVER",
                "x11",
                &sdl2::hint::Hint::Override,
            );
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    prefer_x11_for_rocm_gl_interop(args.present_backend);
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let mut buffers = DemoBuffers::new(args.size)?;
    let mut host_frame = PinnedHostBuffer::<u32>::new_zeroed(buffers.size.pixel_count())?;
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
        gpu_ms: 0.0,
        copy_ms: 0.0,
        draw_ms: 0.0,
        present_ms: 0.0,
        frame_ms: 0.0,
        fps_limit: args.fps_limit,
        gpu_work: args.gpu_work,
        present_scale: args.present_scale,
        render_size: args.size,
        save_requested: false,
    };
    if let Some(mode) = args.mode {
        set_mode(&mut state, mode);
    }

    if args.present_backend == PresentBackend::Gl {
        return run_gl_present(&args, &device, &kernels, buffers, state, resources);
    }
    if args.present_backend == PresentBackend::Vulkan {
        return run_vulkan_present(&args, &device, &kernels, buffers, state, resources);
    }

    if let Some(frames) = args.frames {
        let frames = frames.max(1);
        let mut use_post = false;
        for frame in 0..frames {
            state.frame_index = ((frame as f32) * state.speed * 2.0) as u32;
            if state.auto_cycle {
                state.mode = ((state.frame_index / 180) as usize) % MODES.len();
            }
            let (post, gpu_ms) = render_workload_timed(&kernels, &buffers, &state)?;
            use_post = post;
            state.gpu_ms = gpu_ms;
        }
        let copy_start = Instant::now();
        copy_display_frame(&buffers, use_post, &mut host_frame)?;
        state.copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
        let draw_start = Instant::now();
        draw_overlay(host_frame.as_mut_slice(), buffers.size, &state, &resources);
        state.draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
        save_png(&args.output, host_frame.as_slice(), buffers.size)?;
        println!(
            "saved Spectral Lattice GUI preview after {frames} frame(s): {}",
            args.output.display()
        );
        return Ok(());
    }

    let mut window = create_window(buffers.size, state.fps_limit, state.present_scale)?;

    let start = Instant::now();
    let mut last_fps = Instant::now();
    let mut frames_since_fps = 0u32;
    let mut mouse_was_down = false;
    let mut applied_fps_limit = state.fps_limit;
    let mut applied_present_scale = state.present_scale;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = Instant::now();
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
            host_frame = PinnedHostBuffer::<u32>::new_zeroed(buffers.size.pixel_count())?;
            resources = ResourceSnapshot::new(&device, &kernels, buffers.size.pixel_count())?;
            window = create_window(buffers.size, state.fps_limit, state.present_scale)?;
            applied_fps_limit = state.fps_limit;
            applied_present_scale = state.present_scale;
            state.status = format!("resolution set to {}", display_label(&state));
        } else if state.present_scale != applied_present_scale {
            window = create_window(buffers.size, state.fps_limit, state.present_scale)?;
            applied_fps_limit = state.fps_limit;
            applied_present_scale = state.present_scale;
            state.status = format!("present scale set to {}x", state.present_scale);
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

        let (use_post, gpu_ms) = render_workload_timed(&kernels, &buffers, &state)?;
        state.gpu_ms = gpu_ms;
        let copy_start = Instant::now();
        copy_display_frame(&buffers, use_post, &mut host_frame)?;
        state.copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
        let draw_start = Instant::now();
        draw_overlay(host_frame.as_mut_slice(), buffers.size, &state, &resources);
        state.draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;

        if state.save_requested {
            save_png(&args.output, host_frame.as_slice(), buffers.size)?;
            state.status = format!("saved {}", args.output.display());
            state.save_requested = false;
        }

        let present_start = Instant::now();
        window.update_with_buffer(
            host_frame.as_slice(),
            buffers.size.width,
            buffers.size.height,
        )?;
        state.present_ms = present_start.elapsed().as_secs_f64() * 1000.0;
        state.frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        frames_since_fps = frames_since_fps.saturating_add(1);
        if last_fps.elapsed() >= Duration::from_millis(500) {
            state.fps = frames_since_fps as f64 / last_fps.elapsed().as_secs_f64();
            frames_since_fps = 0;
            last_fps = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide Spectral Lattice | {} | {:.1} FPS | {:.1} ms frame | limit {} | {}",
                display_label(&state),
                state.fps,
                state.frame_ms,
                fps_label(state.fps_limit),
                MODES[state.mode],
            ));
        }
    }

    Ok(())
}

fn create_window(
    size: RenderSize,
    fps_limit: usize,
    present_scale: usize,
) -> Result<Window, Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "ROCm-Oxide Spectral Lattice",
        size.width,
        size.height,
        WindowOptions {
            resize: true,
            scale: minifb_scale(present_scale),
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(fps_limit);
    Ok(window)
}

fn run_gl_present(
    args: &DemoArgs,
    device: &Device,
    kernels: &generated::DeviceKernels,
    mut buffers: DemoBuffers,
    mut state: DemoState,
    mut resources: ResourceSnapshot,
) -> Result<(), Box<dyn std::error::Error>> {
    let sdl = sdl2::init().map_err(other_error)?;
    let mut presenter = GlPresenter::new(&sdl, buffers.size, state.present_scale, state.fps_limit)?;
    let mut events = sdl.event_pump().map_err(other_error)?;
    let mut overlay = OverlayFrame::new(buffers.size);
    let start = Instant::now();
    let run_start = Instant::now();
    let mut last_fps = Instant::now();
    let mut frames_since_fps = 0u32;
    let mut presented_frames = 0u32;
    let mut last_use_post = false;
    let mut frame_budget = args.frames.map(|frames| frames.max(1));

    while frame_budget != Some(0) {
        let frame_start = Instant::now();
        for event in events.poll_iter() {
            if !handle_sdl_event(
                event,
                &presenter.window,
                &mut state,
                kernels,
                &buffers.short,
            ) {
                frame_budget = Some(0);
                break;
            }
        }
        if frame_budget == Some(0) {
            break;
        }

        if state.render_size != buffers.size {
            buffers = DemoBuffers::new(state.render_size)?;
            resources = ResourceSnapshot::new(device, kernels, buffers.size.pixel_count())?;
            overlay.resize(buffers.size);
            presenter.recreate_frame_resources(buffers.size, state.present_scale)?;
            state.status = format!("resolution set to {}", display_label(&state));
        } else if state.present_scale != presenter.present_scale {
            presenter.recreate_frame_resources(buffers.size, state.present_scale)?;
            state.status = format!("present scale set to {}x", state.present_scale);
        }

        if !state.paused {
            state.frame_index = (start.elapsed().as_secs_f32() * 60.0 * state.speed) as u32;
            if state.auto_cycle {
                state.mode = ((state.frame_index / 180) as usize) % MODES.len();
            }
        }

        let (use_post, gpu_ms) = render_workload_timed(kernels, &buffers, &state)?;
        last_use_post = use_post;
        state.gpu_ms = gpu_ms;
        let source = display_source(&buffers, use_post);
        let draw_start = Instant::now();
        overlay.draw(&state, &resources);
        state.draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
        let (interop_ms, present_ms) = presenter.present_device_frame(source, &overlay)?;
        state.copy_ms = interop_ms;
        state.present_ms = present_ms;

        if state.save_requested {
            let mut host_frame = PinnedHostBuffer::<u32>::new_zeroed(buffers.size.pixel_count())?;
            let copy_start = Instant::now();
            copy_display_frame(&buffers, use_post, &mut host_frame)?;
            state.copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
            draw_overlay(host_frame.as_mut_slice(), buffers.size, &state, &resources);
            save_png(&args.output, host_frame.as_slice(), buffers.size)?;
            state.status = format!("saved {}", args.output.display());
            state.save_requested = false;
        }

        state.frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        presented_frames = presented_frames.saturating_add(1);
        frames_since_fps = frames_since_fps.saturating_add(1);
        if last_fps.elapsed() >= Duration::from_millis(500) {
            state.fps = frames_since_fps as f64 / last_fps.elapsed().as_secs_f64();
            frames_since_fps = 0;
            last_fps = Instant::now();
            presenter.window.set_title(&format!(
                "ROCm-Oxide Spectral Lattice GL | {} | {:.1} FPS | gpu {:.2} copy {:.2} present {:.2} | limit {} | {}",
                display_label(&state),
                state.fps,
                state.gpu_ms,
                state.copy_ms,
                state.present_ms,
                fps_label(state.fps_limit),
                MODES[state.mode],
            ))?;
        }

        if let Some(frames) = frame_budget.as_mut() {
            *frames = frames.saturating_sub(1);
        }
        pace_frame(frame_start, state.fps_limit);
    }

    if let Some(frames) = args.frames {
        let elapsed = run_start.elapsed().as_secs_f64().max(f64::EPSILON);
        let mut host_frame = PinnedHostBuffer::<u32>::new_zeroed(buffers.size.pixel_count())?;
        copy_display_frame(&buffers, last_use_post, &mut host_frame)?;
        draw_overlay(host_frame.as_mut_slice(), buffers.size, &state, &resources);
        save_png(&args.output, host_frame.as_slice(), buffers.size)?;
        println!(
            "saved Spectral Lattice GL preview after {} frame(s): {}",
            frames.max(1),
            args.output.display()
        );
        println!(
            "GL-present summary: {:.1} FPS over {} rendered frame(s), last gpu {:.2} ms, interop {:.2} ms, present {:.2} ms, frame {:.2} ms",
            presented_frames as f64 / elapsed,
            presented_frames,
            state.gpu_ms,
            state.copy_ms,
            state.present_ms,
            state.frame_ms,
        );
    }

    Ok(())
}

fn run_vulkan_present(
    args: &DemoArgs,
    device: &Device,
    kernels: &generated::DeviceKernels,
    mut buffers: DemoBuffers,
    mut state: DemoState,
    mut resources: ResourceSnapshot,
) -> Result<(), Box<dyn std::error::Error>> {
    let sdl = sdl2::init().map_err(other_error)?;
    let mut presenter =
        VulkanPresenter::new(&sdl, buffers.size, state.present_scale, state.fps_limit)?;
    let mut events = sdl.event_pump().map_err(other_error)?;
    let mut overlay_renderer = AsyncOverlayRenderer::new(buffers.size, &state, &resources)?;
    let start = Instant::now();
    let run_start = Instant::now();
    let mut last_fps = Instant::now();
    let mut frames_since_fps = 0u32;
    let mut presented_frames = 0u32;
    let mut last_use_post = false;
    let mut frame_budget = args.frames.map(|frames| frames.max(1));

    while frame_budget != Some(0) {
        let frame_start = Instant::now();
        for event in events.poll_iter() {
            if !handle_sdl_event(
                event,
                &presenter.window,
                &mut state,
                kernels,
                &buffers.short,
            ) {
                frame_budget = Some(0);
                break;
            }
        }
        if frame_budget == Some(0) {
            break;
        }

        if state.render_size != buffers.size {
            buffers = DemoBuffers::new(state.render_size)?;
            resources = ResourceSnapshot::new(device, kernels, buffers.size.pixel_count())?;
            presenter.recreate_frame_resources(buffers.size, state.present_scale)?;
            state.status = format!("resolution set to {}", display_label(&state));
            overlay_renderer.reset(buffers.size, &state, &resources);
        } else if state.present_scale != presenter.present_scale {
            presenter.recreate_frame_resources(buffers.size, state.present_scale)?;
            state.status = format!("present scale set to {}x", state.present_scale);
        }

        if !state.paused {
            state.frame_index = (start.elapsed().as_secs_f32() * 60.0 * state.speed) as u32;
            if state.auto_cycle {
                state.mode = ((state.frame_index / 180) as usize) % MODES.len();
            }
        }

        overlay_renderer.poll_ready();
        state.draw_ms = overlay_renderer.last_draw_ms();
        overlay_renderer.schedule(&state, &resources);

        let (use_post, gpu_timing) = render_workload_queued(kernels, &buffers, &state)?;
        last_use_post = use_post;
        let source = display_source(&buffers, use_post);
        overlay_renderer.poll_ready();
        state.draw_ms = overlay_renderer.last_draw_ms();
        let (interop_ms, present_ms) =
            presenter.present_device_frame(source, overlay_renderer.current())?;
        state.copy_ms = interop_ms;
        state.present_ms = present_ms;
        state.gpu_ms = gpu_timing.elapsed_ms()?;

        if state.save_requested {
            let mut host_frame = PinnedHostBuffer::<u32>::new_zeroed(buffers.size.pixel_count())?;
            let copy_start = Instant::now();
            copy_display_frame(&buffers, use_post, &mut host_frame)?;
            state.copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
            draw_overlay(host_frame.as_mut_slice(), buffers.size, &state, &resources);
            save_png(&args.output, host_frame.as_slice(), buffers.size)?;
            state.status = format!("saved {}", args.output.display());
            state.save_requested = false;
        }

        state.frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        presented_frames = presented_frames.saturating_add(1);
        frames_since_fps = frames_since_fps.saturating_add(1);
        if last_fps.elapsed() >= Duration::from_millis(500) {
            state.fps = frames_since_fps as f64 / last_fps.elapsed().as_secs_f64();
            frames_since_fps = 0;
            last_fps = Instant::now();
            presenter.window.set_title(&format!(
                "ROCm-Oxide Spectral Lattice Vulkan | {} | {:.1} FPS | gpu {:.2} d2d {:.2} present {:.2} | limit {} | {}",
                display_label(&state),
                state.fps,
                state.gpu_ms,
                state.copy_ms,
                state.present_ms,
                fps_label(state.fps_limit),
                MODES[state.mode],
            ))?;
        }

        if let Some(frames) = frame_budget.as_mut() {
            *frames = frames.saturating_sub(1);
        }
        pace_frame(frame_start, state.fps_limit);
    }

    if let Some(frames) = args.frames {
        let elapsed = run_start.elapsed().as_secs_f64().max(f64::EPSILON);
        let mut host_frame = PinnedHostBuffer::<u32>::new_zeroed(buffers.size.pixel_count())?;
        copy_display_frame(&buffers, last_use_post, &mut host_frame)?;
        draw_overlay(host_frame.as_mut_slice(), buffers.size, &state, &resources);
        save_png(&args.output, host_frame.as_slice(), buffers.size)?;
        println!(
            "saved Spectral Lattice Vulkan preview after {} frame(s): {}",
            frames.max(1),
            args.output.display()
        );
        println!(
            "Vulkan-present summary: {:.1} FPS over {} rendered frame(s), last gpu {:.2} ms, HIP->Vulkan d2d {:.2} ms, present {:.2} ms, frame {:.2} ms",
            presented_frames as f64 / elapsed,
            presented_frames,
            state.gpu_ms,
            state.copy_ms,
            state.present_ms,
            state.frame_ms,
        );
    }

    Ok(())
}

fn handle_sdl_event(
    event: SdlEvent,
    window: &sdl2::video::Window,
    state: &mut DemoState,
    kernels: &generated::DeviceKernels,
    short_frame: &DeviceBuffer<u32>,
) -> bool {
    match event {
        SdlEvent::Quit { .. } => false,
        SdlEvent::KeyDown {
            keycode: Some(key),
            repeat: false,
            ..
        } => handle_sdl_key(key, state, kernels, short_frame),
        SdlEvent::MouseButtonDown {
            mouse_btn: SdlMouseButton::Left,
            x,
            y,
            ..
        } => {
            if let Some((x, y)) = sdl_buffer_mouse_pos(window, state.render_size, x, y) {
                handle_slider_drag(state, x, y);
                handle_click(state, kernels, short_frame, x, y);
            }
            true
        }
        SdlEvent::MouseMotion {
            mousestate, x, y, ..
        } => {
            if mousestate.left()
                && let Some((x, y)) = sdl_buffer_mouse_pos(window, state.render_size, x, y)
            {
                handle_slider_drag(state, x, y);
            }
            true
        }
        _ => true,
    }
}

fn sdl_buffer_mouse_pos(
    window: &sdl2::video::Window,
    size: RenderSize,
    mx: i32,
    my: i32,
) -> Option<(usize, usize)> {
    let (win_w, win_h) = window.size();
    if win_w == 0 || win_h == 0 {
        return None;
    }

    let x = ((mx.max(0) as f64) * size.width as f64 / win_w as f64)
        .floor()
        .clamp(0.0, (size.width - 1) as f64) as usize;
    let y = ((my.max(0) as f64) * size.height as f64 / win_h as f64)
        .floor()
        .clamp(0.0, (size.height - 1) as f64) as usize;
    Some((x, y))
}

fn handle_sdl_key(
    key: Keycode,
    state: &mut DemoState,
    kernels: &generated::DeviceKernels,
    short_frame: &DeviceBuffer<u32>,
) -> bool {
    match key {
        Keycode::Escape | Keycode::Q => return false,
        Keycode::Num1 => set_mode(state, 0),
        Keycode::Num2 => set_mode(state, 1),
        Keycode::Num3 => set_mode(state, 2),
        Keycode::Num4 => set_mode(state, 3),
        Keycode::Left => set_mode(state, (state.mode + MODES.len() - 1) % MODES.len()),
        Keycode::Right => set_mode(state, (state.mode + 1) % MODES.len()),
        Keycode::Up => state.warp = clamp_f32(state.warp + 0.08, 0.05, 2.25),
        Keycode::Down => state.warp = clamp_f32(state.warp - 0.08, 0.05, 2.25),
        Keycode::PageUp => state.speed = clamp_f32(state.speed + 0.15, 0.1, 3.0),
        Keycode::PageDown => state.speed = clamp_f32(state.speed - 0.15, 0.1, 3.0),
        Keycode::Minus => step_fps_limit(state, -1),
        Keycode::Equals => step_fps_limit(state, 1),
        Keycode::Comma => cycle_resolution(state, -1),
        Keycode::Period => cycle_resolution(state, 1),
        Keycode::M => cycle_present_scale(state),
        Keycode::LeftBracket => step_gpu_work(state, -1),
        Keycode::RightBracket => step_gpu_work(state, 1),
        Keycode::Space => state.paused = !state.paused,
        Keycode::A => state.auto_cycle = !state.auto_cycle,
        Keycode::R => reseed_palette(state),
        Keycode::C => run_contract_check(state, kernels, short_frame),
        Keycode::S => state.save_requested = true,
        _ => {}
    }
    true
}

fn pace_frame(frame_start: Instant, fps_limit: usize) {
    if fps_limit == 0 {
        return;
    }
    let target = Duration::from_secs_f64(1.0 / fps_limit as f64);
    if let Some(remaining) = target.checked_sub(frame_start.elapsed()) {
        std::thread::sleep(remaining);
    }
}

fn copy_display_frame(
    buffers: &DemoBuffers,
    use_post: bool,
    host_frame: &mut PinnedHostBuffer<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if use_post {
        buffers.post.copy_to_pinned_host(host_frame)?;
    } else {
        buffers.base.copy_to_pinned_host(host_frame)?;
    }
    Ok(())
}

fn display_source(buffers: &DemoBuffers, use_post: bool) -> &DeviceBuffer<u32> {
    if use_post {
        &buffers.post
    } else {
        &buffers.base
    }
}

impl VulkanPresenter {
    fn new(
        sdl: &sdl2::Sdl,
        size: RenderSize,
        present_scale: usize,
        fps_limit: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let video = sdl.video().map_err(other_error)?;
        let window = video
            .window(
                "ROCm-Oxide Spectral Lattice Vulkan",
                checked_window_dim(size.width, present_scale)?,
                checked_window_dim(size.height, present_scale)?,
            )
            .vulkan()
            .resizable()
            .position_centered()
            .build()
            .map_err(|err| other_error(err.to_string()))?;

        let extension_names = window.vulkan_instance_extensions().map_err(other_error)?;
        let extension_cstrings = extension_names
            .iter()
            .map(|name| CString::new(*name))
            .collect::<Result<Vec<_>, _>>()?;
        let extension_ptrs = extension_cstrings
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let app_name = CString::new("rocm-oxide-spectral-lattice")?;
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(vk::make_api_version(0, 1, 1, 0));
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_ptrs);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let raw_surface = window
            .vulkan_create_surface(instance.handle().as_raw() as usize)
            .map_err(other_error)?;
        let surface = vk::SurfaceKHR::from_raw(raw_surface as u64);

        let (physical_device, queue_family_index) =
            pick_vulkan_device(&instance, &surface_loader, surface)?;
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let device_extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::external_memory::NAME.as_ptr(),
            ash::khr::external_memory_fd::NAME.as_ptr(),
        ];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let external_memory_fd_loader =
            ash::khr::external_memory_fd::Device::new(&instance, &device);
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        println!("Vulkan renderer: {device_name}");

        let mut presenter = Self {
            window,
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            memory_properties,
            device,
            queue_family_index,
            queue,
            swapchain_loader,
            external_memory_fd_loader,
            size,
            present_scale,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: Vec::new(),
            swapchain_image_initialized: Vec::new(),
            swapchain_extent: vk::Extent2D::default(),
            swapchain_format: vk::Format::UNDEFINED,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            image_available: vk::Semaphore::null(),
            render_finished: vk::Semaphore::null(),
            in_flight: vk::Fence::null(),
            frame_image: vk::Image::null(),
            frame_memory: vk::DeviceMemory::null(),
            frame_image_initialized: false,
            overlay_buffer: vk::Buffer::null(),
            overlay_memory: vk::DeviceMemory::null(),
            overlay_mapped_ptr: ptr::null_mut(),
            overlay_bytes: 0,
            shared_buffer: vk::Buffer::null(),
            shared_memory: vk::DeviceMemory::null(),
            hip_external_memory: ptr::null_mut(),
            hip_mapped_ptr: ptr::null_mut(),
            shared_bytes: 0,
        };
        presenter.recreate_frame_resources(size, present_scale)?;
        presenter.window.set_title(&format!(
            "ROCm-Oxide Spectral Lattice Vulkan | {} | limit {}",
            display_label_for(size, present_scale),
            fps_label(fps_limit)
        ))?;
        Ok(presenter)
    }

    fn recreate_frame_resources(
        &mut self,
        size: RenderSize,
        present_scale: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.destroy_frame_resources();
        self.window
            .set_size(
                checked_window_dim(size.width, present_scale)?,
                checked_window_dim(size.height, present_scale)?,
            )
            .map_err(|err| other_error(err.to_string()))?;
        let byte_len = frame_byte_len(size)?;

        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        if !surface_caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(other_error(
                "Vulkan surface does not support transfer-dst swapchain images",
            ));
        }
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?
        };
        let surface_format = choose_surface_format(&formats)?;
        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)?
        };
        let present_mode = choose_present_mode(&present_modes);
        let extent = choose_swapchain_extent(&self.window, surface_caps, size, present_scale)?;
        let image_count = swapchain_image_count(surface_caps);
        let overlay_size = overlay_size_for(size);

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(choose_composite_alpha(surface_caps))
            .present_mode(present_mode)
            .clipped(true);
        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_info, None)?
        };
        let swapchain_images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain)? };
        let shared = self.create_shared_memory(byte_len)?;
        let overlay = self.create_overlay_memory(frame_byte_len_usize(overlay_size)?)?;
        let (frame_image, frame_memory) = self.create_frame_image(size, surface_format.format)?;
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.queue_family_index);
        let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None)? };
        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { self.device.allocate_command_buffers(&command_info)?[0] };
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { self.device.create_semaphore(&semaphore_info, None)? };
        let render_finished = unsafe { self.device.create_semaphore(&semaphore_info, None)? };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { self.device.create_fence(&fence_info, None)? };

        self.size = size;
        self.present_scale = present_scale;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_initialized = vec![false; self.swapchain_images.len()];
        self.swapchain_extent = extent;
        self.swapchain_format = surface_format.format;
        self.command_pool = command_pool;
        self.command_buffer = command_buffer;
        self.image_available = image_available;
        self.render_finished = render_finished;
        self.in_flight = in_flight;
        self.frame_image = frame_image;
        self.frame_memory = frame_memory;
        self.frame_image_initialized = false;
        self.overlay_buffer = overlay.buffer;
        self.overlay_memory = overlay.memory;
        self.overlay_mapped_ptr = overlay.mapped_ptr;
        self.overlay_bytes = overlay.bytes;
        self.shared_buffer = shared.buffer;
        self.shared_memory = shared.memory;
        self.hip_external_memory = shared.hip_external_memory;
        self.hip_mapped_ptr = shared.hip_mapped_ptr;
        self.shared_bytes = shared.bytes;
        Ok(())
    }

    fn present_device_frame(
        &mut self,
        source: &DeviceBuffer<u32>,
        overlay: &OverlayFrame,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let interop_start = Instant::now();
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)?;
        }
        let required_bytes = frame_byte_len_usize(self.size)?;
        if self.shared_bytes < required_bytes {
            return Err(other_error(format!(
                "shared Vulkan/HIP buffer is too small: got {} bytes, need {} bytes",
                self.shared_bytes, required_bytes
            )));
        }
        unsafe {
            source.copy_to_device_ptr(self.hip_mapped_ptr, source.len())?;
        }
        Stream::null().synchronize()?;
        self.upload_overlay(overlay)?;
        let interop_ms = interop_start.elapsed().as_secs_f64() * 1000.0;

        let present_start = Instant::now();
        let (image_index, suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_frame_resources(self.size, self.present_scale)?;
                return Ok((interop_ms, 0.0));
            }
            Err(err) => return Err(other_error(format!("Vulkan acquire image failed: {err:?}"))),
        };

        unsafe {
            self.device.reset_fences(&[self.in_flight])?;
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        self.record_present_commands(image_index as usize, overlay.size)?;

        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let wait_semaphores = [self.image_available];
        let signal_semaphores = [self.render_finished];
        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), self.in_flight)?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe { self.swapchain_loader.queue_present(self.queue, &present) };
        match present_result {
            Ok(present_suboptimal) => {
                if suboptimal || present_suboptimal {
                    self.recreate_frame_resources(self.size, self.present_scale)?;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_frame_resources(self.size, self.present_scale)?;
            }
            Err(err) => return Err(other_error(format!("Vulkan present failed: {err:?}"))),
        }
        let present_ms = present_start.elapsed().as_secs_f64() * 1000.0;
        Ok((interop_ms, present_ms))
    }

    fn record_present_commands(
        &mut self,
        image_index: usize,
        overlay_size: RenderSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let swapchain_image = self.swapchain_images[image_index];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }

        let frame_old_layout = if self.frame_image_initialized {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        self.image_barrier(
            self.frame_image,
            frame_old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            if self.frame_image_initialized {
                vk::AccessFlags::TRANSFER_READ
            } else {
                vk::AccessFlags::empty()
            },
            vk::AccessFlags::TRANSFER_WRITE,
            if self.frame_image_initialized {
                vk::PipelineStageFlags::TRANSFER
            } else {
                vk::PipelineStageFlags::TOP_OF_PIPE
            },
            vk::PipelineStageFlags::TRANSFER,
        );

        let copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(color_subresource_layers())
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: self.size.width as u32,
                height: self.size.height as u32,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.shared_buffer,
                self.frame_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
        }

        self.image_barrier(
            self.frame_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );

        let swapchain_old_layout = if self.swapchain_image_initialized[image_index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        self.image_barrier(
            swapchain_image,
            swapchain_old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        let blit = vk::ImageBlit::default()
            .src_subresource(color_subresource_layers())
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.size.width as i32,
                    y: self.size.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(color_subresource_layers())
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.swapchain_extent.width as i32,
                    y: self.swapchain_extent.height as i32,
                    z: 1,
                },
            ]);
        unsafe {
            self.device.cmd_blit_image(
                self.command_buffer,
                self.frame_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                vk::Filter::LINEAR,
            );
        }

        let overlay_copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(overlay_size.width as u32)
            .buffer_image_height(overlay_size.height as u32)
            .image_subresource(color_subresource_layers())
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: overlay_size.width as u32,
                height: overlay_size.height as u32,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.overlay_buffer,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&overlay_copy),
            );
        }

        self.image_barrier(
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );

        unsafe {
            self.device.end_command_buffer(self.command_buffer)?;
        }
        self.frame_image_initialized = true;
        self.swapchain_image_initialized[image_index] = true;
        Ok(())
    }

    fn upload_overlay(&mut self, overlay: &OverlayFrame) -> Result<(), Box<dyn std::error::Error>> {
        let required_bytes = overlay.byte_len()?;
        if self.overlay_bytes < required_bytes {
            return Err(other_error(format!(
                "Vulkan overlay buffer is too small: got {} bytes, need {} bytes",
                self.overlay_bytes, required_bytes
            )));
        }
        if self.overlay_mapped_ptr.is_null() {
            return Err(other_error("Vulkan overlay buffer is not mapped"));
        }
        unsafe {
            ptr::copy_nonoverlapping(
                overlay.pixels.as_ptr(),
                self.overlay_mapped_ptr,
                overlay.pixels.len(),
            );
        }
        Ok(())
    }

    fn image_barrier(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_subresource_range());
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }

    fn create_shared_memory(
        &self,
        bytes: usize,
    ) -> Result<VulkanSharedMemory, Box<dyn std::error::Error>> {
        let mut external_buffer = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(bytes as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_buffer);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_info);
        let memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe {
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(other_error(format!(
                    "Vulkan exportable buffer allocation failed: {err:?}"
                )));
            }
        };
        if let Err(err) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            return Err(other_error(format!(
                "Vulkan shared buffer bind failed: {err:?}"
            )));
        }
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let fd = match unsafe { self.external_memory_fd_loader.get_memory_fd(&fd_info) } {
            Ok(fd) => fd,
            Err(err) => {
                unsafe {
                    self.device.free_memory(memory, None);
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(other_error(format!(
                    "Vulkan exportable memory FD query failed: {err:?}"
                )));
            }
        };
        let (hip_external_memory, hip_mapped_ptr) =
            match import_hip_external_memory_fd(fd, requirements.size, bytes) {
                Ok(imported) => imported,
                Err(err) => {
                    unsafe {
                        drop(OwnedFd::from_raw_fd(fd));
                        self.device.free_memory(memory, None);
                        self.device.destroy_buffer(buffer, None);
                    }
                    return Err(err);
                }
            };
        println!(
            "HIP/Vulkan zero-copy buffer: {} bytes imported from OPAQUE_FD Vulkan device memory",
            bytes
        );
        Ok(VulkanSharedMemory {
            buffer,
            memory,
            hip_external_memory,
            hip_mapped_ptr,
            bytes,
        })
    }

    fn create_overlay_memory(
        &self,
        bytes: usize,
    ) -> Result<VulkanOverlayMemory, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(bytes as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe {
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(other_error(format!(
                    "Vulkan overlay buffer allocation failed: {err:?}"
                )));
            }
        };
        if let Err(err) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            return Err(other_error(format!(
                "Vulkan overlay buffer bind failed: {err:?}"
            )));
        }
        let mapped = match unsafe {
            self.device.map_memory(
                memory,
                0,
                bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(mapped) => mapped.cast::<u32>(),
            Err(err) => {
                unsafe {
                    self.device.free_memory(memory, None);
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(other_error(format!(
                    "Vulkan overlay buffer map failed: {err:?}"
                )));
            }
        };
        Ok(VulkanOverlayMemory {
            buffer,
            memory,
            mapped_ptr: mapped,
            bytes,
        })
    }

    fn create_frame_image(
        &self,
        size: RenderSize,
        format: vk::Format,
    ) -> Result<(vk::Image, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: size.width as u32,
                height: size.height as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&image_info, None)? };
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_type_index = find_memory_type(
            self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe {
                    self.device.destroy_image(image, None);
                }
                return Err(other_error(format!(
                    "Vulkan transfer image allocation failed: {err:?}"
                )));
            }
        };
        if let Err(err) = unsafe { self.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_image(image, None);
            }
            return Err(other_error(format!(
                "Vulkan transfer image bind failed: {err:?}"
            )));
        }
        Ok((image, memory))
    }

    fn destroy_frame_resources(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if !self.hip_external_memory.is_null() {
                let _ = hipDestroyExternalMemory(self.hip_external_memory);
                self.hip_external_memory = ptr::null_mut();
                self.hip_mapped_ptr = ptr::null_mut();
            }
            if self.shared_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.shared_buffer, None);
                self.shared_buffer = vk::Buffer::null();
            }
            if self.shared_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.shared_memory, None);
                self.shared_memory = vk::DeviceMemory::null();
            }
            if self.frame_image != vk::Image::null() {
                self.device.destroy_image(self.frame_image, None);
                self.frame_image = vk::Image::null();
            }
            if self.frame_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.frame_memory, None);
                self.frame_memory = vk::DeviceMemory::null();
            }
            if self.overlay_memory != vk::DeviceMemory::null() {
                self.device.unmap_memory(self.overlay_memory);
                self.overlay_mapped_ptr = ptr::null_mut();
            }
            if self.overlay_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.overlay_buffer, None);
                self.overlay_buffer = vk::Buffer::null();
            }
            if self.overlay_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.overlay_memory, None);
                self.overlay_memory = vk::DeviceMemory::null();
            }
            if self.image_available != vk::Semaphore::null() {
                self.device.destroy_semaphore(self.image_available, None);
                self.image_available = vk::Semaphore::null();
            }
            if self.render_finished != vk::Semaphore::null() {
                self.device.destroy_semaphore(self.render_finished, None);
                self.render_finished = vk::Semaphore::null();
            }
            if self.in_flight != vk::Fence::null() {
                self.device.destroy_fence(self.in_flight, None);
                self.in_flight = vk::Fence::null();
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
                self.command_pool = vk::CommandPool::null();
                self.command_buffer = vk::CommandBuffer::null();
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
            self.swapchain_images.clear();
            self.swapchain_image_initialized.clear();
            self.frame_image_initialized = false;
            self.shared_bytes = 0;
            self.overlay_bytes = 0;
        }
    }
}

impl Drop for VulkanPresenter {
    fn drop(&mut self) {
        self.destroy_frame_resources();
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_vulkan_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), Box<dyn std::error::Error>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    for physical_device in physical_devices {
        if !vulkan_device_has_extension(instance, physical_device, ash::khr::swapchain::NAME)?
            || !vulkan_device_has_extension(
                instance,
                physical_device,
                ash::khr::external_memory_fd::NAME,
            )?
        {
            continue;
        }
        if !vulkan_external_buffer_exportable(instance, physical_device) {
            continue;
        }
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, queue_family) in queue_families.iter().enumerate() {
            let supports_graphics = queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_present = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )?
            };
            if supports_graphics && supports_present {
                return Ok((physical_device, index as u32));
            }
        }
    }
    Err(other_error(
        "no Vulkan device supports graphics+present and exportable OPAQUE_FD transfer buffers",
    ))
}

fn vulkan_device_has_extension(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    name: &CStr,
) -> Result<bool, Box<dyn std::error::Error>> {
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
    Ok(extensions.iter().any(|extension| {
        let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
        extension_name == name
    }))
}

fn vulkan_external_buffer_exportable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut properties = vk::ExternalBufferProperties::default();
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            physical_device,
            &info,
            &mut properties,
        );
    }
    properties
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
}

fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> Result<vk::SurfaceFormatKHR, Box<dyn std::error::Error>> {
    if formats.is_empty() {
        return Err(other_error("Vulkan surface reported no formats"));
    }
    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        return Ok(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });
    }
    formats
        .iter()
        .copied()
        .find(|format| {
            matches!(
                format.format,
                vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB
            ) && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first().copied())
        .ok_or_else(|| other_error("Vulkan surface format selection failed"))
}

fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE
    } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

fn choose_swapchain_extent(
    window: &sdl2::video::Window,
    surface_caps: vk::SurfaceCapabilitiesKHR,
    size: RenderSize,
    present_scale: usize,
) -> Result<vk::Extent2D, Box<dyn std::error::Error>> {
    if surface_caps.current_extent.width != u32::MAX {
        return Ok(surface_caps.current_extent);
    }
    let (drawable_w, drawable_h) = window.vulkan_drawable_size();
    let desired_w = drawable_w.max(checked_window_dim(size.width, present_scale)?);
    let desired_h = drawable_h.max(checked_window_dim(size.height, present_scale)?);
    Ok(vk::Extent2D {
        width: desired_w.clamp(
            surface_caps.min_image_extent.width,
            surface_caps.max_image_extent.width,
        ),
        height: desired_h.clamp(
            surface_caps.min_image_extent.height,
            surface_caps.max_image_extent.height,
        ),
    })
}

fn swapchain_image_count(surface_caps: vk::SurfaceCapabilitiesKHR) -> u32 {
    let requested = surface_caps.min_image_count.saturating_add(1);
    if surface_caps.max_image_count > 0 {
        requested.min(surface_caps.max_image_count)
    } else {
        requested
    }
}

fn choose_composite_alpha(surface_caps: vk::SurfaceCapabilitiesKHR) -> vk::CompositeAlphaFlagsKHR {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .into_iter()
    .find(|mode| surface_caps.supported_composite_alpha.contains(*mode))
    .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE)
}

fn color_subresource_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn import_hip_external_memory_fd(
    fd: c_int,
    allocation_size: vk::DeviceSize,
    buffer_size: usize,
) -> Result<(HipExternalMemory, *mut u32), Box<dyn std::error::Error>> {
    let handle_desc = HipExternalMemoryHandleDesc {
        handle_type: HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        handle: HipExternalMemoryHandle { fd },
        size: allocation_size,
        flags: 0,
        reserved: [0; 16],
    };
    let mut external_memory = ptr::null_mut();
    unsafe {
        hip_context(
            "hipImportExternalMemory(Vulkan OPAQUE_FD)",
            rocm_oxide::hip::check(hipImportExternalMemory(&mut external_memory, &handle_desc)),
        )?;
    }
    let buffer_desc = HipExternalMemoryBufferDesc {
        offset: 0,
        size: buffer_size as u64,
        flags: 0,
        reserved: [0; 16],
    };
    let mut mapped = ptr::null_mut();
    let mapped_result = unsafe {
        hip_context(
            "hipExternalMemoryGetMappedBuffer(Vulkan shared buffer)",
            rocm_oxide::hip::check(hipExternalMemoryGetMappedBuffer(
                &mut mapped,
                external_memory,
                &buffer_desc,
            )),
        )
    };
    if let Err(err) = mapped_result {
        unsafe {
            let _ = hipDestroyExternalMemory(external_memory);
        }
        return Err(err);
    }
    Ok((external_memory, mapped.cast::<u32>()))
}

fn find_memory_type(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32, Box<dyn std::error::Error>> {
    for index in 0..memory_properties.memory_type_count {
        let supported = (memory_type_bits & (1 << index)) != 0;
        let properties = memory_properties.memory_types[index as usize].property_flags;
        if supported && properties.contains(required) {
            return Ok(index);
        }
    }
    Err(other_error(format!(
        "no Vulkan memory type matches bits {memory_type_bits:#x} and flags {required:?}"
    )))
}

impl GlPresenter {
    fn new(
        sdl: &sdl2::Sdl,
        size: RenderSize,
        present_scale: usize,
        fps_limit: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let video = sdl.video().map_err(other_error)?;
        {
            let attrs = video.gl_attr();
            attrs.set_context_profile(sdl2::video::GLProfile::Core);
            attrs.set_context_version(3, 3);
            attrs.set_double_buffer(true);
        }
        let window = video
            .window(
                "ROCm-Oxide Spectral Lattice GL",
                checked_window_dim(size.width, present_scale)?,
                checked_window_dim(size.height, present_scale)?,
            )
            .opengl()
            .resizable()
            .position_centered()
            .build()
            .map_err(|err| other_error(err.to_string()))?;
        let context = window.gl_create_context().map_err(other_error)?;
        window.gl_make_current(&context).map_err(other_error)?;
        gl::load_with(|symbol| video.gl_get_proc_address(symbol).cast());
        let _ = video.gl_set_swap_interval(0);
        println!(
            "OpenGL renderer: {} ({})",
            gl_string(gl::RENDERER),
            gl_string(gl::VENDOR)
        );
        let gl_devices = hip_gl_devices().map_err(|err| {
            other_error(format!(
                "HIP/OpenGL interop is unavailable for this SDL context: {err}. Try SDL_VIDEODRIVER=x11 if it was overridden."
            ))
        })?;
        if gl_devices.is_empty() {
            return Err(other_error(
                "HIP/OpenGL interop did not report any devices for the current GL context",
            ));
        }
        println!("HIP devices visible to current GL context: {gl_devices:?}");

        let program = create_present_program()?;
        let mut vao = 0;
        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);
            gl::UseProgram(program);
            for (name, unit) in [("u_frame", 0), ("u_overlay", 1)] {
                let sampler = CString::new(name)?;
                let location = gl::GetUniformLocation(program, sampler.as_ptr());
                if location >= 0 {
                    gl::Uniform1i(location, unit);
                }
            }
        }
        check_gl("create GL presenter")?;

        let mut presenter = Self {
            window,
            _context: context,
            size,
            present_scale,
            texture: 0,
            pbo: 0,
            overlay_texture: 0,
            vao,
            program,
            resource: ptr::null_mut(),
        };
        presenter.recreate_frame_resources(size, present_scale)?;
        presenter.window.set_title(&format!(
            "ROCm-Oxide Spectral Lattice GL | {} | limit {}",
            display_label_for(size, present_scale),
            fps_label(fps_limit)
        ))?;
        Ok(presenter)
    }

    fn recreate_frame_resources(
        &mut self,
        size: RenderSize,
        present_scale: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.destroy_frame_resources();
        self.window
            .set_size(
                checked_window_dim(size.width, present_scale)?,
                checked_window_dim(size.height, present_scale)?,
            )
            .map_err(|err| other_error(err.to_string()))?;

        let byte_len = gl_frame_byte_len(size)?;
        let width = gl_size(size.width, "frame width")?;
        let height = gl_size(size.height, "frame height")?;
        let overlay_size = overlay_size_for(size);
        let overlay_width = gl_size(overlay_size.width, "overlay width")?;
        let overlay_height = gl_size(overlay_size.height, "overlay height")?;
        let mut texture = 0;
        let mut overlay_texture = 0;
        let mut pbo = 0;
        unsafe {
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_S,
                gl::CLAMP_TO_EDGE as GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_T,
                gl::CLAMP_TO_EDGE as GLint,
            );
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 4);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as GLint,
                width,
                height,
                0,
                gl::BGRA,
                gl::UNSIGNED_BYTE,
                ptr::null(),
            );

            gl::GenTextures(1, &mut overlay_texture);
            gl::BindTexture(gl::TEXTURE_2D, overlay_texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_S,
                gl::CLAMP_TO_EDGE as GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_T,
                gl::CLAMP_TO_EDGE as GLint,
            );
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as GLint,
                overlay_width,
                overlay_height,
                0,
                gl::BGRA,
                gl::UNSIGNED_BYTE,
                ptr::null(),
            );

            gl::GenBuffers(1, &mut pbo);
            gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, pbo);
            gl::BufferData(
                gl::PIXEL_UNPACK_BUFFER,
                byte_len,
                ptr::null(),
                gl::STREAM_DRAW,
            );
            gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
            gl::Finish();
        }
        check_gl("allocate GL frame resources")?;

        let mut resource = ptr::null_mut();
        let register = unsafe {
            hip_context(
                "hipGraphicsGLRegisterBuffer",
                rocm_oxide::hip::check(hipGraphicsGLRegisterBuffer(
                    &mut resource,
                    pbo,
                    HIP_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
                )),
            )
        };
        if let Err(err) = register {
            unsafe {
                gl::DeleteBuffers(1, &pbo);
                gl::DeleteTextures(1, &overlay_texture);
                gl::DeleteTextures(1, &texture);
            }
            return Err(err);
        }

        self.size = size;
        self.present_scale = present_scale;
        self.texture = texture;
        self.pbo = pbo;
        self.overlay_texture = overlay_texture;
        self.resource = resource;
        Ok(())
    }

    fn present_device_frame(
        &mut self,
        source: &DeviceBuffer<u32>,
        overlay: &OverlayFrame,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let interop_start = Instant::now();
        let mapped = self.map_buffer()?;
        if mapped.len < source.len() {
            self.unmap_buffer()?;
            return Err(other_error(format!(
                "mapped GL buffer is too small: got {} u32 values, need {}",
                mapped.len,
                source.len()
            )));
        }
        unsafe {
            source.copy_to_device_ptr(mapped.ptr, source.len())?;
        }
        self.unmap_buffer()?;
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, self.texture);
            gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, self.pbo);
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                gl_size(self.size.width, "frame width")?,
                gl_size(self.size.height, "frame height")?,
                gl::BGRA,
                gl::UNSIGNED_BYTE,
                ptr::null(),
            );
            gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        }
        check_gl("upload GL texture from HIP-mapped PBO")?;
        unsafe {
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, self.overlay_texture);
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 4);
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                gl_size(overlay.size.width, "overlay width")?,
                gl_size(overlay.size.height, "overlay height")?,
                gl::BGRA,
                gl::UNSIGNED_BYTE,
                overlay.pixels.as_ptr().cast(),
            );
        }
        check_gl("upload GL overlay texture")?;
        let interop_ms = interop_start.elapsed().as_secs_f64() * 1000.0;

        let present_start = Instant::now();
        unsafe {
            let (drawable_w, drawable_h) = self.window.drawable_size();
            gl::Viewport(0, 0, drawable_w as GLsizei, drawable_h as GLsizei);
            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::UseProgram(self.program);
            let fraction_name = CString::new("u_overlay_panel_fraction")?;
            let fraction_location = gl::GetUniformLocation(self.program, fraction_name.as_ptr());
            if fraction_location >= 0 {
                gl::Uniform1f(
                    fraction_location,
                    overlay.size.width as f32 / self.size.width.max(1) as f32,
                );
            }
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, self.texture);
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, self.overlay_texture);
            gl::BindVertexArray(self.vao);
            gl::DrawArrays(gl::TRIANGLES, 0, 3);
        }
        check_gl("draw GL presenter")?;
        self.window.gl_swap_window();
        let present_ms = present_start.elapsed().as_secs_f64() * 1000.0;
        Ok((interop_ms, present_ms))
    }

    fn map_buffer(&mut self) -> Result<MappedGlBuffer, Box<dyn std::error::Error>> {
        unsafe {
            hip_context(
                "hipGraphicsMapResources",
                rocm_oxide::hip::check(hipGraphicsMapResources(
                    1,
                    &mut self.resource,
                    Stream::null().as_raw(),
                )),
            )?;
        }
        let mut ptr = ptr::null_mut();
        let mut bytes = 0usize;
        unsafe {
            hip_context(
                "hipGraphicsResourceGetMappedPointer",
                rocm_oxide::hip::check(hipGraphicsResourceGetMappedPointer(
                    &mut ptr,
                    &mut bytes,
                    self.resource,
                )),
            )?;
        }
        Ok(MappedGlBuffer {
            ptr: ptr.cast::<u32>(),
            len: bytes / std::mem::size_of::<u32>(),
        })
    }

    fn unmap_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            hip_context(
                "hipGraphicsUnmapResources",
                rocm_oxide::hip::check(hipGraphicsUnmapResources(
                    1,
                    &mut self.resource,
                    Stream::null().as_raw(),
                )),
            )?;
        }
        Ok(())
    }

    fn destroy_frame_resources(&mut self) {
        unsafe {
            gl::Finish();
            if !self.resource.is_null() {
                let _ = hipGraphicsUnregisterResource(self.resource);
                self.resource = ptr::null_mut();
            }
            if self.pbo != 0 {
                gl::DeleteBuffers(1, &self.pbo);
                self.pbo = 0;
            }
            if self.overlay_texture != 0 {
                gl::DeleteTextures(1, &self.overlay_texture);
                self.overlay_texture = 0;
            }
            if self.texture != 0 {
                gl::DeleteTextures(1, &self.texture);
                self.texture = 0;
            }
        }
    }
}

impl Drop for GlPresenter {
    fn drop(&mut self) {
        self.destroy_frame_resources();
        unsafe {
            if self.program != 0 {
                gl::DeleteProgram(self.program);
            }
            if self.vao != 0 {
                gl::DeleteVertexArrays(1, &self.vao);
            }
        }
    }
}

fn create_present_program() -> Result<GLuint, Box<dyn std::error::Error>> {
    const VERTEX: &str = r#"#version 330 core
out vec2 v_uv;
const vec2 vertices[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);
void main() {
    vec2 position = vertices[gl_VertexID];
    v_uv = (position + vec2(1.0)) * 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"#;
    const FRAGMENT: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D u_frame;
uniform sampler2D u_overlay;
uniform float u_overlay_panel_fraction;
out vec4 color;
void main() {
    vec2 frame_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 frame = texture(u_frame, frame_uv);
    float panel = max(u_overlay_panel_fraction, 0.0001);
    if (v_uv.x <= panel) {
        vec4 overlay = texture(u_overlay, vec2(v_uv.x / panel, frame_uv.y));
        frame = mix(frame, vec4(overlay.rgb, 1.0), overlay.a);
    }
    color = frame;
}
"#;

    let vertex = compile_shader(gl::VERTEX_SHADER, VERTEX)?;
    let fragment = compile_shader(gl::FRAGMENT_SHADER, FRAGMENT)?;
    let program = unsafe { gl::CreateProgram() };
    unsafe {
        gl::AttachShader(program, vertex);
        gl::AttachShader(program, fragment);
        gl::LinkProgram(program);
    }
    let mut status = 0;
    unsafe {
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);
        gl::DeleteShader(vertex);
        gl::DeleteShader(fragment);
    }
    if status == 0 {
        let log = program_log(program);
        unsafe {
            gl::DeleteProgram(program);
        }
        return Err(other_error(format!(
            "GL presenter program link failed: {log}"
        )));
    }
    Ok(program)
}

fn compile_shader(kind: GLenum, source: &str) -> Result<GLuint, Box<dyn std::error::Error>> {
    let shader = unsafe { gl::CreateShader(kind) };
    let source = CString::new(source)?;
    unsafe {
        gl::ShaderSource(shader, 1, &source.as_ptr(), ptr::null());
        gl::CompileShader(shader);
    }
    let mut status = 0;
    unsafe {
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);
    }
    if status == 0 {
        let log = shader_log(shader);
        unsafe {
            gl::DeleteShader(shader);
        }
        return Err(other_error(format!(
            "GL presenter shader compile failed: {log}"
        )));
    }
    Ok(shader)
}

fn shader_log(shader: GLuint) -> String {
    let mut len = 0;
    unsafe {
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
    }
    let mut buffer = vec![0u8; len.max(1) as usize];
    unsafe {
        gl::GetShaderInfoLog(
            shader,
            len,
            ptr::null_mut(),
            buffer.as_mut_ptr().cast::<GLchar>(),
        );
    }
    String::from_utf8_lossy(&buffer)
        .trim_end_matches('\0')
        .trim()
        .to_string()
}

fn program_log(program: GLuint) -> String {
    let mut len = 0;
    unsafe {
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
    }
    let mut buffer = vec![0u8; len.max(1) as usize];
    unsafe {
        gl::GetProgramInfoLog(
            program,
            len,
            ptr::null_mut(),
            buffer.as_mut_ptr().cast::<GLchar>(),
        );
    }
    String::from_utf8_lossy(&buffer)
        .trim_end_matches('\0')
        .trim()
        .to_string()
}

fn check_gl(label: &str) -> Result<(), Box<dyn std::error::Error>> {
    let error = unsafe { gl::GetError() };
    if error == gl::NO_ERROR {
        Ok(())
    } else {
        Err(other_error(format!("{label}: GL error 0x{error:04x}")))
    }
}

fn gl_string(name: GLenum) -> String {
    let ptr = unsafe { gl::GetString(name) };
    if ptr.is_null() {
        "<unknown>".into()
    } else {
        unsafe {
            std::ffi::CStr::from_ptr(ptr.cast())
                .to_string_lossy()
                .into_owned()
        }
    }
}

fn hip_gl_devices() -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let mut count = 0u32;
    let mut devices = [0i32; 16];
    unsafe {
        hip_context(
            "hipGLGetDevices",
            rocm_oxide::hip::check(hipGLGetDevices(
                &mut count,
                devices.as_mut_ptr(),
                devices.len() as c_uint,
                HIP_GL_DEVICE_LIST_ALL,
            )),
        )?;
    }
    Ok(devices[..(count as usize).min(devices.len())].to_vec())
}

fn hip_context(
    label: &str,
    result: rocm_oxide::hip::Result<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    result.map_err(|err| other_error(format!("{label}: {err}")))
}

fn gl_frame_byte_len(size: RenderSize) -> Result<isize, Box<dyn std::error::Error>> {
    let bytes = size
        .pixel_count()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| other_error("GL frame byte length overflows usize"))?;
    isize::try_from(bytes).map_err(|_| other_error("GL frame byte length exceeds isize"))
}

fn frame_byte_len_usize(size: RenderSize) -> Result<usize, Box<dyn std::error::Error>> {
    size.pixel_count()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| other_error("frame byte length overflows usize"))
}

fn frame_byte_len(size: RenderSize) -> Result<usize, Box<dyn std::error::Error>> {
    frame_byte_len_usize(size)
}

fn gl_size(value: usize, label: &str) -> Result<GLint, Box<dyn std::error::Error>> {
    GLint::try_from(value).map_err(|_| other_error(format!("{label} exceeds GLsizei range")))
}

fn checked_window_dim(value: usize, scale: usize) -> Result<u32, Box<dyn std::error::Error>> {
    value
        .checked_mul(scale)
        .and_then(|dim| u32::try_from(dim).ok())
        .ok_or_else(|| other_error(format!("window dimension {value} x {scale} overflows u32")))
}

fn other_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::other(message.into()))
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
            parity_line: (if parity.cluster_launch.requires_runtime_capability {
                "parity: cooperative launch path"
            } else {
                "parity: stream tiled path"
            })
            .to_string(),
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
            tile_count,
        })
    }
}

fn render_workload(
    kernels: &generated::DeviceKernels,
    buffers: &DemoBuffers,
    state: &DemoState,
) -> rocm_oxide::Result<bool> {
    render_frame(kernels, buffers, state, state.frame_index)
}

fn render_workload_timed(
    kernels: &generated::DeviceKernels,
    buffers: &DemoBuffers,
    state: &DemoState,
) -> Result<(bool, f32), Box<dyn std::error::Error>> {
    let stream = Stream::null();
    let start = Event::new()?;
    let stop = Event::new()?;
    start.record(&stream)?;
    let use_post = render_workload(kernels, buffers, state)?;
    stop.record(&stream)?;
    stop.synchronize()?;
    Ok((use_post, start.elapsed_ms_until(&stop)?))
}

fn render_workload_queued(
    kernels: &generated::DeviceKernels,
    buffers: &DemoBuffers,
    state: &DemoState,
) -> Result<(bool, QueuedGpuTiming), Box<dyn std::error::Error>> {
    let stream = Stream::null();
    let start = Event::new()?;
    let stop = Event::new()?;
    start.record(&stream)?;
    let use_post = render_workload(kernels, buffers, state)?;
    stop.record(&stream)?;
    Ok((use_post, QueuedGpuTiming { start, stop }))
}

fn render_frame(
    kernels: &generated::DeviceKernels,
    buffers: &DemoBuffers,
    state: &DemoState,
    frame_index: u32,
) -> rocm_oxide::Result<bool> {
    unsafe {
        kernels.spectral_lattice(
            LaunchConfig::for_num_elems_with_block_size(buffers.size.pixel_count(), BLOCK_X),
            &buffers.base,
            buffers.size.width as u32,
            buffers.size.height as u32,
            buffers.size.pixel_count(),
            frame_index,
            state.mode as u32,
            state.palette[0],
            state.palette[1],
            state.palette[2],
            state.warp,
            state.gain,
            state.gpu_work as u32,
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
            buffers.histogram.set_zero()?;
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
                    frame_index,
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
                    frame_index,
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
            Key::M => cycle_present_scale(state),
            Key::LeftBracket => step_gpu_work(state, -1),
            Key::RightBracket => step_gpu_work(state, 1),
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
    } else if scale_rect(slider_rect(4), scale).contains(x, y) {
        set_gpu_work_from_slider(state, x);
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

fn cycle_present_scale(state: &mut DemoState) {
    let current = PRESENT_SCALES
        .iter()
        .position(|scale| *scale == state.present_scale)
        .unwrap_or(0);
    let next = (current + 1) % PRESENT_SCALES.len();
    state.present_scale = PRESENT_SCALES[next];
    state.status = format!("present scale set to {}x", state.present_scale);
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

fn set_gpu_work_from_slider(state: &mut DemoState, x: usize) {
    let rect = scale_rect(slider_rect(4), ui_scale(state.render_size));
    let t = ((x.saturating_sub(rect.x)) as f32 / rect.w.max(1) as f32).clamp(0.0, 1.0);
    let index = (t * (GPU_WORK_PRESETS.len() - 1) as f32).round() as usize;
    state.gpu_work = GPU_WORK_PRESETS[index.min(GPU_WORK_PRESETS.len() - 1)];
}

fn step_gpu_work(state: &mut DemoState, delta: isize) {
    let current = gpu_work_index(state.gpu_work);
    let len = GPU_WORK_PRESETS.len() as isize;
    let next = (current as isize + delta).rem_euclid(len) as usize;
    state.gpu_work = GPU_WORK_PRESETS[next];
    state.status = format!("GPU work set to {}x", state.gpu_work);
}

fn gpu_work_index(work: usize) -> usize {
    GPU_WORK_PRESETS
        .iter()
        .position(|value| *value == work)
        .unwrap_or_else(|| {
            GPU_WORK_PRESETS
                .iter()
                .enumerate()
                .min_by_key(|(_, value)| value.abs_diff(work))
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
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
            state.gpu_work as u32,
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
    draw_overlay_contents(frame, size, state, resources);
}

fn draw_overlay_contents(
    frame: &mut [u32],
    size: RenderSize,
    state: &DemoState,
    resources: &ResourceSnapshot,
) {
    let scale = ui_scale(size);
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

    for (index, mode) in MODES.iter().enumerate() {
        let rect = scale_rect(mode_rect(index), scale);
        let active = index == state.mode;
        draw_button(
            frame,
            size,
            scale,
            rect,
            mode,
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
        &format!("Res {}", display_label(state)),
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
    draw_gpu_work_slider(
        frame,
        size,
        scale,
        scale_rect(slider_rect(4), scale),
        state.gpu_work,
    );

    draw_text(
        frame,
        size,
        scale,
        24 * scale,
        468 * scale,
        "Runtime",
        0xffffff,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        486 * scale,
        &resources.resource_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        504 * scale,
        &resources.launch_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        522 * scale,
        &resources.library_line,
        0xc7d4e0,
        (PANEL_W - 42) * scale,
    );
    draw_text_clipped(
        frame,
        size,
        scale,
        24 * scale,
        540 * scale,
        &format!(
            "gpu{:.2} copy{:.2} draw{:.2}",
            state.gpu_ms, state.copy_ms, state.draw_ms
        ),
        0x9adfb1,
        (PANEL_W - 42) * scale,
    );
    if size.height >= 596 * scale {
        draw_text_clipped(
            frame,
            size,
            scale,
            24 * scale,
            558 * scale,
            &format!(
                "present{:.2} frame{:.2} work{}x",
                state.present_ms, state.frame_ms, state.gpu_work
            ),
            0x9adfb1,
            (PANEL_W - 42) * scale,
        );
        draw_text_clipped(
            frame,
            size,
            scale,
            24 * scale,
            576 * scale,
            &resources.parity_line,
            0xc7d4e0,
            (PANEL_W - 42) * scale,
        );
        draw_text_clipped(
            frame,
            size,
            scale,
            24 * scale,
            594 * scale,
            &format!("palette: {}", state.palette_source),
            0x9adfb1,
            (PANEL_W - 42) * scale,
        );
        if size.height >= 614 * scale {
            draw_text_clipped(
                frame,
                size,
                scale,
                24 * scale,
                612 * scale,
                &state.status,
                0xffcc8a,
                (PANEL_W - 42) * scale,
            );
        }
    } else {
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
    }
}

fn overlay_panel_background_rgb() -> u32 {
    blend(0, 0x071018, 224)
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

fn draw_gpu_work_slider(
    frame: &mut [u32],
    size: RenderSize,
    scale: usize,
    rect: Rect,
    work: usize,
) {
    draw_text(
        frame,
        size,
        scale,
        rect.x,
        rect.y.saturating_sub(18 * scale),
        &format!("GPU Work {}x", work),
        0xdce8f4,
    );
    blend_rect(frame, size, rect.x, rect.y, rect.w, rect.h, 0x142434, 210);
    let index = gpu_work_index(work);
    let t = index as f32 / (GPU_WORK_PRESETS.len() - 1) as f32;
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
    Rect::new(24, 292 + index * 34, PANEL_W - 48, 14)
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
    let mut gpu_work = DEFAULT_GPU_WORK;
    let mut present_scale = 1usize;
    let mut present_backend = PresentBackend::Cpu;
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
            "--gpu-work" | "--work" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--gpu-work requires an iteration count".to_string())?;
                gpu_work = parse_gpu_work(&value)?;
            }
            "--present" | "--present-backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--present requires cpu, gl, or vulkan".to_string())?;
                present_backend = parse_present_backend(&value)?;
            }
            "--present-scale" | "--scale" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--present-scale requires 1, 2, or 4".to_string())?;
                present_scale = parse_present_scale(&value)?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --example spectral_lattice -- [--frames N] [--mode MODE] [--resolution 4k|WIDTHxHEIGHT] [--present cpu|gl|vulkan] [--present-scale 1|2|4] [--fps-limit FPS|uncapped] [--gpu-work ITERATIONS] [--output PATH]"
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
        gpu_work,
        present_scale,
        present_backend,
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

fn parse_gpu_work(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let iterations = value.parse::<usize>()?;
    if !(1..=1024).contains(&iterations) {
        return Err(format!(
            "GPU work iteration count {iterations} is outside supported bounds 1..1024"
        )
        .into());
    }
    Ok(iterations)
}

fn parse_present_backend(value: &str) -> Result<PresentBackend, Box<dyn std::error::Error>> {
    match value.to_ascii_lowercase().as_str() {
        "cpu" | "minifb" | "readback" => Ok(PresentBackend::Cpu),
        "gl" | "opengl" | "native" | "gpu" => Ok(PresentBackend::Gl),
        "vk" | "vulkan" | "hip-vulkan" | "zero-copy" => Ok(PresentBackend::Vulkan),
        _ => Err(format!("unknown present backend `{value}`; expected cpu, gl, or vulkan").into()),
    }
}

fn parse_present_scale(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let scale = value
        .strip_suffix('x')
        .or_else(|| value.strip_suffix('X'))
        .unwrap_or(value)
        .parse::<usize>()?;
    if PRESENT_SCALES.contains(&scale) {
        Ok(scale)
    } else {
        Err(format!("present scale {scale} is outside supported values 1, 2, or 4").into())
    }
}

fn minifb_scale(scale: usize) -> Scale {
    match scale {
        2 => Scale::X2,
        4 => Scale::X4,
        _ => Scale::X1,
    }
}

fn display_label(state: &DemoState) -> String {
    display_label_for(state.render_size, state.present_scale)
}

fn display_label_for(size: RenderSize, present_scale: usize) -> String {
    let label = size.label();
    if present_scale == 1 {
        label
    } else {
        format!("{label} x{present_scale}")
    }
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
