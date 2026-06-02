use ash::vk::Handle;
use ash::{Entry, vk};
use image::{Rgb, RgbImage};
use libwayshot_xcap::WayshotConnection;
use libwayshot_xcap::region::{
    EmbeddedRegion, LogicalRegion, Position, Region, Size as WayshotSize,
};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig, Stream};
use sdl2::event::Event as SdlEvent;
use sdl2::keyboard::Keycode;
use std::ffi::{CStr, CString, c_int, c_uint, c_void};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::path::PathBuf;
use std::ptr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, RecvTimeoutError},
};
use std::thread;
use std::time::{Duration, Instant};
use xcap::{Frame as VideoFrame, Monitor, VideoRecorder};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));

    impl DeviceKernels {
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn matrix_lens_fx_external(
            &self,
            config: rocm_oxide::LaunchConfig,
            frame_ptr: *mut u32,
            frame_len: usize,
            input_ptr: *const u32,
            input_len: usize,
            width: u32,
            height: u32,
            pixel_count: usize,
            frame_index: u32,
            mode: u32,
        ) -> rocm_oxide::Result<()> {
            rocm_oxide::validate_launch_config(config)?;
            rocm_oxide::validate_buffer_len("frame", frame_len, pixel_count)?;
            rocm_oxide::validate_buffer_len("input", input_len, pixel_count)?;
            if frame_ptr.is_null() || input_ptr.is_null() {
                return Err(rocm_oxide::Error::InvalidLaunch(
                    "matrix_lens_fx_external received a null imported pointer".into(),
                ));
            }
            let frame_start = frame_ptr as usize;
            let frame_end = frame_start.saturating_add(frame_len.saturating_mul(4));
            let input_start = input_ptr as usize;
            let input_end = input_start.saturating_add(input_len.saturating_mul(4));
            if frame_start < input_end && input_start < frame_end {
                return Err(rocm_oxide::Error::InvalidLaunch(
                    "matrix_lens_fx_external frame/input buffers alias".into(),
                ));
            }
            let mut __arg0 = frame_ptr;
            let mut __arg1 = frame_len;
            let mut __arg2 = input_ptr;
            let mut __arg3 = input_len;
            let mut __arg4 = width;
            let mut __arg5 = height;
            let mut __arg6 = pixel_count;
            let mut __arg7 = frame_index;
            let mut __arg8 = mode;
            let mut __params = [
                rocm_oxide::__private::arg_ptr(&mut __arg0),
                rocm_oxide::__private::arg_ptr(&mut __arg1),
                rocm_oxide::__private::arg_ptr(&mut __arg2),
                rocm_oxide::__private::arg_ptr(&mut __arg3),
                rocm_oxide::__private::arg_ptr(&mut __arg4),
                rocm_oxide::__private::arg_ptr(&mut __arg5),
                rocm_oxide::__private::arg_ptr(&mut __arg6),
                rocm_oxide::__private::arg_ptr(&mut __arg7),
                rocm_oxide::__private::arg_ptr(&mut __arg8),
            ];
            unsafe {
                self.__kernel_matrix_lens_fx
                    .launch_raw(config, &mut __params)
            }
        }
    }
}

const DEFAULT_OUTPUT: &str = "target/matrix_lens.png";
const DEFAULT_FPS_LIMIT: usize = 60;
const DEFAULT_DRM_RENDER_NODE: &str = "/dev/dri/renderD128";
const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_int = 1;
const DRM_FORMAT_ARGB8888: u32 = fourcc(*b"AR24");
const DRM_FORMAT_XRGB8888: u32 = fourcc(*b"XR24");
const DRM_FORMAT_ABGR8888: u32 = fourcc(*b"AB24");
const DRM_FORMAT_XBGR8888: u32 = fourcc(*b"XB24");
const MODES: [&str; 4] = ["matrix", "glass", "thermal", "xray"];
const RESOLUTION_PRESETS: [ResolutionPreset; 3] = [
    ResolutionPreset::new("540p", 960, 540),
    ResolutionPreset::new("720p", 1280, 720),
    ResolutionPreset::new("1080p", 1920, 1080),
];

type HipExternalMemory = *mut c_void;

const fn fourcc(code: [u8; 4]) -> u32 {
    (code[0] as u32) | ((code[1] as u32) << 8) | ((code[2] as u32) << 16) | ((code[3] as u32) << 24)
}

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

#[derive(Clone, Copy)]
struct ResolutionPreset {
    label: &'static str,
    size: RenderSize,
}

impl ResolutionPreset {
    const fn new(label: &'static str, width: usize, height: usize) -> Self {
        Self {
            label,
            size: RenderSize { width, height },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RenderSize {
    width: usize,
    height: usize,
}

impl RenderSize {
    fn pixel_count(self) -> usize {
        self.width * self.height
    }
}

struct DemoArgs {
    frames: Option<u32>,
    output: PathBuf,
    mode: usize,
    size: RenderSize,
    fps_limit: usize,
}

#[derive(Clone, Copy)]
struct CaptureRequest {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

struct SharedCapture {
    pixels: Vec<u32>,
    sequence: u64,
    captures: u64,
    errors: u64,
    status: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonitorKey {
    id: u32,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

struct ActiveVideoStream {
    key: MonitorKey,
    name: String,
    recorder: VideoRecorder,
    receiver: Receiver<VideoFrame>,
    latest_frame: Option<VideoFrame>,
}

struct GpuCaptureBackend {
    wayshot: WayshotConnection,
    render_node: String,
}

struct GpuCaptureFrame {
    fd: OwnedFd,
    drm_format: u32,
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
    modifier: u64,
    status: String,
}

struct ImportedDmaImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
}

struct VulkanSharedMemory {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    hip_external_memory: HipExternalMemory,
    hip_mapped_ptr: *mut u32,
    bytes: usize,
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
    supports_dma_buf_import: bool,
    size: RenderSize,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_initialized: Vec<bool>,
    swapchain_extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    frame_image: vk::Image,
    frame_memory: vk::DeviceMemory,
    frame_image_initialized: bool,
    shared_buffer: vk::Buffer,
    shared_memory: vk::DeviceMemory,
    hip_external_memory: HipExternalMemory,
    hip_mapped_ptr: *mut u32,
    shared_bytes: usize,
    input_shared_buffer: vk::Buffer,
    input_shared_memory: vk::DeviceMemory,
    input_hip_external_memory: HipExternalMemory,
    input_hip_mapped_ptr: *mut u32,
    input_shared_bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let sdl = sdl2::init().map_err(other_error)?;
    let mut presenter = VulkanPresenter::new(&sdl, args.size)?;
    let _ = presenter.window.set_opacity(0.94);
    let mut events = sdl.event_pump().map_err(other_error)?;

    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;
    let pixel_count = args.size.pixel_count();
    let device_input = DeviceBuffer::<u32>::new(pixel_count)?;
    let mut host_input = vec![0u32; pixel_count];
    fill_boot_pattern(&mut host_input, args.size);
    device_input.copy_from_host(&host_input)?;
    presenter.copy_device_input_to_shared(&device_input)?;

    let shared = Arc::new(Mutex::new(SharedCapture {
        pixels: host_input.clone(),
        sequence: 0,
        captures: 0,
        errors: 0,
        status: "video stream warming up".to_string(),
    }));
    let request = Arc::new(Mutex::new(CaptureRequest {
        x: 0,
        y: 0,
        width: args.size.width as u32,
        height: args.size.height as u32,
    }));
    let running = Arc::new(AtomicBool::new(true));
    let frozen = Arc::new(AtomicBool::new(false));
    let mut gpu_capture = if presenter.supports_dma_buf_import() {
        match GpuCaptureBackend::new() {
            Ok(backend) => {
                println!(
                    "Matrix Lens capture: wlroots dma-buf via {}",
                    backend.render_node()
                );
                Some(backend)
            }
            Err(err) => {
                eprintln!(
                    "Matrix Lens capture: dma-buf unavailable ({err}); falling back to video stream"
                );
                None
            }
        }
    } else {
        eprintln!(
            "Matrix Lens capture: Vulkan dma-buf import unsupported; falling back to video stream"
        );
        None
    };
    let mut capture_thread = if gpu_capture.is_some() {
        None
    } else {
        Some(spawn_video_capture_thread(
            args.size,
            Arc::clone(&shared),
            Arc::clone(&request),
            Arc::clone(&running),
            Arc::clone(&frozen),
        ))
    };

    let start = Instant::now();
    let mut last_fps = Instant::now();
    let mut frames_since_fps = 0u32;
    let mut rendered_frames = 0u32;
    let mut mode = args.mode;
    let mut last_sequence = u64::MAX;
    let mut last_capture_count = 0u64;
    let mut gpu_sequence = 0u64;
    let mut gpu_captures = 0u64;
    let mut gpu_errors = 0u64;
    let mut capture_status = "capture warming up".to_string();
    let mut copy_ms = 0.0f64;
    let mut present_ms = 0.0f64;
    let mut frame_budget = args.frames.map(|frames| frames.max(1));

    while frame_budget != Some(0) {
        let frame_start = Instant::now();
        for event in events.poll_iter() {
            match event {
                SdlEvent::Quit { .. } => frame_budget = Some(0),
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => frame_budget = Some(0),
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Num1 | Keycode::Kp1),
                    repeat: false,
                    ..
                } => mode = 0,
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Num2 | Keycode::Kp2),
                    repeat: false,
                    ..
                } => mode = 1,
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Num3 | Keycode::Kp3),
                    repeat: false,
                    ..
                } => mode = 2,
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Num4 | Keycode::Kp4),
                    repeat: false,
                    ..
                } => mode = 3,
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Right),
                    repeat: false,
                    ..
                } => mode = (mode + 1) % MODES.len(),
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Left),
                    repeat: false,
                    ..
                } => mode = (mode + MODES.len() - 1) % MODES.len(),
                SdlEvent::KeyDown {
                    keycode: Some(Keycode::Space | Keycode::F),
                    repeat: false,
                    ..
                } => {
                    let current = frozen.load(Ordering::Relaxed);
                    frozen.store(!current, Ordering::Relaxed);
                }
                _ => {}
            }
        }
        if frame_budget == Some(0) {
            break;
        }

        let current_request = update_capture_request(&presenter.window, args.size, &request);
        let mut switch_to_video_stream = false;
        let (sequence, captures, errors) = if let Some(gpu_capture) = gpu_capture.as_mut() {
            if !frozen.load(Ordering::Relaxed) {
                match gpu_capture.capture(current_request) {
                    Ok(frame) => {
                        capture_status = frame.status.clone();
                        copy_ms = presenter.copy_dma_capture_to_shared_input(frame)?;
                        gpu_sequence = gpu_sequence.wrapping_add(1);
                        gpu_captures = gpu_captures.wrapping_add(1);
                    }
                    Err(err) => {
                        gpu_errors = gpu_errors.wrapping_add(1);
                        capture_status = format!("gpu capture kept previous frame: {err}");
                        if gpu_errors >= 3 {
                            switch_to_video_stream = true;
                        }
                    }
                }
            }
            (gpu_sequence, gpu_captures, gpu_errors)
        } else {
            let shared = shared.lock().expect("capture mutex poisoned");
            if shared.sequence != last_sequence {
                host_input.copy_from_slice(&shared.pixels);
                last_sequence = shared.sequence;
                let upload_start = Instant::now();
                device_input.copy_from_host(&host_input)?;
                presenter.copy_device_input_to_shared(&device_input)?;
                copy_ms = upload_start.elapsed().as_secs_f64() * 1000.0;
            }
            capture_status.clone_from(&shared.status);
            (shared.sequence, shared.captures, shared.errors)
        };
        if switch_to_video_stream {
            eprintln!(
                "Matrix Lens capture: dma-buf capture failed repeatedly; falling back to video stream"
            );
            gpu_capture = None;
            capture_thread = Some(spawn_video_capture_thread(
                args.size,
                Arc::clone(&shared),
                Arc::clone(&request),
                Arc::clone(&running),
                Arc::clone(&frozen),
            ));
        }

        let frame_index = (start.elapsed().as_millis() / 16) as u32;
        unsafe {
            kernels.matrix_lens_fx_external(
                LaunchConfig::for_num_elems_with_block_size(pixel_count, 256),
                presenter.frame_hip_mapped_ptr(),
                pixel_count,
                presenter.input_hip_mapped_ptr(),
                pixel_count,
                args.size.width as u32,
                args.size.height as u32,
                pixel_count,
                frame_index,
                mode as u32,
            )?;
        }
        let frame_present_ms = presenter.present_shared_frame()?;
        present_ms = frame_present_ms;

        frames_since_fps = frames_since_fps.saturating_add(1);
        rendered_frames = rendered_frames.saturating_add(1);
        if last_fps.elapsed() >= Duration::from_millis(500) {
            let render_fps = frames_since_fps as f64 / last_fps.elapsed().as_secs_f64();
            let capture_fps = (captures.saturating_sub(last_capture_count)) as f64
                / last_fps.elapsed().as_secs_f64();
            frames_since_fps = 0;
            last_capture_count = captures;
            last_fps = Instant::now();
            presenter.window.set_title(&format!(
                "ROCm-Oxide Matrix Lens Vulkan | {} | render {:.1} capture {:.1} | input {:.2} present {:.2} | {} seq {} | errors {} | {}",
                MODES[mode],
                render_fps,
                capture_fps,
                copy_ms,
                present_ms,
                if frozen.load(Ordering::Relaxed) { "frozen" } else { "live" },
                sequence,
                errors,
                capture_status,
            ))?;
        }

        if let Some(frames) = frame_budget.as_mut() {
            *frames = frames.saturating_sub(1);
        }
        pace_frame(frame_start, args.fps_limit);
    }

    running.store(false, Ordering::Relaxed);
    if let Some(capture_thread) = capture_thread {
        if capture_thread.is_finished() {
            let _ = capture_thread.join();
        }
    }
    if args.frames.is_some() {
        let (final_captures, final_errors, final_status) = if gpu_capture.is_some() {
            (gpu_captures, gpu_errors, capture_status.clone())
        } else {
            let shared = shared.lock().expect("capture mutex poisoned");
            (shared.captures, shared.errors, shared.status.clone())
        };
        let device_frame = DeviceBuffer::<u32>::new(pixel_count)?;
        unsafe {
            device_frame.copy_from_device_ptr(presenter.frame_hip_mapped_ptr(), pixel_count)?;
        }
        let mut host_frame = vec![0u32; pixel_count];
        device_frame.copy_to_host(&mut host_frame)?;
        save_png(&args.output, &host_frame, args.size)?;
        println!(
            "Matrix Lens Vulkan summary: {:.1} FPS over {} rendered frame(s), captures {}, errors {}, last input {:.3} ms, present {:.3} ms, saved {}, status: {}",
            rendered_frames as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON),
            rendered_frames,
            final_captures,
            final_errors,
            copy_ms,
            present_ms,
            args.output.display(),
            final_status,
        );
    }
    Ok(())
}

impl GpuCaptureBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let render_node = std::env::var("ROCM_OXIDE_MATRIX_LENS_DRM_DEVICE")
            .unwrap_or_else(|_| DEFAULT_DRM_RENDER_NODE.to_string());
        let conn = wayland_client::Connection::connect_to_env()?;
        let wayshot = WayshotConnection::from_connection_with_dmabuf(conn, &render_node)?;
        Ok(Self {
            wayshot,
            render_node,
        })
    }

    fn render_node(&self) -> &str {
        &self.render_node
    }

    fn capture(
        &mut self,
        request: CaptureRequest,
    ) -> Result<GpuCaptureFrame, Box<dyn std::error::Error>> {
        self.wayshot.refresh_outputs()?;
        let viewport = LogicalRegion {
            inner: Region {
                position: Position {
                    x: request.x,
                    y: request.y,
                },
                size: WayshotSize {
                    width: request.width,
                    height: request.height,
                },
            },
        };
        let (output_index, embedded) = self
            .wayshot
            .get_all_outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| {
                EmbeddedRegion::new(viewport, output.logical_region).map(|embedded| {
                    let area = u64::from(embedded.inner.size.width)
                        * u64::from(embedded.inner.size.height);
                    (index, embedded, area)
                })
            })
            .max_by_key(|(_, _, area)| *area)
            .map(|(index, embedded, _)| (index, embedded))
            .ok_or_else(|| other_error("window is outside capturable Wayland outputs"))?;
        let output = &self.wayshot.get_all_outputs()[output_index];
        let (frame_format, _guard, bo) =
            self.wayshot
                .capture_output_frame_dmabuf(false, &output.wl_output, Some(embedded))?;
        let fd = bo.fd_for_plane(0)?;
        let width = bo.width();
        let height = bo.height();
        let stride = bo.stride_for_plane(0);
        let offset = bo.offset(0);
        let modifier = bo.modifier().into();
        let drm_format = frame_format.format;
        Ok(GpuCaptureFrame {
            fd,
            drm_format,
            width,
            height,
            stride,
            offset,
            modifier,
            status: format!(
                "gpu-dmabuf {} {}x{} stride {}",
                output.name, width, height, stride
            ),
        })
    }
}

fn spawn_video_capture_thread(
    size: RenderSize,
    shared: Arc<Mutex<SharedCapture>>,
    request: Arc<Mutex<CaptureRequest>>,
    running: Arc<AtomicBool>,
    frozen: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut local = vec![0u32; size.pixel_count()];
        let mut active_stream: Option<ActiveVideoStream> = None;
        let mut fallback_is_current = false;
        while running.load(Ordering::Relaxed) {
            if frozen.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(16));
                continue;
            }
            let request = *request.lock().expect("capture request mutex poisoned");
            let needs_new_stream = active_stream
                .as_ref()
                .is_none_or(|stream| !stream.covers(request));
            if needs_new_stream {
                if let Some(stream) = active_stream.take() {
                    let _ = stream.recorder.stop();
                }
                {
                    let mut shared = shared.lock().expect("capture mutex poisoned");
                    shared.status =
                        "video stream starting; select a screen if the desktop portal asks"
                            .to_string();
                }
                match ActiveVideoStream::new(request) {
                    Ok(stream) => {
                        fallback_is_current = false;
                        let mut shared = shared.lock().expect("capture mutex poisoned");
                        shared.status = format!(
                            "video stream {} {}x{} warming up",
                            stream.name, stream.key.width, stream.key.height
                        );
                        active_stream = Some(stream);
                    }
                    Err(err) => {
                        if !fallback_is_current {
                            fill_matrix_fallback(&mut local, size);
                            let mut shared = shared.lock().expect("capture mutex poisoned");
                            shared.pixels.copy_from_slice(&local);
                            shared.sequence = shared.sequence.wrapping_add(1);
                            shared.status = format!("video stream unavailable: {err}");
                            fallback_is_current = true;
                        } else {
                            let mut shared = shared.lock().expect("capture mutex poisoned");
                            shared.status = format!("video stream unavailable: {err}");
                        }
                        let mut shared = shared.lock().expect("capture mutex poisoned");
                        shared.errors = shared.errors.wrapping_add(1);
                        thread::sleep(Duration::from_millis(250));
                        continue;
                    }
                }
            }

            if active_stream.is_none() {
                thread::sleep(Duration::from_millis(16));
                continue;
            };
            let frame_result = active_stream
                .as_mut()
                .expect("active video stream checked above")
                .copy_latest_frame_to_pixels(request, size, &mut local);
            match frame_result {
                Ok(Some(status)) => {
                    let mut shared = shared.lock().expect("capture mutex poisoned");
                    shared.pixels.copy_from_slice(&local);
                    shared.sequence = shared.sequence.wrapping_add(1);
                    shared.captures = shared.captures.wrapping_add(1);
                    shared.status = status;
                    fallback_is_current = false;
                }
                Ok(None) => {
                    let stream = active_stream
                        .as_ref()
                        .expect("active video stream checked above");
                    let mut shared = shared.lock().expect("capture mutex poisoned");
                    shared.status = format!(
                        "video stream {} {}x{} waiting for frames",
                        stream.name, stream.key.width, stream.key.height
                    );
                }
                Err(err) => {
                    if let Some(stream) = active_stream.take() {
                        let _ = stream.recorder.stop();
                    }
                    fill_matrix_fallback(&mut local, size);
                    let mut shared = shared.lock().expect("capture mutex poisoned");
                    shared.pixels.copy_from_slice(&local);
                    shared.sequence = shared.sequence.wrapping_add(1);
                    shared.errors = shared.errors.wrapping_add(1);
                    shared.status = format!("video stream fallback: {err}");
                    fallback_is_current = true;
                    thread::sleep(Duration::from_millis(250));
                }
            }
            thread::sleep(Duration::from_millis(16));
        }
        if let Some(stream) = active_stream.take() {
            let _ = stream.recorder.stop();
        }
    })
}

impl ActiveVideoStream {
    fn new(request: CaptureRequest) -> Result<Self, Box<dyn std::error::Error>> {
        let monitor = monitor_for_request(request)?;
        let key = monitor_key(&monitor)?;
        let name = monitor.name().unwrap_or_else(|_| "monitor".to_string());
        let (recorder, receiver) = monitor.video_recorder()?;
        recorder.start()?;
        Ok(Self {
            key,
            name,
            recorder,
            receiver,
            latest_frame: None,
        })
    }

    fn covers(&self, request: CaptureRequest) -> bool {
        let center_x = request.x + (request.width as i32 / 2);
        let center_y = request.y + (request.height as i32 / 2);
        center_x >= self.key.x
            && center_y >= self.key.y
            && center_x < self.key.x + self.key.width as i32
            && center_y < self.key.y + self.key.height as i32
    }

    fn copy_latest_frame_to_pixels(
        &mut self,
        request: CaptureRequest,
        size: RenderSize,
        output: &mut [u32],
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if self.latest_frame.is_none() {
            match self.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(frame) => self.latest_frame = Some(frame),
                Err(RecvTimeoutError::Timeout) => return Ok(None),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(other_error("video stream frame channel disconnected"));
                }
            }
        }
        while let Ok(frame) = self.receiver.try_recv() {
            self.latest_frame = Some(frame);
        }
        let Some(frame) = self.latest_frame.as_ref() else {
            return Ok(None);
        };
        video_frame_to_pixels(frame, request, size, self.key, &self.name, output).map(Some)
    }
}

fn monitor_for_request(request: CaptureRequest) -> Result<Monitor, Box<dyn std::error::Error>> {
    let center_x = request.x + (request.width as i32 / 2);
    let center_y = request.y + (request.height as i32 / 2);
    match Monitor::from_point(center_x, center_y) {
        Ok(monitor) => Ok(monitor),
        Err(_) => Ok(Monitor::all()?
            .into_iter()
            .next()
            .ok_or_else(|| other_error("no capturable monitors found"))?),
    }
}

fn monitor_key(monitor: &Monitor) -> Result<MonitorKey, Box<dyn std::error::Error>> {
    Ok(MonitorKey {
        id: monitor.id().unwrap_or(0),
        x: monitor.x()?,
        y: monitor.y()?,
        width: monitor.width()?,
        height: monitor.height()?,
    })
}

fn video_frame_to_pixels(
    frame: &VideoFrame,
    request: CaptureRequest,
    size: RenderSize,
    monitor: MonitorKey,
    monitor_name: &str,
    output: &mut [u32],
) -> Result<String, Box<dyn std::error::Error>> {
    if output.len() != size.pixel_count() {
        return Err(other_error(format!(
            "video output has {} pixels, expected {}",
            output.len(),
            size.pixel_count()
        )));
    }
    let expected_bytes = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| other_error("video frame byte length overflows usize"))?;
    if frame.raw.len() < expected_bytes {
        return Err(other_error(format!(
            "video frame has {} bytes, expected at least {expected_bytes}",
            frame.raw.len()
        )));
    }

    let monitor_w = monitor.width as i32;
    let monitor_h = monitor.height as i32;
    let left = request.x.max(monitor.x);
    let top = request.y.max(monitor.y);
    let right = (request.x + request.width as i32).min(monitor.x + monitor_w);
    let bottom = (request.y + request.height as i32).min(monitor.y + monitor_h);
    if right <= left || bottom <= top {
        return Err(other_error("window is outside capturable monitor"));
    }

    output.fill(0);
    let frame_w = frame.width.max(1);
    let frame_h = frame.height.max(1);
    for y in 0..size.height {
        let screen_y = request.y + y as i32;
        if screen_y < top || screen_y >= bottom {
            continue;
        }
        let frame_y = (((screen_y - monitor.y) as u64) * u64::from(frame_h)
            / u64::from(monitor.height.max(1)))
        .min(u64::from(frame_h - 1)) as u32;
        for x in 0..size.width {
            let screen_x = request.x + x as i32;
            if screen_x < left || screen_x >= right {
                continue;
            }
            let frame_x = (((screen_x - monitor.x) as u64) * u64::from(frame_w)
                / u64::from(monitor.width.max(1)))
            .min(u64::from(frame_w - 1)) as u32;
            let source_index = ((frame_y as usize) * (frame.width as usize) + frame_x as usize)
                .checked_mul(4)
                .ok_or_else(|| other_error("video frame source index overflows usize"))?;
            let px = &frame.raw[source_index..source_index + 4];
            output[y * size.width + x] =
                ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32;
        }
    }
    Ok(format!(
        "video stream {monitor_name} {}x{} -> {}x{}",
        frame.width,
        frame.height,
        right - left,
        bottom - top
    ))
}

fn update_capture_request(
    window: &sdl2::video::Window,
    size: RenderSize,
    request: &Arc<Mutex<CaptureRequest>>,
) -> CaptureRequest {
    let (x, y) = window.position();
    let next = CaptureRequest {
        x,
        y,
        width: size.width as u32,
        height: size.height as u32,
    };
    let mut request = request.lock().expect("capture request mutex poisoned");
    *request = next;
    next
}

impl VulkanPresenter {
    fn new(sdl: &sdl2::Sdl, size: RenderSize) -> Result<Self, Box<dyn std::error::Error>> {
        let video = sdl.video().map_err(other_error)?;
        let window = video
            .window(
                "ROCm-Oxide Matrix Lens Vulkan",
                size.width as u32,
                size.height as u32,
            )
            .vulkan()
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
        let app_name = CString::new("rocm-oxide-matrix-lens")?;
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

        let (physical_device, queue_family_index, supports_dma_buf_import) =
            pick_vulkan_device(&instance, &surface_loader, surface)?;
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let mut device_extensions = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::external_memory::NAME.as_ptr(),
            ash::khr::external_memory_fd::NAME.as_ptr(),
        ];
        if supports_dma_buf_import {
            device_extensions.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
            device_extensions.push(ash::ext::image_drm_format_modifier::NAME.as_ptr());
        }
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
            supports_dma_buf_import,
            size,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: Vec::new(),
            swapchain_image_initialized: Vec::new(),
            swapchain_extent: vk::Extent2D::default(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            image_available: vk::Semaphore::null(),
            render_finished: vk::Semaphore::null(),
            in_flight: vk::Fence::null(),
            frame_image: vk::Image::null(),
            frame_memory: vk::DeviceMemory::null(),
            frame_image_initialized: false,
            shared_buffer: vk::Buffer::null(),
            shared_memory: vk::DeviceMemory::null(),
            hip_external_memory: ptr::null_mut(),
            hip_mapped_ptr: ptr::null_mut(),
            shared_bytes: 0,
            input_shared_buffer: vk::Buffer::null(),
            input_shared_memory: vk::DeviceMemory::null(),
            input_hip_external_memory: ptr::null_mut(),
            input_hip_mapped_ptr: ptr::null_mut(),
            input_shared_bytes: 0,
        };
        presenter.recreate_frame_resources(size)?;
        Ok(presenter)
    }

    fn recreate_frame_resources(
        &mut self,
        size: RenderSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.destroy_frame_resources();
        let byte_len = frame_byte_len(size)?;
        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?
        };
        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)?
        };
        let surface_format = choose_surface_format(&formats)?;
        let present_mode = choose_present_mode(&present_modes);
        let extent = choose_swapchain_extent(&self.window, surface_caps, size)?;
        let image_count = swapchain_image_count(surface_caps);
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
        let shared =
            self.create_shared_memory(byte_len, vk::BufferUsageFlags::TRANSFER_SRC, "lens output")?;
        let input_shared =
            self.create_shared_memory(byte_len, vk::BufferUsageFlags::TRANSFER_DST, "lens input")?;
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
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_initialized = vec![false; self.swapchain_images.len()];
        self.swapchain_extent = extent;
        self.command_pool = command_pool;
        self.command_buffer = command_buffer;
        self.image_available = image_available;
        self.render_finished = render_finished;
        self.in_flight = in_flight;
        self.frame_image = frame_image;
        self.frame_memory = frame_memory;
        self.frame_image_initialized = false;
        self.shared_buffer = shared.buffer;
        self.shared_memory = shared.memory;
        self.hip_external_memory = shared.hip_external_memory;
        self.hip_mapped_ptr = shared.hip_mapped_ptr;
        self.shared_bytes = shared.bytes;
        self.input_shared_buffer = input_shared.buffer;
        self.input_shared_memory = input_shared.memory;
        self.input_hip_external_memory = input_shared.hip_external_memory;
        self.input_hip_mapped_ptr = input_shared.hip_mapped_ptr;
        self.input_shared_bytes = input_shared.bytes;
        Ok(())
    }

    fn supports_dma_buf_import(&self) -> bool {
        self.supports_dma_buf_import
    }

    fn frame_hip_mapped_ptr(&self) -> *mut u32 {
        self.hip_mapped_ptr
    }

    fn input_hip_mapped_ptr(&self) -> *const u32 {
        self.input_hip_mapped_ptr.cast_const()
    }

    fn copy_device_input_to_shared(
        &mut self,
        source: &DeviceBuffer<u32>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if source.len() != self.size.pixel_count() {
            return Err(other_error(format!(
                "input frame has {} pixels, presenter expects {}",
                source.len(),
                self.size.pixel_count()
            )));
        }
        let source_bytes = source
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| other_error("input frame byte length overflows usize"))?;
        if self.input_hip_mapped_ptr.is_null() || source_bytes > self.input_shared_bytes {
            return Err(other_error(format!(
                "shared Vulkan/HIP input buffer is not ready or too small: source {source_bytes} bytes, shared {} bytes",
                self.input_shared_bytes
            )));
        }
        let copy_start = Instant::now();
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)?;
            source.copy_to_device_ptr(self.input_hip_mapped_ptr, source.len())?;
        }
        Stream::null().synchronize()?;
        Ok(copy_start.elapsed().as_secs_f64() * 1000.0)
    }

    fn present_shared_frame(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        let expected_bytes = frame_byte_len(self.size)?;
        if self.hip_mapped_ptr.is_null() || self.shared_bytes < expected_bytes {
            return Err(other_error(format!(
                "shared Vulkan/HIP output buffer is not ready or too small: expected {expected_bytes} bytes, shared {} bytes",
                self.shared_bytes
            )));
        }
        Stream::null().synchronize()?;
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
                self.recreate_frame_resources(self.size)?;
                return Ok(0.0);
            }
            Err(err) => return Err(other_error(format!("Vulkan acquire image failed: {err:?}"))),
        };

        unsafe {
            self.device.reset_fences(&[self.in_flight])?;
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        self.record_present_commands(image_index as usize)?;
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&self.image_available))
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&self.command_buffer))
            .signal_semaphores(std::slice::from_ref(&self.render_finished));
        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), self.in_flight)?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&self.render_finished))
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe { self.swapchain_loader.queue_present(self.queue, &present) };
        match present_result {
            Ok(present_suboptimal) => {
                if suboptimal || present_suboptimal {
                    self.recreate_frame_resources(self.size)?;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_frame_resources(self.size)?;
            }
            Err(err) => return Err(other_error(format!("Vulkan present failed: {err:?}"))),
        }
        Ok(present_start.elapsed().as_secs_f64() * 1000.0)
    }

    fn copy_dma_capture_to_shared_input(
        &mut self,
        frame: GpuCaptureFrame,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if !self.supports_dma_buf_import {
            return Err(other_error(
                "Vulkan device does not support dma-buf image import",
            ));
        }
        if frame.width != self.size.width as u32 || frame.height != self.size.height as u32 {
            return Err(other_error(format!(
                "gpu capture was clipped to {}x{}, expected {}x{}",
                frame.width, frame.height, self.size.width, self.size.height
            )));
        }
        let copy_start = Instant::now();
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)?;
        }
        let imported = self.import_dma_capture_image(frame)?;
        let copy_result = self.copy_imported_image_to_input(&imported);
        unsafe {
            self.device.destroy_image(imported.image, None);
            self.device.free_memory(imported.memory, None);
        }
        copy_result?;
        Ok(copy_start.elapsed().as_secs_f64() * 1000.0)
    }

    fn import_dma_capture_image(
        &self,
        frame: GpuCaptureFrame,
    ) -> Result<ImportedDmaImage, Box<dyn std::error::Error>> {
        let format = drm_format_to_vk(frame.drm_format)?;
        let byte_size = u64::from(frame.stride)
            .checked_mul(u64::from(frame.height))
            .ok_or_else(|| other_error("dma-buf byte size overflows u64"))?;
        let plane_layouts = [vk::SubresourceLayout::default()
            .offset(u64::from(frame.offset))
            .size(byte_size)
            .row_pitch(u64::from(frame.stride))
            .array_pitch(byte_size)
            .depth_pitch(byte_size)];
        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(frame.modifier)
            .plane_layouts(&plane_layouts);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: frame.width,
                height: frame.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_info)
            .push_next(&mut modifier_info);
        let image = unsafe { self.device.create_image(&image_info, None)? };
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let raw_fd = frame.fd.into_raw_fd();
        let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
        if let Err(err) = unsafe {
            self.external_memory_fd_loader.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                raw_fd,
                &mut fd_properties,
            )
        } {
            unsafe {
                drop(OwnedFd::from_raw_fd(raw_fd));
                self.device.destroy_image(image, None);
            }
            return Err(other_error(format!(
                "Vulkan dma-buf memory properties failed: {err:?}"
            )));
        }
        let memory_type_bits = requirements.memory_type_bits & fd_properties.memory_type_bits;
        let memory_type_index = find_memory_type(
            self.memory_properties,
            memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|_| {
            find_memory_type(
                self.memory_properties,
                memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })?;
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(raw_fd);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);
        let memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe {
                    drop(OwnedFd::from_raw_fd(raw_fd));
                    self.device.destroy_image(image, None);
                }
                return Err(other_error(format!(
                    "Vulkan dma-buf memory import failed: {err:?}"
                )));
            }
        };
        if let Err(err) = unsafe { self.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_image(image, None);
            }
            return Err(other_error(format!(
                "Vulkan dma-buf image bind failed: {err:?}"
            )));
        }
        Ok(ImportedDmaImage {
            image,
            memory,
            width: frame.width,
            height: frame.height,
        })
    }

    fn copy_imported_image_to_input(
        &mut self,
        imported: &ImportedDmaImage,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.reset_fences(&[self.in_flight])?;
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        self.image_barrier(
            imported.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );
        let copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(color_subresource_layers())
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: imported.width,
                height: imported.height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                imported.image,
                vk::ImageLayout::GENERAL,
                self.input_shared_buffer,
                std::slice::from_ref(&copy),
            );
            self.device.end_command_buffer(self.command_buffer)?;
        }
        let submit =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), self.in_flight)?;
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)?;
        }
        Ok(())
    }

    fn record_present_commands(
        &mut self,
        image_index: usize,
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
        let old_swapchain_layout = if self.swapchain_image_initialized[image_index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        self.image_barrier(
            swapchain_image,
            old_swapchain_layout,
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
        usage: vk::BufferUsageFlags,
        label: &str,
    ) -> Result<VulkanSharedMemory, Box<dyn std::error::Error>> {
        let mut external_buffer = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(bytes as vk::DeviceSize)
            .usage(usage)
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
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
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
        let fd = unsafe { self.external_memory_fd_loader.get_memory_fd(&fd_info) }?;
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
        println!("HIP/Vulkan {label} buffer: {bytes} bytes imported from OPAQUE_FD memory");
        Ok(VulkanSharedMemory {
            buffer,
            memory,
            hip_external_memory,
            hip_mapped_ptr,
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
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_image_memory(image, memory, 0)? };
        Ok((image, memory))
    }

    fn destroy_frame_resources(&mut self) {
        unsafe {
            if self.device.handle().is_null() {
                return;
            }
            let _ = self.device.device_wait_idle();
            if !self.hip_external_memory.is_null() {
                let _ = hipDestroyExternalMemory(self.hip_external_memory);
                self.hip_external_memory = ptr::null_mut();
                self.hip_mapped_ptr = ptr::null_mut();
            }
            if !self.input_hip_external_memory.is_null() {
                let _ = hipDestroyExternalMemory(self.input_hip_external_memory);
                self.input_hip_external_memory = ptr::null_mut();
                self.input_hip_mapped_ptr = ptr::null_mut();
            }
            if self.shared_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.shared_buffer, None);
                self.shared_buffer = vk::Buffer::null();
            }
            if self.shared_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.shared_memory, None);
                self.shared_memory = vk::DeviceMemory::null();
            }
            if self.input_shared_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.input_shared_buffer, None);
                self.input_shared_buffer = vk::Buffer::null();
            }
            if self.input_shared_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.input_shared_memory, None);
                self.input_shared_memory = vk::DeviceMemory::null();
            }
            if self.frame_image != vk::Image::null() {
                self.device.destroy_image(self.frame_image, None);
                self.frame_image = vk::Image::null();
            }
            if self.frame_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.frame_memory, None);
                self.frame_memory = vk::DeviceMemory::null();
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
            self.input_shared_bytes = 0;
        }
    }
}

impl Drop for VulkanPresenter {
    fn drop(&mut self) {
        self.destroy_frame_resources();
        unsafe {
            if self.device.handle() != vk::Device::null() {
                self.device.destroy_device(None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
            }
            if self.instance.handle() != vk::Instance::null() {
                self.instance.destroy_instance(None);
            }
        }
    }
}

fn pick_vulkan_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, bool), Box<dyn std::error::Error>> {
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
        let supports_dma_buf_import = vulkan_device_has_extension(
            instance,
            physical_device,
            ash::ext::external_memory_dma_buf::NAME,
        )? && vulkan_device_has_extension(
            instance,
            physical_device,
            ash::ext::image_drm_format_modifier::NAME,
        )?;
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in families.iter().enumerate() {
            let supports_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_present = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )?
            };
            let supports_external = supports_external_buffer(instance, physical_device);
            if supports_graphics && supports_present && supports_external {
                return Ok((physical_device, index as u32, supports_dma_buf_import));
            }
        }
    }
    Err(other_error(
        "no Vulkan device supports graphics, present, and external memory",
    ))
}

fn vulkan_device_has_extension(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension: &CStr,
) -> Result<bool, Box<dyn std::error::Error>> {
    let properties = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
    Ok(properties.iter().any(|property| {
        let name = unsafe { CStr::from_ptr(property.extension_name.as_ptr()) };
        name == extension
    }))
}

fn supports_external_buffer(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
    supports_external_buffer_usage(
        instance,
        physical_device,
        vk::BufferUsageFlags::TRANSFER_SRC,
    ) && supports_external_buffer_usage(
        instance,
        physical_device,
        vk::BufferUsageFlags::TRANSFER_DST,
    )
}

fn supports_external_buffer_usage(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    usage: vk::BufferUsageFlags,
) -> bool {
    let mut properties = vk::ExternalBufferProperties::default();
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
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

fn drm_format_to_vk(format: u32) -> Result<vk::Format, Box<dyn std::error::Error>> {
    match format {
        DRM_FORMAT_ARGB8888 | DRM_FORMAT_XRGB8888 => Ok(vk::Format::B8G8R8A8_UNORM),
        DRM_FORMAT_ABGR8888 | DRM_FORMAT_XBGR8888 => Err(other_error(
            "compositor returned ABGR/XBGR dma-buf; this demo needs ARGB/XRGB for direct kernel layout",
        )),
        _ => Err(other_error(format!(
            "unsupported compositor dma-buf fourcc `{}` ({format:#x})",
            fourcc_to_string(format)
        ))),
    }
}

fn fourcc_to_string(format: u32) -> String {
    let bytes = format.to_le_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> Result<vk::SurfaceFormatKHR, Box<dyn std::error::Error>> {
    if formats.is_empty() {
        return Err(other_error("Vulkan surface reported no formats"));
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
) -> Result<vk::Extent2D, Box<dyn std::error::Error>> {
    if surface_caps.current_extent.width != u32::MAX {
        return Ok(surface_caps.current_extent);
    }
    let (drawable_w, drawable_h) = window.vulkan_drawable_size();
    Ok(vk::Extent2D {
        width: drawable_w.max(size.width as u32).clamp(
            surface_caps.min_image_extent.width,
            surface_caps.max_image_extent.width,
        ),
        height: drawable_h.max(size.height as u32).clamp(
            surface_caps.min_image_extent.height,
            surface_caps.max_image_extent.height,
        ),
    })
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

fn swapchain_image_count(surface_caps: vk::SurfaceCapabilitiesKHR) -> u32 {
    let desired = surface_caps.min_image_count.saturating_add(1).max(2);
    if surface_caps.max_image_count == 0 {
        desired
    } else {
        desired.min(surface_caps.max_image_count)
    }
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
        rocm_oxide::hip::check(hipImportExternalMemory(&mut external_memory, &handle_desc))?;
    }
    let buffer_desc = HipExternalMemoryBufferDesc {
        offset: 0,
        size: buffer_size as u64,
        flags: 0,
        reserved: [0; 16],
    };
    let mut mapped = ptr::null_mut();
    unsafe {
        rocm_oxide::hip::check(hipExternalMemoryGetMappedBuffer(
            &mut mapped,
            external_memory,
            &buffer_desc,
        ))?;
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

fn frame_byte_len(size: RenderSize) -> Result<usize, Box<dyn std::error::Error>> {
    size.pixel_count()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| other_error("frame byte length overflows usize"))
}

fn fill_boot_pattern(frame: &mut [u32], size: RenderSize) {
    for y in 0..size.height {
        for x in 0..size.width {
            let g = ((x ^ y) & 255) as u32;
            frame[y * size.width + x] = (g << 8) | (g / 3);
        }
    }
}

fn fill_matrix_fallback(frame: &mut [u32], size: RenderSize) {
    for y in 0..size.height {
        for x in 0..size.width {
            let stripe = if ((x / 8 + y / 16) & 3) == 0 { 150 } else { 35 };
            frame[y * size.width + x] = (stripe << 8) | (stripe / 4);
        }
    }
}

fn save_png(
    path: &PathBuf,
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

fn pace_frame(frame_start: Instant, fps_limit: usize) {
    if fps_limit == 0 {
        return;
    }
    let target = Duration::from_secs_f64(1.0 / fps_limit as f64);
    if let Some(remaining) = target.checked_sub(frame_start.elapsed()) {
        thread::sleep(remaining);
    }
}

fn parse_args() -> Result<DemoArgs, Box<dyn std::error::Error>> {
    let mut frames = None;
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut mode = 0usize;
    let mut size = RESOLUTION_PRESETS[0].size;
    let mut fps_limit = DEFAULT_FPS_LIMIT;
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
                let value = args
                    .next()
                    .ok_or_else(|| "--mode requires a mode".to_string())?;
                mode = parse_mode(&value)?;
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
                    "Usage: cargo run --example matrix_lens -- [--frames N] [--mode matrix|glass|thermal|xray] [--resolution 540p|720p|1080p|WIDTHxHEIGHT] [--fps-limit FPS|uncapped]"
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
        if index < MODES.len() {
            return Ok(index);
        }
        if (1..=MODES.len()).contains(&index) {
            return Ok(index - 1);
        }
    }
    MODES
        .iter()
        .position(|mode| mode.eq_ignore_ascii_case(value))
        .ok_or_else(|| {
            format!("unknown mode `{value}`; expected matrix, glass, thermal, or xray").into()
        })
}

fn parse_resolution(value: &str) -> Result<RenderSize, Box<dyn std::error::Error>> {
    if let Some(preset) = RESOLUTION_PRESETS
        .iter()
        .find(|preset| preset.label.eq_ignore_ascii_case(value))
    {
        return Ok(preset.size);
    }
    let Some((width, height)) = value.split_once(['x', 'X']) else {
        return Err(format!("unknown resolution `{value}`").into());
    };
    let width = width.parse::<usize>()?;
    let height = height.parse::<usize>()?;
    if width < 320 || height < 180 || width > 3840 || height > 2160 {
        return Err(format!("resolution {width}x{height} is outside supported bounds").into());
    }
    width
        .checked_mul(height)
        .ok_or_else(|| format!("resolution {width}x{height} overflows pixel count"))?;
    Ok(RenderSize { width, height })
}

fn parse_fps_limit(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    if value.eq_ignore_ascii_case("uncapped") || value.eq_ignore_ascii_case("off") {
        return Ok(0);
    }
    Ok(value.parse::<usize>()?)
}

fn other_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::other(message.into()))
}
