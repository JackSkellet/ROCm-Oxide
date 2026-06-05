#![allow(dead_code)]

use ash::vk::Handle;
use ash::{Entry, vk};
use rocm_oxide::DeviceBuffer;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_int, c_uint, c_void};
use std::os::fd::{FromRawFd, OwnedFd};
use std::ptr;

const VULKAN_WAIT_TIMEOUT_NS: u64 = 2_000_000_000;
const VULKAN_WAIT_TIMEOUT_MS: u64 = VULKAN_WAIT_TIMEOUT_NS / 1_000_000;
const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_int = 1;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PresentBackend {
    Minifb,
    Vulkan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Escape,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Space,
    LeftShift,
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    W,
    A,
    S,
    D,
    R,
    P,
    C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyRepeat {
    No,
    Yes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    Left,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseMode {
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    X1,
}

#[derive(Clone, Copy, Debug)]
pub struct WindowOptions {
    pub resize: bool,
    pub scale: Scale,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            resize: false,
            scale: Scale::X1,
        }
    }
}

pub struct Window {
    inner: WindowInner,
    host_frame: Vec<u32>,
    readback_frame: Vec<u32>,
}

enum WindowInner {
    Minifb(Box<minifb::Window>),
    Vulkan(Box<VulkanWindow>),
}

impl Window {
    pub fn new(
        title: &str,
        width: usize,
        height: usize,
        options: WindowOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match selected_backend() {
            PresentBackend::Minifb => {
                let mut window = minifb::Window::new(
                    title,
                    width,
                    height,
                    minifb::WindowOptions {
                        resize: options.resize,
                        scale: minifb_scale(options.scale),
                        ..minifb::WindowOptions::default()
                    },
                )?;
                window.set_title(&format!("{title} [minifb]"));
                Ok(Self {
                    inner: WindowInner::Minifb(Box::new(window)),
                    host_frame: Vec::new(),
                    readback_frame: Vec::new(),
                })
            }
            PresentBackend::Vulkan => Ok(Self {
                inner: WindowInner::Vulkan(Box::new(VulkanWindow::new(title, width, height)?)),
                host_frame: Vec::new(),
                readback_frame: Vec::new(),
            }),
        }
    }

    pub fn set_target_fps(&mut self, fps: usize) {
        if let WindowInner::Minifb(window) = &mut self.inner {
            window.set_target_fps(fps);
        }
    }

    pub fn is_open(&mut self) -> bool {
        match &mut self.inner {
            WindowInner::Minifb(window) => window.is_open(),
            WindowInner::Vulkan(window) => window.poll_events(),
        }
    }

    pub fn is_key_down(&self, key: Key) -> bool {
        match &self.inner {
            WindowInner::Minifb(window) => window.is_key_down(minifb_key(key)),
            WindowInner::Vulkan(window) => window.input.down.contains(&key),
        }
    }

    pub fn is_key_pressed(&self, key: Key, repeat: KeyRepeat) -> bool {
        self.get_keys_pressed(repeat).contains(&key)
    }

    pub fn get_keys_pressed(&self, repeat: KeyRepeat) -> Vec<Key> {
        match &self.inner {
            WindowInner::Minifb(window) => window
                .get_keys_pressed(minifb_repeat(repeat))
                .into_iter()
                .filter_map(demo_key_from_minifb)
                .collect(),
            WindowInner::Vulkan(window) => match repeat {
                KeyRepeat::No => window.input.pressed_no.clone(),
                KeyRepeat::Yes => window.input.pressed_yes.clone(),
            },
        }
    }

    pub fn get_mouse_down(&self, button: MouseButton) -> bool {
        match &self.inner {
            WindowInner::Minifb(window) => window.get_mouse_down(minifb_mouse_button(button)),
            WindowInner::Vulkan(window) => match button {
                MouseButton::Left => window.input.mouse_left_down,
            },
        }
    }

    pub fn get_mouse_pos(&self, _mode: MouseMode) -> Option<(f32, f32)> {
        self.framebuffer_mouse_pos()
    }

    pub fn get_unscaled_mouse_pos(&self, _mode: MouseMode) -> Option<(f32, f32)> {
        self.framebuffer_mouse_pos()
    }

    pub fn get_size(&self) -> (usize, usize) {
        match &self.inner {
            WindowInner::Minifb(window) => window.get_size(),
            WindowInner::Vulkan(window) => window.input.window_size,
        }
    }

    pub fn update_with_buffer(
        &mut self,
        buffer: &[u32],
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &mut self.inner {
            WindowInner::Minifb(window) => Ok(window.update_with_buffer(buffer, width, height)?),
            WindowInner::Vulkan(window) => window.present(buffer, width, height),
        }
    }

    pub fn update_with_frame<F>(
        &mut self,
        width: usize,
        height: usize,
        draw: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(&mut [u32]),
    {
        match &mut self.inner {
            WindowInner::Minifb(window) => {
                ensure_frame_len(&mut self.host_frame, width, height)?;
                draw(&mut self.host_frame);
                Ok(window.update_with_buffer(&self.host_frame, width, height)?)
            }
            WindowInner::Vulkan(window) => window.present_with_host_draw(width, height, draw),
        }
    }

    pub fn update_with_device_buffer(
        &mut self,
        source: &DeviceBuffer<u32>,
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &mut self.inner {
            WindowInner::Minifb(window) => {
                ensure_frame_len(&mut self.host_frame, width, height)?;
                source.copy_to_host(&mut self.host_frame)?;
                Ok(window.update_with_buffer(&self.host_frame, width, height)?)
            }
            WindowInner::Vulkan(window) => window.present_device_frame(source, width, height),
        }
    }

    pub fn update_with_device_buffer_and_regions<F>(
        &mut self,
        source: &DeviceBuffer<u32>,
        width: usize,
        height: usize,
        regions: &[CopyRegion],
        draw_overlay: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(&mut [u32]),
    {
        match &mut self.inner {
            WindowInner::Minifb(window) => {
                ensure_frame_len(&mut self.host_frame, width, height)?;
                self.host_frame.fill(0);
                draw_overlay(&mut self.host_frame);
                ensure_frame_len(&mut self.readback_frame, width, height)?;
                source.copy_to_host(&mut self.readback_frame)?;
                copy_regions(
                    &mut self.readback_frame,
                    &self.host_frame,
                    width,
                    height,
                    regions,
                )?;
                Ok(window.update_with_buffer(&self.readback_frame, width, height)?)
            }
            WindowInner::Vulkan(window) => window.present_device_frame_with_host_regions(
                source,
                width,
                height,
                regions,
                draw_overlay,
            ),
        }
    }

    pub fn set_title(&mut self, title: &str) {
        match &mut self.inner {
            WindowInner::Minifb(window) => window.set_title(title),
            WindowInner::Vulkan(window) => {
                let _ = window.window.set_title(title);
            }
        }
    }

    pub fn update(&mut self) {
        if let WindowInner::Minifb(window) = &mut self.inner {
            window.update();
        }
    }

    fn framebuffer_mouse_pos(&self) -> Option<(f32, f32)> {
        match &self.inner {
            WindowInner::Minifb(window) => {
                let (mx, my) = window.get_unscaled_mouse_pos(minifb::MouseMode::Discard)?;
                let (win_w, win_h) = window.get_size();
                if win_w == 0 || win_h == 0 {
                    return None;
                }
                Some((mx, my))
            }
            WindowInner::Vulkan(window) => {
                let (mx, my) = window.input.mouse_pos?;
                let (win_w, win_h) = window.input.window_size;
                if win_w == 0 || win_h == 0 {
                    return None;
                }
                let x = (mx.max(0) as f32 * window.width as f32 / win_w as f32)
                    .clamp(0.0, (window.width - 1) as f32);
                let y = (my.max(0) as f32 * window.height as f32 / win_h as f32)
                    .clamp(0.0, (window.height - 1) as f32);
                Some((x, y))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CopyRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl CopyRegion {
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    fn buffer_offset(
        self,
        frame_width: usize,
    ) -> Result<vk::DeviceSize, Box<dyn std::error::Error>> {
        let pixels = self
            .y
            .checked_mul(frame_width)
            .and_then(|row| row.checked_add(self.x))
            .ok_or_else(|| other_error("overlay region offset overflows usize"))?;
        pixels
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| vk::DeviceSize::try_from(bytes).ok())
            .ok_or_else(|| other_error("overlay region byte offset overflows VkDeviceSize"))
    }
}

pub fn requested_frames(env_key: &str) -> Option<u32> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--frames=") {
            return value.parse::<u32>().ok();
        }
        if arg == "--frames" {
            return args.next().and_then(|value| value.parse::<u32>().ok());
        }
    }
    std::env::var(env_key)
        .or_else(|_| std::env::var("ROCM_OXIDE_VISUAL_MAX_FRAMES"))
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
}

#[derive(Default)]
struct InputState {
    down: HashSet<Key>,
    pressed_no: Vec<Key>,
    pressed_yes: Vec<Key>,
    mouse_left_down: bool,
    mouse_pos: Option<(i32, i32)>,
    window_size: (usize, usize),
    open: bool,
}

struct VulkanWindow {
    _sdl: sdl2::Sdl,
    event_pump: sdl2::EventPump,
    window: sdl2::video::Window,
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
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_initialized: Vec<bool>,
    swapchain_extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u32,
    staging_bytes: usize,
    shared_buffer: vk::Buffer,
    shared_memory: vk::DeviceMemory,
    hip_external_memory: HipExternalMemory,
    hip_mapped_ptr: *mut u32,
    shared_bytes: usize,
    width: usize,
    height: usize,
    input: InputState,
}

struct VulkanSharedMemory {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    hip_external_memory: HipExternalMemory,
    hip_mapped_ptr: *mut u32,
    bytes: usize,
}

impl VulkanWindow {
    fn new(title: &str, width: usize, height: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let sdl = sdl2::init().map_err(other_error)?;
        let video = sdl.video().map_err(other_error)?;
        let window = video
            .window(
                &format!("{title} [vulkan]"),
                u32_from_usize(width, "window width")?,
                u32_from_usize(height, "window height")?,
            )
            .position_centered()
            .vulkan()
            .build()
            .map_err(|err| other_error(err.to_string()))?;
        let event_pump = sdl.event_pump().map_err(other_error)?;
        let entry = unsafe { Entry::load()? };
        let extension_names = window.vulkan_instance_extensions().map_err(other_error)?;
        let extension_cstrings = extension_names
            .iter()
            .map(|name| CString::new(*name))
            .collect::<Result<Vec<_>, _>>()?;
        let extension_ptrs = extension_cstrings
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let app_name = CString::new("rocm-oxide-visual-demo")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .engine_version(1)
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
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority);
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
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let mut this = Self {
            _sdl: sdl,
            event_pump,
            window,
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
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: Vec::new(),
            swapchain_image_initialized: Vec::new(),
            swapchain_extent: vk::Extent2D::default(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            image_available: vk::Semaphore::null(),
            render_finished: vk::Semaphore::null(),
            in_flight: vk::Fence::null(),
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            staging_ptr: std::ptr::null_mut(),
            staging_bytes: 0,
            shared_buffer: vk::Buffer::null(),
            shared_memory: vk::DeviceMemory::null(),
            hip_external_memory: ptr::null_mut(),
            hip_mapped_ptr: ptr::null_mut(),
            shared_bytes: 0,
            width,
            height,
            input: InputState {
                window_size: (width, height),
                open: true,
                ..InputState::default()
            },
        };
        this.create_resources()?;
        Ok(this)
    }

    fn poll_events(&mut self) -> bool {
        self.input.pressed_no.clear();
        self.input.pressed_yes.clear();
        for event in self.event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. } => self.input.open = false,
                sdl2::event::Event::KeyDown {
                    keycode: Some(key),
                    repeat,
                    ..
                } => {
                    if let Some(key) = demo_key_from_sdl(key) {
                        self.input.down.insert(key);
                        self.input.pressed_yes.push(key);
                        if !repeat {
                            self.input.pressed_no.push(key);
                        }
                    }
                }
                sdl2::event::Event::KeyUp {
                    keycode: Some(key), ..
                } => {
                    if let Some(key) = demo_key_from_sdl(key) {
                        self.input.down.remove(&key);
                    }
                }
                sdl2::event::Event::MouseButtonDown {
                    mouse_btn, x, y, ..
                } => {
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        self.input.mouse_left_down = true;
                    }
                    self.input.mouse_pos = Some((x, y));
                }
                sdl2::event::Event::MouseButtonUp {
                    mouse_btn, x, y, ..
                } => {
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        self.input.mouse_left_down = false;
                    }
                    self.input.mouse_pos = Some((x, y));
                }
                sdl2::event::Event::MouseMotion { x, y, .. } => {
                    self.input.mouse_pos = Some((x, y));
                }
                sdl2::event::Event::Window {
                    win_event:
                        sdl2::event::WindowEvent::SizeChanged(width, height)
                        | sdl2::event::WindowEvent::Resized(width, height),
                    ..
                } => {
                    self.input.window_size = (width.max(1) as usize, height.max(1) as usize);
                }
                _ => {}
            }
        }
        self.input.open && !self.input.down.contains(&Key::Escape)
    }

    fn present(
        &mut self,
        buffer: &[u32],
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_frame_shape(buffer.len(), width, height)?;
        let bytes = buffer
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| other_error("Vulkan frame byte length overflows usize"))?;
        if bytes > self.staging_bytes || self.staging_ptr.is_null() {
            return Err(other_error("Vulkan staging buffer is not ready"));
        }
        self.wait_for_in_flight("waiting for the previous Vulkan frame")?;
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.as_ptr(), self.staging_ptr, buffer.len());
        }
        self.present_from_buffers(self.staging_buffer, None)
    }

    fn present_with_host_draw<F>(
        &mut self,
        width: usize,
        height: usize,
        draw: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(&mut [u32]),
    {
        self.validate_frame_shape(frame_pixel_len(width, height)?, width, height)?;
        let bytes = frame_byte_len_usize(width, height)?;
        if bytes > self.staging_bytes || self.staging_ptr.is_null() {
            return Err(other_error("Vulkan staging buffer is not ready"));
        }
        self.wait_for_in_flight("waiting for the previous Vulkan frame")?;
        let frame =
            unsafe { std::slice::from_raw_parts_mut(self.staging_ptr, self.width * self.height) };
        draw(frame);
        self.present_from_buffers(self.staging_buffer, None)
    }

    fn present_device_frame(
        &mut self,
        source: &DeviceBuffer<u32>,
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_frame_shape(source.len(), width, height)?;
        let bytes = frame_byte_len_usize(width, height)?;
        if bytes > self.shared_bytes || self.hip_mapped_ptr.is_null() {
            return Err(other_error("Vulkan/HIP shared buffer is not ready"));
        }
        self.wait_for_in_flight("waiting for the previous Vulkan frame")?;
        unsafe {
            source.copy_to_device_ptr(self.hip_mapped_ptr, source.len())?;
        }
        self.present_from_buffers(self.shared_buffer, None)
    }

    fn present_device_frame_with_host_regions(
        &mut self,
        source: &DeviceBuffer<u32>,
        width: usize,
        height: usize,
        regions: &[CopyRegion],
        draw_overlay: impl FnOnce(&mut [u32]),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_frame_shape(source.len(), width, height)?;
        validate_regions(width, height, regions)?;
        let bytes = frame_byte_len_usize(width, height)?;
        if bytes > self.shared_bytes || self.hip_mapped_ptr.is_null() {
            return Err(other_error("Vulkan/HIP shared buffer is not ready"));
        }
        if bytes > self.staging_bytes || self.staging_ptr.is_null() {
            return Err(other_error("Vulkan overlay staging buffer is not ready"));
        }
        self.wait_for_in_flight("waiting for the previous Vulkan frame")?;
        unsafe {
            source.copy_to_device_ptr(self.hip_mapped_ptr, source.len())?;
        }
        let overlay =
            unsafe { std::slice::from_raw_parts_mut(self.staging_ptr, self.width * self.height) };
        clear_regions(overlay, self.width, self.height, regions)?;
        draw_overlay(overlay);
        self.present_from_buffers(self.shared_buffer, Some(regions))
    }

    fn present_from_buffers(
        &mut self,
        frame_buffer: vk::Buffer,
        overlay_regions: Option<&[CopyRegion]>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.reset_fences(&[self.in_flight])?;
        }
        let (image_index, suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                VULKAN_WAIT_TIMEOUT_NS,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::TIMEOUT) => {
                return Err(vulkan_wait_timeout_error(
                    "acquiring the next Vulkan swapchain image",
                ));
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_resources()?;
                return Ok(());
            }
            Err(err) => return Err(other_error(format!("Vulkan acquire image failed: {err:?}"))),
        };
        self.record_present_commands(image_index as usize, frame_buffer, overlay_regions)?;
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
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&self.render_finished))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(std::slice::from_ref(&image_index));
        match unsafe { self.swapchain_loader.queue_present(self.queue, &present) } {
            Ok(_) if suboptimal => self.recreate_resources()?,
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_resources()?;
            }
            Err(err) => return Err(other_error(format!("Vulkan present failed: {err:?}"))),
        }
        Ok(())
    }

    fn validate_frame_shape(
        &self,
        len: usize,
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if width != self.width || height != self.height {
            return Err(other_error(format!(
                "Vulkan presenter got {width}x{height} frame, expected {}x{}",
                self.width, self.height
            )));
        }
        validate_frame_shape("device", len, width, height)
    }

    fn create_resources(&mut self) -> Result<(), Box<dyn std::error::Error>> {
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
        self.swapchain_extent = choose_swapchain_extent(surface_caps, self.width, self.height);
        let image_count = swapchain_image_count(surface_caps);
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(self.swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(choose_composite_alpha(surface_caps))
            .present_mode(choose_present_mode(&present_modes))
            .clipped(true);
        self.swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_info, None)?
        };
        self.swapchain_images =
            unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain)? };
        self.swapchain_image_initialized = vec![false; self.swapchain_images.len()];

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.queue_family_index);
        self.command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None)? };
        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        self.command_buffer = unsafe { self.device.allocate_command_buffers(&command_info)?[0] };
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        self.image_available = unsafe { self.device.create_semaphore(&semaphore_info, None)? };
        self.render_finished = unsafe { self.device.create_semaphore(&semaphore_info, None)? };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        self.in_flight = unsafe { self.device.create_fence(&fence_info, None)? };
        self.create_staging_buffer()?;
        let shared = self.create_shared_memory(frame_byte_len_usize(self.width, self.height)?)?;
        self.shared_buffer = shared.buffer;
        self.shared_memory = shared.memory;
        self.hip_external_memory = shared.hip_external_memory;
        self.hip_mapped_ptr = shared.hip_mapped_ptr;
        self.shared_bytes = shared.bytes;
        Ok(())
    }

    fn recreate_resources(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.destroy_frame_resources()?;
        self.create_resources()
    }

    fn create_staging_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = self
            .width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<u32>()))
            .ok_or_else(|| other_error("Vulkan staging byte length overflows usize"))?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(bytes as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        self.staging_buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe {
            self.device
                .get_buffer_memory_requirements(self.staging_buffer)
        };
        let memory_type_index = find_memory_type(
            self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        self.staging_memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe {
            self.device
                .bind_buffer_memory(self.staging_buffer, self.staging_memory, 0)?;
            self.staging_ptr = self
                .device
                .map_memory(
                    self.staging_memory,
                    0,
                    bytes as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )?
                .cast::<u32>();
        }
        self.staging_bytes = bytes;
        Ok(())
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
        Ok(VulkanSharedMemory {
            buffer,
            memory,
            hip_external_memory,
            hip_mapped_ptr,
            bytes,
        })
    }

    fn record_present_commands(
        &mut self,
        image_index: usize,
        frame_buffer: vk::Buffer,
        overlay_regions: Option<&[CopyRegion]>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let swapchain_image = self.swapchain_images[image_index];
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        let old_layout = if self.swapchain_image_initialized[image_index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let src_stage = if self.swapchain_image_initialized[image_index] {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            vk::PipelineStageFlags::TOP_OF_PIPE
        };
        self.image_barrier(
            swapchain_image,
            old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(color_subresource_layers())
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: self.width as u32,
                height: self.height as u32,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                frame_buffer,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
        }
        if let Some(regions) = overlay_regions {
            for region in regions.iter().copied().filter(|region| !region.is_empty()) {
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(region.buffer_offset(self.width)?)
                    .buffer_row_length(u32_from_usize(self.width, "overlay row width")?)
                    .buffer_image_height(u32_from_usize(self.height, "overlay row height")?)
                    .image_subresource(color_subresource_layers())
                    .image_offset(vk::Offset3D {
                        x: i32_from_usize(region.x, "overlay x")?,
                        y: i32_from_usize(region.y, "overlay y")?,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: u32_from_usize(region.width, "overlay width")?,
                        height: u32_from_usize(region.height, "overlay height")?,
                        depth: 1,
                    });
                unsafe {
                    self.device.cmd_copy_buffer_to_image(
                        self.command_buffer,
                        self.staging_buffer,
                        swapchain_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(&copy),
                    );
                }
            }
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

    fn wait_for_in_flight(&self, action: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.in_flight == vk::Fence::null() {
            return Ok(());
        }
        match unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, VULKAN_WAIT_TIMEOUT_NS)
        } {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => Err(vulkan_wait_timeout_error(action)),
            Err(err) => Err(other_error(format!(
                "{action} failed while waiting for in-flight Vulkan work: {err:?}"
            ))),
        }
    }

    fn destroy_frame_resources(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.wait_for_in_flight("destroying Vulkan visual presenter resources")?;
        unsafe {
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
            if self.staging_memory != vk::DeviceMemory::null() && !self.staging_ptr.is_null() {
                self.device.unmap_memory(self.staging_memory);
                self.staging_ptr = std::ptr::null_mut();
            }
            if self.staging_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.staging_buffer, None);
                self.staging_buffer = vk::Buffer::null();
            }
            if self.staging_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.staging_memory, None);
                self.staging_memory = vk::DeviceMemory::null();
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
            self.staging_bytes = 0;
            self.shared_bytes = 0;
        }
        Ok(())
    }
}

impl Drop for VulkanWindow {
    fn drop(&mut self) {
        if let Err(err) = self.destroy_frame_resources() {
            eprintln!("skipping Vulkan visual resource cleanup after timeout/error: {err}");
            return;
        }
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn selected_backend() -> PresentBackend {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--present=") {
            return parse_backend(value);
        }
        if arg == "--present" {
            if let Some(value) = args.next() {
                return parse_backend(&value);
            }
        }
    }
    std::env::var("ROCM_OXIDE_VISUAL_PRESENT")
        .ok()
        .map(|value| parse_backend(&value))
        .unwrap_or(PresentBackend::Minifb)
}

fn parse_backend(value: &str) -> PresentBackend {
    match value.trim().to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => PresentBackend::Vulkan,
        _ => PresentBackend::Minifb,
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
        "no Vulkan device supports graphics, presentation, and exportable OPAQUE_FD transfer buffers",
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
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

fn choose_swapchain_extent(
    surface_caps: vk::SurfaceCapabilitiesKHR,
    width: usize,
    height: usize,
) -> vk::Extent2D {
    if surface_caps.current_extent.width != u32::MAX {
        return surface_caps.current_extent;
    }
    vk::Extent2D {
        width: (width as u32).clamp(
            surface_caps.min_image_extent.width,
            surface_caps.max_image_extent.width,
        ),
        height: (height as u32).clamp(
            surface_caps.min_image_extent.height,
            surface_caps.max_image_extent.height,
        ),
    }
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
        rocm_oxide::hip::check(hipImportExternalMemory(&mut external_memory, &handle_desc))
            .map_err(|err| {
                other_error(format!("hipImportExternalMemory(Vulkan OPAQUE_FD): {err}"))
            })?;
    }
    let buffer_desc = HipExternalMemoryBufferDesc {
        offset: 0,
        size: buffer_size as u64,
        flags: 0,
        reserved: [0; 16],
    };
    let mut mapped = ptr::null_mut();
    let mapped_result = unsafe {
        rocm_oxide::hip::check(hipExternalMemoryGetMappedBuffer(
            &mut mapped,
            external_memory,
            &buffer_desc,
        ))
        .map_err(|err| {
            other_error(format!(
                "hipExternalMemoryGetMappedBuffer(Vulkan shared buffer): {err}"
            ))
        })
    };
    if let Err(err) = mapped_result {
        unsafe {
            let _ = hipDestroyExternalMemory(external_memory);
        }
        return Err(err);
    }
    Ok((external_memory, mapped.cast::<u32>()))
}

fn ensure_frame_len(
    frame: &mut Vec<u32>,
    width: usize,
    height: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let len = frame_pixel_len(width, height)?;
    if frame.len() != len {
        frame.resize(len, 0);
    }
    Ok(())
}

fn validate_frame_shape(
    label: &str,
    len: usize,
    width: usize,
    height: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = frame_pixel_len(width, height)?;
    if len != expected {
        return Err(other_error(format!(
            "{label} frame has {len} pixels, expected {expected} for {width}x{height}"
        )));
    }
    Ok(())
}

fn frame_pixel_len(width: usize, height: usize) -> Result<usize, Box<dyn std::error::Error>> {
    width
        .checked_mul(height)
        .ok_or_else(|| other_error("frame pixel count overflows usize"))
}

fn frame_byte_len_usize(width: usize, height: usize) -> Result<usize, Box<dyn std::error::Error>> {
    frame_pixel_len(width, height)?
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| other_error("frame byte length overflows usize"))
}

fn validate_regions(
    width: usize,
    height: usize,
    regions: &[CopyRegion],
) -> Result<(), Box<dyn std::error::Error>> {
    for region in regions {
        let right = region
            .x
            .checked_add(region.width)
            .ok_or_else(|| other_error("overlay region width overflows usize"))?;
        let bottom = region
            .y
            .checked_add(region.height)
            .ok_or_else(|| other_error("overlay region height overflows usize"))?;
        if right > width || bottom > height {
            return Err(other_error(format!(
                "overlay region {:?} exceeds frame bounds {width}x{height}",
                region
            )));
        }
    }
    Ok(())
}

fn copy_regions(
    dst: &mut [u32],
    src: &[u32],
    width: usize,
    height: usize,
    regions: &[CopyRegion],
) -> Result<(), Box<dyn std::error::Error>> {
    validate_frame_shape("destination", dst.len(), width, height)?;
    validate_frame_shape("source", src.len(), width, height)?;
    validate_regions(width, height, regions)?;
    for region in regions.iter().copied().filter(|region| !region.is_empty()) {
        for row in region.y..region.y + region.height {
            let start = row
                .checked_mul(width)
                .and_then(|base| base.checked_add(region.x))
                .ok_or_else(|| other_error("overlay copy row offset overflows usize"))?;
            let end = start + region.width;
            dst[start..end].copy_from_slice(&src[start..end]);
        }
    }
    Ok(())
}

fn clear_regions(
    frame: &mut [u32],
    width: usize,
    height: usize,
    regions: &[CopyRegion],
) -> Result<(), Box<dyn std::error::Error>> {
    validate_frame_shape("frame", frame.len(), width, height)?;
    validate_regions(width, height, regions)?;
    for region in regions.iter().copied().filter(|region| !region.is_empty()) {
        for row in region.y..region.y + region.height {
            let start = row
                .checked_mul(width)
                .and_then(|base| base.checked_add(region.x))
                .ok_or_else(|| other_error("overlay clear row offset overflows usize"))?;
            let end = start + region.width;
            frame[start..end].fill(0);
        }
    }
    Ok(())
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

fn minifb_scale(scale: Scale) -> minifb::Scale {
    match scale {
        Scale::X1 => minifb::Scale::X1,
    }
}

fn minifb_repeat(repeat: KeyRepeat) -> minifb::KeyRepeat {
    match repeat {
        KeyRepeat::No => minifb::KeyRepeat::No,
        KeyRepeat::Yes => minifb::KeyRepeat::Yes,
    }
}

fn minifb_mouse_button(button: MouseButton) -> minifb::MouseButton {
    match button {
        MouseButton::Left => minifb::MouseButton::Left,
    }
}

fn minifb_key(key: Key) -> minifb::Key {
    match key {
        Key::Escape => minifb::Key::Escape,
        Key::Left => minifb::Key::Left,
        Key::Right => minifb::Key::Right,
        Key::Up => minifb::Key::Up,
        Key::Down => minifb::Key::Down,
        Key::PageUp => minifb::Key::PageUp,
        Key::PageDown => minifb::Key::PageDown,
        Key::Space => minifb::Key::Space,
        Key::LeftShift => minifb::Key::LeftShift,
        Key::Key0 => minifb::Key::Key0,
        Key::Key1 => minifb::Key::Key1,
        Key::Key2 => minifb::Key::Key2,
        Key::Key3 => minifb::Key::Key3,
        Key::Key4 => minifb::Key::Key4,
        Key::Key5 => minifb::Key::Key5,
        Key::Key6 => minifb::Key::Key6,
        Key::Key7 => minifb::Key::Key7,
        Key::Key8 => minifb::Key::Key8,
        Key::Key9 => minifb::Key::Key9,
        Key::W => minifb::Key::W,
        Key::A => minifb::Key::A,
        Key::S => minifb::Key::S,
        Key::D => minifb::Key::D,
        Key::R => minifb::Key::R,
        Key::P => minifb::Key::P,
        Key::C => minifb::Key::C,
    }
}

fn demo_key_from_minifb(key: minifb::Key) -> Option<Key> {
    Some(match key {
        minifb::Key::Escape => Key::Escape,
        minifb::Key::Left => Key::Left,
        minifb::Key::Right => Key::Right,
        minifb::Key::Up => Key::Up,
        minifb::Key::Down => Key::Down,
        minifb::Key::PageUp => Key::PageUp,
        minifb::Key::PageDown => Key::PageDown,
        minifb::Key::Space => Key::Space,
        minifb::Key::LeftShift => Key::LeftShift,
        minifb::Key::Key0 => Key::Key0,
        minifb::Key::Key1 => Key::Key1,
        minifb::Key::Key2 => Key::Key2,
        minifb::Key::Key3 => Key::Key3,
        minifb::Key::Key4 => Key::Key4,
        minifb::Key::Key5 => Key::Key5,
        minifb::Key::Key6 => Key::Key6,
        minifb::Key::Key7 => Key::Key7,
        minifb::Key::Key8 => Key::Key8,
        minifb::Key::Key9 => Key::Key9,
        minifb::Key::W => Key::W,
        minifb::Key::A => Key::A,
        minifb::Key::S => Key::S,
        minifb::Key::D => Key::D,
        minifb::Key::R => Key::R,
        minifb::Key::P => Key::P,
        minifb::Key::C => Key::C,
        _ => return None,
    })
}

fn demo_key_from_sdl(key: sdl2::keyboard::Keycode) -> Option<Key> {
    Some(match key {
        sdl2::keyboard::Keycode::Escape => Key::Escape,
        sdl2::keyboard::Keycode::Left => Key::Left,
        sdl2::keyboard::Keycode::Right => Key::Right,
        sdl2::keyboard::Keycode::Up => Key::Up,
        sdl2::keyboard::Keycode::Down => Key::Down,
        sdl2::keyboard::Keycode::PageUp => Key::PageUp,
        sdl2::keyboard::Keycode::PageDown => Key::PageDown,
        sdl2::keyboard::Keycode::Space => Key::Space,
        sdl2::keyboard::Keycode::LShift => Key::LeftShift,
        sdl2::keyboard::Keycode::Num0 | sdl2::keyboard::Keycode::Kp0 => Key::Key0,
        sdl2::keyboard::Keycode::Num1 | sdl2::keyboard::Keycode::Kp1 => Key::Key1,
        sdl2::keyboard::Keycode::Num2 | sdl2::keyboard::Keycode::Kp2 => Key::Key2,
        sdl2::keyboard::Keycode::Num3 | sdl2::keyboard::Keycode::Kp3 => Key::Key3,
        sdl2::keyboard::Keycode::Num4 | sdl2::keyboard::Keycode::Kp4 => Key::Key4,
        sdl2::keyboard::Keycode::Num5 | sdl2::keyboard::Keycode::Kp5 => Key::Key5,
        sdl2::keyboard::Keycode::Num6 | sdl2::keyboard::Keycode::Kp6 => Key::Key6,
        sdl2::keyboard::Keycode::Num7 | sdl2::keyboard::Keycode::Kp7 => Key::Key7,
        sdl2::keyboard::Keycode::Num8 | sdl2::keyboard::Keycode::Kp8 => Key::Key8,
        sdl2::keyboard::Keycode::Num9 | sdl2::keyboard::Keycode::Kp9 => Key::Key9,
        sdl2::keyboard::Keycode::W => Key::W,
        sdl2::keyboard::Keycode::A => Key::A,
        sdl2::keyboard::Keycode::S => Key::S,
        sdl2::keyboard::Keycode::D => Key::D,
        sdl2::keyboard::Keycode::R => Key::R,
        sdl2::keyboard::Keycode::P => Key::P,
        sdl2::keyboard::Keycode::C => Key::C,
        _ => return None,
    })
}

fn u32_from_usize(value: usize, label: &str) -> Result<u32, Box<dyn std::error::Error>> {
    u32::try_from(value).map_err(|_| other_error(format!("{label} exceeds u32")))
}

fn i32_from_usize(value: usize, label: &str) -> Result<i32, Box<dyn std::error::Error>> {
    i32::try_from(value).map_err(|_| other_error(format!("{label} exceeds i32")))
}

fn vulkan_wait_timeout_error(action: &str) -> Box<dyn std::error::Error> {
    other_error(format!(
        "{action} timed out after {VULKAN_WAIT_TIMEOUT_MS} ms"
    ))
}

fn other_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::other(message.into()))
}
