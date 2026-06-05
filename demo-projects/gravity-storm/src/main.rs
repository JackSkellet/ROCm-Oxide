//! # Gravity Storm
//!
//! A live N-body gravitational particle simulation demonstrating ROCm-Oxide's
//! HIP compute ↔ Vulkan graphics pipeline with **true zero-copy** data sharing.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────────────┐
//!  │ Per frame                                                             │
//!  │                                                                       │
//!  │  1. HIPRTC physics kernel ──▶  Particle SSBO  (pinned host memory)   │
//!  │                                ↑ ONE allocation — zero copies        │
//!  │  2. hipDeviceSynchronize()  — HIP writes globally visible            │
//!  │                                                                       │
//!  │  3. Vulkan sync2 MEMORY_BARRIER  HOST_WRITE → SHADER_STORAGE_READ   │
//!  │  4. vkCmdDraw (point sprites, additive blend)                        │
//!  │     reads the SAME allocation via VK_EXT_external_memory_host buffer │
//!  └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Controls
//!
//! | Input         | Action                                   |
//! |---------------|------------------------------------------|
//! | Left click    | Place gravitational attractor            |
//! | Right click   | Place repulsor (negative gravity)        |
//! | Middle click  | Clear all attractors                     |
//! | Space         | Scatter all particles randomly           |
//! | R             | Reset to rotating ring formation         |
//! | + / =         | Add 4 096 particles                      |
//! | -             | Remove 4 096 particles                   |
//! | G             | Cycle gravity strength (1× / 4× / 16×)  |
//! | D             | Cycle damping (1.0 / 0.999 / 0.995 / 0.980)   |
//! | ESC / Q       | Exit                                     |

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::mem;
use std::time::Instant;

use ash::vk::Handle as _;
use ash::{ext, khr, vk};
use rocm_oxide::Device;
use rocm_oxide::hip::{
    DeviceBuffer, DevicePod, Function as HipFunction, Module as HipModule, PinnedHostBuffer, Stream,
};
use rocm_oxide::hiprtc;

// ─── HIP raw symbols ────────────────────────────────────────────────────────

unsafe extern "C" {
    fn hipDeviceSynchronize() -> i32;
}

// ─── App-level error type ────────────────────────────────────────────────────

type AppResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ─── Constants ──────────────────────────────────────────────────────────────

const DEFAULT_PARTICLES: usize = 32_768;
const MIN_PARTICLES: usize = 4_096;
const MAX_PARTICLES: usize = 931_072;
const PARTICLE_STEP: usize = 4_096;
const BLOCK_SIZE: u32 = 256;

const MAX_ATTRACTORS: usize = 16;
const ATTRACTOR_STRENGTH: f32 = 6_000.0;
const REPULSOR_STRENGTH: f32 = -3_000.0;
const SOFT_RADIUS: f32 = 70.0;
const MAX_EXPECTED_SPEED: f32 = 600.0;

const GRAVITY_SCALES: [f32; 3] = [1.0, 4.0, 16.0];
const DAMPINGS: [f32; 4] = [0.9999, 0.999, 0.995, 0.980];

const FRAMES_IN_FLIGHT: usize = 2;
const WINDOW_TITLE: &str = "ROCm-Oxide · Gravity Storm";
const WINDOW_W: u32 = 1920;
const WINDOW_H: u32 = 1080;

// ─── Particle layout ─────────────────────────────────────────────────────────
//
// 32 bytes / particle (8 × f32).  Indices: 0=x  1=y  2=vx  3=vy  4=speed_n
// Both the HIPRTC kernel and the GLSL SSBO use this stride explicitly.

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    speed_n: f32, // speed / MAX_EXPECTED_SPEED — written by kernel, read by VS
    _pad: [f32; 3],
}
const _: () = assert!(mem::size_of::<Particle>() == 32);
unsafe impl DevicePod for Particle {}

// ─── Attractor ───────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct Attractor {
    x: f32,
    y: f32,
    strength: f32,
    radius: f32,
}
unsafe impl DevicePod for Attractor {}

// ─── Push constants ───────────────────────────────────────────────────────────

#[repr(C)]
struct PushConstants {
    width: f32,
    height: f32,
    time: f32,
    _pad: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL shaders
// ─────────────────────────────────────────────────────────────────────────────

const VERT_GLSL: &str = r#"
#version 450

layout(std430, set = 0, binding = 0) readonly buffer Particles {
    float data[];
} pb;

layout(push_constant) uniform PC {
    float width;
    float height;
    float time;
    float _pad;
};

layout(location = 0) out float v_speed;

void main() {
    int   b  = gl_VertexIndex * 8;
    float px = pb.data[b + 0];
    float py = pb.data[b + 1];
    float sp = pb.data[b + 4];

    gl_Position  = vec4((px / width) * 2.0 - 1.0,
                        (py / height) * 2.0 - 1.0,
                        0.0, 1.0);
    gl_PointSize = 3.0;
    v_speed      = sp;
}
"#;

const FRAG_GLSL: &str = r#"
#version 450

layout(location = 0) in  float v_speed;
layout(location = 0) out vec4  out_color;

vec3 thermal(float t) {
    t = clamp(t, 0.0, 1.0);
    float s = t * 3.0;
    vec3 c0 = vec3(0.05, 0.10, 0.90);
    vec3 c1 = vec3(0.00, 0.95, 0.75);
    vec3 c2 = vec3(1.00, 0.65, 0.00);
    vec3 c3 = vec3(1.00, 0.05, 0.55);
    if (s < 1.0) return mix(c0, c1, s);
    if (s < 2.0) return mix(c1, c2, s - 1.0);
    return mix(c2, c3, s - 2.0);
}

void main() {
    float dist  = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (dist > 1.0) discard;
    float alpha = clamp((1.0 - dist) * (1.0 - dist) * 1.6, 0.0, 1.0);
    vec3  col   = thermal(v_speed);
    out_color   = vec4(col * alpha, alpha * 0.55);
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// HIPRTC physics kernel
// ─────────────────────────────────────────────────────────────────────────────

const KERNEL_SRC: &str = r#"
struct Attractor { float x, y, strength, radius; };

extern "C" __global__ void update_particles(
    float*           __restrict__ particles,
    const Attractor* __restrict__ attractors,
    int n_particles, int n_attractors,
    float dt, float damping,
    float bx, float by,
    float gravity_scale, float max_speed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    int b  = i * 8;
    float px = particles[b + 0], py = particles[b + 1];
    float vx = particles[b + 2], vy = particles[b + 3];

    float ax = 0.0f, ay = 0.0f;
    for (int j = 0; j < n_attractors; ++j) {
        float dx = attractors[j].x - px;
        float dy = attractors[j].y - py;
        float sr = attractors[j].radius;
        float r  = sqrtf(dx*dx + dy*dy + sr*sr);
        float f  = attractors[j].strength * gravity_scale / (r * r * r);
        ax += f * dx;
        ay += f * dy;
    }

    float m = 60.0f;
    if (px < m)        ax += (m - px)        * 0.05f;
    if (px > bx - m)   ax -= (px - (bx - m)) * 0.05f;
    if (py < m)        ay += (m - py)        * 0.05f;
    if (py > by - m)   ay -= (py - (by - m)) * 0.05f;

    vx = (vx + ax * dt) * damping;
    vy = (vy + ay * dt) * damping;
    px += vx * dt;
    py += vy * dt;

    particles[b + 0] = px;
    particles[b + 1] = py;
    particles[b + 2] = vx;
    particles[b + 3] = vy;
    particles[b + 4] = fminf(sqrtf(vx*vx + vy*vy) / max_speed, 1.0f);
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

fn compile_glsl_to_spirv(source: &str, stage: &str) -> Vec<u8> {
    use std::process::Command;
    let pid = std::process::id();
    let src = format!("/tmp/rocm_oxide_gs_{pid}_{stage}.glsl");
    let spv = format!("/tmp/rocm_oxide_gs_{pid}_{stage}.spv");
    std::fs::write(&src, source).expect("write GLSL");
    let stage_flag = format!("-fshader-stage={stage}");
    let ok = Command::new("glslc")
        .args([stage_flag.as_str(), &src, "-o", &spv])
        .status()
        .expect("glslc not found — install Vulkan SDK");
    assert!(ok.success(), "glslc failed for {stage}");
    let bytes = std::fs::read(&spv).expect("read SPIR-V");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&spv);
    bytes
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}

/// Reinterpret `&[u8]` (SPIR-V blob) as `&[u32]` for `vkCreateShaderModule`.
fn spv_bytes_to_words(bytes: &[u8]) -> &[u32] {
    assert!(bytes.len() % 4 == 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), bytes.len() / 4) }
}

fn scatter_random(particles: &mut [Particle], width: f32, height: f32) {
    // Deterministic LCG — no external crates needed.
    for (i, p) in particles.iter_mut().enumerate() {
        let mut s = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r1 = (s >> 33) as f32 / (u32::MAX as f32);
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r2 = (s >> 33) as f32 / (u32::MAX as f32);
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r3 = (s >> 33) as f32 / (u32::MAX as f32);
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r4 = (s >> 33) as f32 / (u32::MAX as f32);
        *p = Particle {
            x: 80.0 + r1 * (width - 160.0),
            y: 80.0 + r2 * (height - 160.0),
            vx: (r3 - 0.5) * 120.0,
            vy: (r4 - 0.5) * 120.0,
            speed_n: 0.0,
            _pad: [0.0; 3],
        };
    }
}

fn scatter_ring(particles: &mut [Particle], cx: f32, cy: f32, radius: f32) {
    let n = particles.len();
    for (i, p) in particles.iter_mut().enumerate() {
        let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
        let r = radius * (0.5 + 0.5 * ((i * 7 + 3) % 13) as f32 / 12.0);
        *p = Particle {
            x: cx + angle.cos() * r,
            y: cy + angle.sin() * r,
            vx: -angle.sin() * 150.0 * 0.8,
            vy: angle.cos() * 150.0 * 0.8,
            speed_n: 0.0,
            _pad: [0.0; 3],
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────────────────────

// Fields like `entry`, `hip_module` are retained for RAII drop ordering even
// when not read after construction.
#[allow(dead_code)]
struct GravityStorm {
    // SDL2
    sdl_ctx: sdl2::Sdl,
    window: sdl2::video::Window,
    extent: vk::Extent2D,

    // Vulkan core
    entry: ash::Entry,
    instance: ash::Instance,
    phys_dev: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    qfam: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,

    // WSI
    surface_ext: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_ext: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    sw_images: Vec<vk::Image>,
    sw_views: Vec<vk::ImageView>,
    sw_format: vk::Format,

    // Extension loaders
    dyn_render_ext: khr::dynamic_rendering::Device,
    sync2_ext: khr::synchronization2::Device,
    #[allow(dead_code)]
    ext_mem_host: ext::external_memory_host::Device,

    // Per-frame sync
    image_avail: [vk::Semaphore; FRAMES_IN_FLIGHT],
    render_done: [vk::Semaphore; FRAMES_IN_FLIGHT],
    in_flight: [vk::Fence; FRAMES_IN_FLIGHT],

    // Commands
    cmd_pool: vk::CommandPool,
    cmd_bufs: [vk::CommandBuffer; FRAMES_IN_FLIGHT],

    // Descriptors
    desc_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    desc_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],

    // Pipeline
    pipe_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    // Zero-copy particle buffer:
    // particle_host = hipHostMalloc allocation; Vulkan imports the same pointer
    // via VK_EXT_external_memory_host so both HIP and Vulkan share one page.
    particle_host: PinnedHostBuffer<Particle>,
    particle_mem: vk::DeviceMemory,
    particle_buf: vk::Buffer,

    // HIP compute
    hip_stream: Stream,
    hip_module: HipModule,
    hip_kernel: HipFunction,
    attractor_buf: DeviceBuffer<Attractor>,

    // Simulation state
    n_particles: usize,
    attractors: Vec<Attractor>,
    gravity_idx: usize,
    damping_idx: usize,
    start: Instant,
    frame_idx: usize,
}

impl GravityStorm {
    fn new() -> AppResult<Self> {
        // ── SDL2 ──────────────────────────────────────────────────────────
        let sdl = sdl2::init()?;
        let video = sdl.video()?;
        let window = video
            .window(WINDOW_TITLE, WINDOW_W, WINDOW_H)
            .vulkan()
            .resizable()
            .build()?;
        let extent = vk::Extent2D {
            width: WINDOW_W,
            height: WINDOW_H,
        };

        // ── Vulkan entry ──────────────────────────────────────────────────
        let entry = unsafe { ash::Entry::load()? };

        // ── Instance extensions (SDL2 + debug utils) ──────────────────────
        let sdl_exts = window.vulkan_instance_extensions()?;
        let mut ext_cstrings: Vec<std::ffi::CString> = sdl_exts
            .iter()
            .map(|s| std::ffi::CString::new(*s).unwrap())
            .collect();
        ext_cstrings.push(std::ffi::CString::new("VK_EXT_debug_utils").unwrap());
        let ext_ptrs: Vec<*const std::os::raw::c_char> =
            ext_cstrings.iter().map(|s| s.as_ptr()).collect();

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Gravity Storm")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"ROCm-Oxide")
            .api_version(vk::API_VERSION_1_3);

        let inst_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_ptrs);
        let instance = unsafe { entry.create_instance(&inst_ci, None)? };

        // ── Physical device ───────────────────────────────────────────────
        let phys_devs = unsafe { instance.enumerate_physical_devices()? };
        let phys_dev = *phys_devs.first().expect("no Vulkan GPU found");
        let mem_props = unsafe { instance.get_physical_device_memory_properties(phys_dev) };

        // ── Queue family (graphics + compute + transfer) ──────────────────
        let qfams = unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };
        let qfam = qfams
            .iter()
            .enumerate()
            .find(|(_, p)| {
                p.queue_flags.contains(
                    vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                )
            })
            .map(|(i, _)| i as u32)
            .expect("no suitable queue family");

        // ── Logical device ────────────────────────────────────────────────
        let q_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&[1.0f32]);

        let dev_exts: &[*const std::os::raw::c_char] = &[
            khr::swapchain::NAME.as_ptr(),
            khr::dynamic_rendering::NAME.as_ptr(),
            khr::synchronization2::NAME.as_ptr(),
            ext::external_memory_host::NAME.as_ptr(),
        ];

        let mut f_dyn =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut f_sync2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let dev_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&q_ci))
            .enabled_extension_names(dev_exts)
            .push_next(&mut f_dyn)
            .push_next(&mut f_sync2);

        let device = unsafe { instance.create_device(phys_dev, &dev_ci, None)? };
        let queue = unsafe { device.get_device_queue(qfam, 0) };

        // ── Extension loaders ─────────────────────────────────────────────
        let surface_ext = khr::surface::Instance::new(&entry, &instance);
        let swapchain_ext = khr::swapchain::Device::new(&instance, &device);
        let dyn_render_ext = khr::dynamic_rendering::Device::new(&instance, &device);
        let sync2_ext = khr::synchronization2::Device::new(&instance, &device);
        let ext_mem_host = ext::external_memory_host::Device::new(&instance, &device);

        // ── SDL2 Vulkan surface ───────────────────────────────────────────
        let surface = {
            let mut raw: u64 = 0;
            let ok = unsafe {
                sdl2::sys::SDL_Vulkan_CreateSurface(
                    window.raw(),
                    instance.handle().as_raw() as usize, // VkInstance = usize in sdl2-sys
                    &mut raw as *mut u64,                // VkSurfaceKHR = u64 in sdl2-sys
                ) as i32
            };
            assert_ne!(ok, 0, "SDL_Vulkan_CreateSurface failed");
            vk::SurfaceKHR::from_raw(raw)
        };

        // ── Swapchain ─────────────────────────────────────────────────────
        let (swapchain, sw_images, sw_views, sw_format) = build_swapchain(
            &phys_dev,
            &device,
            &surface_ext,
            &swapchain_ext,
            surface,
            extent,
        )?;

        // ── Command pool ──────────────────────────────────────────────────
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qfam)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None)? };

        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        let cbs = unsafe { device.allocate_command_buffers(&cb_alloc)? };
        let cmd_bufs = [cbs[0], cbs[1]];

        // ── Per-frame sync ────────────────────────────────────────────────
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let image_avail = [unsafe { device.create_semaphore(&sem_ci, None)? }, unsafe {
            device.create_semaphore(&sem_ci, None)?
        }];
        let render_done = [unsafe { device.create_semaphore(&sem_ci, None)? }, unsafe {
            device.create_semaphore(&sem_ci, None)?
        }];
        let in_flight = [unsafe { device.create_fence(&fence_ci, None)? }, unsafe {
            device.create_fence(&fence_ci, None)?
        }];

        // ── Descriptor set layout ─────────────────────────────────────────
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let dl_ci =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
        let desc_layout = unsafe { device.create_descriptor_set_layout(&dl_ci, None)? };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: FRAMES_IN_FLIGHT as u32,
        };
        let dp_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(FRAMES_IN_FLIGHT as u32);
        let desc_pool = unsafe { device.create_descriptor_pool(&dp_ci, None)? };

        let dl_arr = [desc_layout; FRAMES_IN_FLIGHT];
        let ds_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&dl_arr);
        let dsets = unsafe { device.allocate_descriptor_sets(&ds_alloc)? };
        let desc_sets = [dsets[0], dsets[1]];

        // ── Pipeline layout (descriptor + push constants) ─────────────────
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(mem::size_of::<PushConstants>() as u32);
        let pl_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_layout))
            .push_constant_ranges(std::slice::from_ref(&pc_range));
        let pipe_layout = unsafe { device.create_pipeline_layout(&pl_ci, None)? };

        // ── GLSL → SPIR-V → pipeline ──────────────────────────────────────
        let vert_spv = compile_glsl_to_spirv(VERT_GLSL, "vert");
        let frag_spv = compile_glsl_to_spirv(FRAG_GLSL, "frag");

        let vert_mod = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spv_bytes_to_words(&vert_spv)),
                None,
            )?
        };
        let frag_mod = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spv_bytes_to_words(&frag_spv)),
                None,
            )?
        };
        let pipeline = build_pipeline(&device, pipe_layout, sw_format, vert_mod, frag_mod)?;
        unsafe {
            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);
        }

        // ── Zero-copy particle buffer (VK_EXT_external_memory_host) ───────
        //
        // hipHostMalloc returns page-aligned memory (≥ 4096 B).
        // minImportedHostPointerAlignment on RADV is 4096 B, so this satisfies
        // the host pointer alignment requirement with no additional padding.
        let particle_host =
            PinnedHostBuffer::<Particle>::new_zeroed_mapped_coherent(MAX_PARTICLES)?;
        let host_ptr = particle_host.as_ptr() as *mut c_void;
        let buf_bytes = (MAX_PARTICLES * mem::size_of::<Particle>()) as u64;

        // Declare the handle type that Vulkan must accept for this buffer.
        let mut ext_buf_ci = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let buf_ci = vk::BufferCreateInfo::default()
            .size(buf_bytes)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_buf_ci);
        let particle_buf = unsafe { device.create_buffer(&buf_ci, None)? };
        let mem_reqs = unsafe { device.get_buffer_memory_requirements(particle_buf) };

        // Query which memory types Vulkan will accept for the HIP host pointer.
        // ash 0.38 exposes the raw fp only; call it directly.
        let mem_type_bits = unsafe {
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            let res = (ext_mem_host.fp().get_memory_host_pointer_properties_ext)(
                device.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                host_ptr,
                &mut props,
            );
            assert_eq!(
                res,
                vk::Result::SUCCESS,
                "vkGetMemoryHostPointerPropertiesEXT failed"
            );
            props.memory_type_bits
        };
        let mem_type = find_memory_type(
            &mem_props,
            mem_type_bits & mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .expect("no HOST_VISIBLE memory type for import");

        // Import the HIP host pointer as Vulkan device memory.
        let mut import_ci = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(host_ptr);
        let ma_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size.max(buf_bytes))
            .memory_type_index(mem_type)
            .push_next(&mut import_ci);
        let particle_mem = unsafe { device.allocate_memory(&ma_ci, None)? };
        unsafe { device.bind_buffer_memory(particle_buf, particle_mem, 0)? };

        // Point both descriptor sets at the one particle buffer.
        for &ds in &desc_sets {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(particle_buf)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        // ── HIP compute setup ─────────────────────────────────────────────
        let hip_stream = Stream::new()?;
        let arch = Device::first()?.arch().to_owned();
        eprintln!("[gravity_storm] Compiling HIPRTC kernel for {arch}…");
        let code_obj =
            hiprtc::compile_code_object(KERNEL_SRC, &arch).expect("HIPRTC compilation failed");
        let hip_module = HipModule::from_code_object(&code_obj)?;
        let hip_kernel = hip_module.function(c"update_particles")?;
        let attractor_buf = DeviceBuffer::<Attractor>::new(MAX_ATTRACTORS)?;

        // ── Initial particle state ────────────────────────────────────────
        let n_particles = DEFAULT_PARTICLES;
        // Safety: particle_host owns the allocation; we have exclusive access here.
        unsafe {
            scatter_random(
                std::slice::from_raw_parts_mut(particle_host.as_mut_ptr(), n_particles),
                WINDOW_W as f32,
                WINDOW_H as f32,
            );
        }

        let attractors = vec![Attractor {
            x: WINDOW_W as f32 / 2.0,
            y: WINDOW_H as f32 / 2.0,
            strength: ATTRACTOR_STRENGTH,
            radius: SOFT_RADIUS,
        }];

        Ok(Self {
            sdl_ctx: sdl,
            window,
            extent,
            entry,
            instance,
            phys_dev,
            device,
            queue,
            qfam,
            mem_props,
            surface_ext,
            surface,
            swapchain_ext,
            swapchain,
            sw_images,
            sw_views,
            sw_format,
            dyn_render_ext,
            sync2_ext,
            ext_mem_host,
            image_avail,
            render_done,
            in_flight,
            cmd_pool,
            cmd_bufs,
            desc_pool,
            desc_layout,
            desc_sets,
            pipe_layout,
            pipeline,
            particle_host,
            particle_mem,
            particle_buf,
            hip_stream,
            hip_module,
            hip_kernel,
            attractor_buf,
            n_particles,
            attractors,
            gravity_idx: 0,
            damping_idx: 0,
            start: Instant::now(),
            frame_idx: 0,
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // Main loop
    // ─────────────────────────────────────────────────────────────────────

    fn run(mut self) -> AppResult<()> {
        let mut events = self.sdl_ctx.event_pump()?;

        'main: loop {
            for event in events.poll_iter() {
                use sdl2::event::Event;
                use sdl2::keyboard::Keycode;
                use sdl2::mouse::MouseButton;

                match event {
                    Event::Quit { .. } => break 'main,

                    Event::KeyDown {
                        keycode: Some(kc), ..
                    } => match kc {
                        Keycode::Escape | Keycode::Q => break 'main,
                        Keycode::Space => self.scatter_all_random(),
                        Keycode::R => self.reset_ring(),

                        Keycode::Equals | Keycode::KpPlus | Keycode::RCTRL => {
                            if self.n_particles + PARTICLE_STEP <= MAX_PARTICLES {
                                let old = self.n_particles;
                                self.n_particles += PARTICLE_STEP;
                                // Safety: exclusive CPU write before any HIP launch.
                                unsafe {
                                    scatter_random(
                                        &mut std::slice::from_raw_parts_mut(
                                            self.particle_host.as_mut_ptr(),
                                            self.n_particles,
                                        )[old..],
                                        self.extent.width as f32,
                                        self.extent.height as f32,
                                    );
                                }
                                self.update_title();
                            }
                        }

                        Keycode::Minus | Keycode::KpMinus => {
                            if self.n_particles >= MIN_PARTICLES + PARTICLE_STEP {
                                self.n_particles -= PARTICLE_STEP;
                                self.update_title();
                            }
                        }

                        Keycode::G => {
                            self.gravity_idx = (self.gravity_idx + 1) % GRAVITY_SCALES.len();
                            self.update_title();
                        }
                        Keycode::D => {
                            self.damping_idx = (self.damping_idx + 1) % DAMPINGS.len();
                            self.update_title();
                        }
                        Keycode::C => self.attractors.clear(),
                        _ => {}
                    },

                    Event::MouseButtonDown {
                        mouse_btn, x, y, ..
                    } => match mouse_btn {
                        MouseButton::Left if self.attractors.len() < MAX_ATTRACTORS => {
                            self.attractors.push(Attractor {
                                x: x as f32,
                                y: y as f32,
                                strength: ATTRACTOR_STRENGTH,
                                radius: SOFT_RADIUS,
                            });
                        }
                        MouseButton::Right if self.attractors.len() < MAX_ATTRACTORS => {
                            self.attractors.push(Attractor {
                                x: x as f32,
                                y: y as f32,
                                strength: REPULSOR_STRENGTH,
                                radius: SOFT_RADIUS * 0.5,
                            });
                        }
                        MouseButton::Middle => self.attractors.clear(),
                        _ => {}
                    },

                    Event::Window {
                        win_event: sdl2::event::WindowEvent::Resized(w, h),
                        ..
                    } => {
                        unsafe { self.device.device_wait_idle().unwrap() };
                        self.extent = vk::Extent2D {
                            width: w as u32,
                            height: h as u32,
                        };
                        self.rebuild_swapchain()?;
                    }

                    _ => {}
                }
            }

            self.dispatch_physics()?;

            match self.render_frame() {
                Ok(()) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    unsafe { self.device.device_wait_idle().unwrap() };
                    self.rebuild_swapchain()?;
                }
                Err(e) => return Err(Box::new(e)),
            }

            self.frame_idx = self.frame_idx.wrapping_add(1);
        }

        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Simulation helpers
    // ─────────────────────────────────────────────────────────────────────

    fn update_title(&mut self) {
        let g = GRAVITY_SCALES[self.gravity_idx];
        let d = DAMPINGS[self.damping_idx];
        let _ = self.window.set_title(&format!(
            "ROCm-Oxide · Gravity Storm — {} particles  gravity={g:.0}×  damping={d:.3}",
            self.n_particles
        ));
    }

    fn scatter_all_random(&self) {
        // Safety: HIP kernel is not running during event processing.
        unsafe {
            scatter_random(
                std::slice::from_raw_parts_mut(self.particle_host.as_mut_ptr(), self.n_particles),
                self.extent.width as f32,
                self.extent.height as f32,
            );
        }
    }

    fn reset_ring(&self) {
        let cx = self.extent.width as f32 / 2.0;
        let cy = self.extent.height as f32 / 2.0;
        let r = f32::min(self.extent.width as f32, self.extent.height as f32) * 0.35;
        // Safety: HIP kernel is not running during event processing.
        unsafe {
            scatter_ring(
                std::slice::from_raw_parts_mut(self.particle_host.as_mut_ptr(), self.n_particles),
                cx,
                cy,
                r,
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // HIP physics dispatch
    // ─────────────────────────────────────────────────────────────────────

    fn dispatch_physics(&mut self) -> AppResult<()> {
        // Upload attractor list to GPU (at most 8 × 16 bytes = 128 bytes).
        let mut att_padded = self.attractors.clone();
        att_padded.resize(
            MAX_ATTRACTORS,
            Attractor {
                x: 0.0,
                y: 0.0,
                strength: 0.0,
                radius: 1.0,
            },
        );
        // Safety: att_padded lives for the duration of copy_from_host_async;
        //         we synchronize (hipDeviceSynchronize) before this stack frame returns.
        unsafe {
            self.attractor_buf
                .copy_from_host_async(&self.hip_stream, &att_padded)?;
        }

        let particles_dev = self.particle_host.device_ptr()?;
        let attractors_dev = self.attractor_buf.as_ptr();

        // Kernel parameters — must match the kernel signature order exactly.
        let mut p_part = particles_dev as *mut c_void;
        let mut p_attr = attractors_dev as *mut c_void;
        let mut p_n = self.n_particles as i32;
        let mut p_natt = self.attractors.len() as i32;
        let mut p_dt = 1.0f32 / 60.0;
        let mut p_damp = DAMPINGS[self.damping_idx];
        let mut p_bx = self.extent.width as f32;
        let mut p_by = self.extent.height as f32;
        let mut p_gscale = GRAVITY_SCALES[self.gravity_idx];
        let mut p_maxsp = MAX_EXPECTED_SPEED;

        let mut params: [*mut c_void; 10] = [
            &mut p_part as *mut _ as *mut c_void,
            &mut p_attr as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_natt as *mut _ as *mut c_void,
            &mut p_dt as *mut _ as *mut c_void,
            &mut p_damp as *mut _ as *mut c_void,
            &mut p_bx as *mut _ as *mut c_void,
            &mut p_by as *mut _ as *mut c_void,
            &mut p_gscale as *mut _ as *mut c_void,
            &mut p_maxsp as *mut _ as *mut c_void,
        ];

        let grid_x = (self.n_particles as u32).div_ceil(BLOCK_SIZE);
        unsafe {
            self.hip_kernel.launch_on_stream(
                (grid_x, 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                self.hip_stream.as_raw(),
                &mut params,
            )?;
            // After this call the HIP writes are in system memory and visible
            // to Vulkan via the shared host pointer.  The Vulkan barrier in
            // render_frame() then ensures the GPU re-reads from DRAM.
            hipDeviceSynchronize();
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Vulkan frame render
    // ─────────────────────────────────────────────────────────────────────

    fn render_frame(&mut self) -> Result<(), vk::Result> {
        let fi = self.frame_idx % FRAMES_IN_FLIGHT;

        let fences = [self.in_flight[fi]];
        unsafe { self.device.wait_for_fences(&fences, true, u64::MAX)? };
        unsafe { self.device.reset_fences(&fences)? };

        let (img_idx, _) = unsafe {
            self.swapchain_ext.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_avail[fi],
                vk::Fence::null(),
            )?
        };

        let cmd = self.cmd_bufs[fi];
        let sw_image = self.sw_images[img_idx as usize];
        let sw_view = self.sw_views[img_idx as usize];

        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // ── Barrier: host writes (HIP) → VS shader reads ────────────
            // After hipDeviceSynchronize() particle data is in DRAM.  This
            // barrier flushes/invalidates any GPU caches before the VS reads.
            let buf_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::HOST)
                .src_access_mask(vk::AccessFlags2::HOST_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .buffer(self.particle_buf)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.sync2_ext.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(std::slice::from_ref(&buf_barrier)),
            );

            // ── Transition: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL ────────
            let subres = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            self.sync2_ext.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .image(sw_image)
                        .subresource_range(subres),
                ]),
            );

            // ── Dynamic rendering ────────────────────────────────────────
            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.00, 0.00, 0.02, 1.0],
                },
            };
            let color_att = vk::RenderingAttachmentInfo::default()
                .image_view(sw_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear);

            self.dyn_render_ext.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_att)),
            );

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipe_layout,
                0,
                &[self.desc_sets[fi]],
                &[],
            );

            let t = self.start.elapsed().as_secs_f32();
            let pc = PushConstants {
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                time: t,
                _pad: 0.0,
            };
            self.device.cmd_push_constants(
                cmd,
                self.pipe_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    &pc as *const PushConstants as *const u8,
                    mem::size_of::<PushConstants>(),
                ),
            );

            self.device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.extent.width as f32,
                    height: self.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                }],
            );

            // One vertex per particle; the VS reads data via SSBO + gl_VertexIndex.
            self.device.cmd_draw(cmd, self.n_particles as u32, 1, 0, 0);
            self.dyn_render_ext.cmd_end_rendering(cmd);

            // ── Transition: COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR ──
            self.sync2_ext.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                        .dst_access_mask(vk::AccessFlags2::empty())
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(sw_image)
                        .subresource_range(subres),
                ]),
            );

            self.device.end_command_buffer(cmd)?;
        }

        // Submit
        let wait_sems = [self.image_avail[fi]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let sig_sems = [self.render_done[fi]];
        let cmds = [cmd];
        unsafe {
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait_sems)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&cmds)
                    .signal_semaphores(&sig_sems)],
                self.in_flight[fi],
            )?;
        }

        // Present
        unsafe {
            self.swapchain_ext.queue_present(
                self.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&sig_sems)
                    .swapchains(&[self.swapchain])
                    .image_indices(&[img_idx]),
            )?;
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Swapchain recreation on resize
    // ─────────────────────────────────────────────────────────────────────

    fn rebuild_swapchain(&mut self) -> AppResult<()> {
        for v in &self.sw_views {
            unsafe { self.device.destroy_image_view(*v, None) };
        }
        unsafe { self.swapchain_ext.destroy_swapchain(self.swapchain, None) };

        let (sw, imgs, views, fmt) = build_swapchain(
            &self.phys_dev,
            &self.device,
            &self.surface_ext,
            &self.swapchain_ext,
            self.surface,
            self.extent,
        )?;
        self.swapchain = sw;
        self.sw_images = imgs;
        self.sw_views = views;
        self.sw_format = fmt;
        Ok(())
    }
}

impl Drop for GravityStorm {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipe_layout, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_buffer(self.particle_buf, None);
            self.device.free_memory(self.particle_mem, None);
            for v in &self.sw_views {
                self.device.destroy_image_view(*v, None);
            }
            self.swapchain_ext.destroy_swapchain(self.swapchain, None);
            for &s in self.image_avail.iter().chain(self.render_done.iter()) {
                self.device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                self.device.destroy_fence(f, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.surface_ext.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan helper functions
// ─────────────────────────────────────────────────────────────────────────────

fn build_swapchain(
    phys_dev: &vk::PhysicalDevice,
    device: &ash::Device,
    surface_ext: &khr::surface::Instance,
    swapchain_ext: &khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
) -> AppResult<(
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Format,
)> {
    let caps = unsafe { surface_ext.get_physical_device_surface_capabilities(*phys_dev, surface)? };
    let formats = unsafe { surface_ext.get_physical_device_surface_formats(*phys_dev, surface)? };

    let fmt = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .unwrap_or(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });

    let image_count = {
        let n = caps.min_image_count + 1;
        if caps.max_image_count == 0 {
            n
        } else {
            n.min(caps.max_image_count)
        }
    };

    let sw_extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let sw_ci = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(fmt.format)
        .image_color_space(fmt.color_space)
        .image_extent(sw_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::MAILBOX)
        .clipped(true);

    let swapchain = unsafe { swapchain_ext.create_swapchain(&sw_ci, None)? };
    let images = unsafe { swapchain_ext.get_swapchain_images(swapchain)? };
    let views: Vec<vk::ImageView> = images
        .iter()
        .map(|&img| {
            let vi = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(fmt.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { device.create_image_view(&vi, None) }.unwrap()
        })
        .collect();

    Ok((swapchain, images, views, fmt.format))
}

fn build_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    vert_mod: vk::ShaderModule,
    frag_mod: vk::ShaderModule,
) -> AppResult<vk::Pipeline> {
    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
            .name(entry),
    ];

    let vi = vk::PipelineVertexInputStateCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::POINT_LIST);
    let vp_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Additive blend: final colour = src·srcAlpha + dst
    let blend_att = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE, // additive glow
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&blend_att));

    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

    // Dynamic rendering — no render pass object required.
    let color_fmts = [color_format];
    let mut dyn_render_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_fmts);

    let pipe_ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vi)
        .input_assembly_state(&ia)
        .viewport_state(&vp_state)
        .rasterization_state(&raster)
        .multisample_state(&ms)
        .color_blend_state(&blend)
        .dynamic_state(&dyn_state)
        .layout(layout)
        .push_next(&mut dyn_render_info);

    let pipes = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipe_ci], None)
            .map_err(|(_, e)| e)?
    };
    Ok(pipes[0])
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let app = GravityStorm::new().unwrap_or_else(|e| {
        eprintln!("Gravity Storm init error: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "[gravity_storm] {DEFAULT_PARTICLES} particles, zero-copy HIP→Vulkan via VK_EXT_external_memory_host\n\
         Controls:\n\
         \x20 Left click / Right click  — attractor / repulsor\n\
         \x20 Middle click / C          — clear attractors\n\
         \x20 Space                     — scatter randomly\n\
         \x20 R                         — ring formation\n\
         \x20 + / -                     — ±{PARTICLE_STEP} particles\n\
         \x20 G / D                     — cycle gravity / damping\n\
         \x20 ESC / Q                   — exit"
    );

    app.run().unwrap_or_else(|e| {
        eprintln!("Gravity Storm error: {e}");
        std::process::exit(1);
    });
}
