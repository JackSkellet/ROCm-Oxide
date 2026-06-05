use font8x8::{BASIC_FONTS, UnicodeFonts};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig, Module};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod visual_presenter;
use visual_presenter::{
    CopyRegion, Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions,
    requested_frames,
};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1024;
const HEIGHT: usize = 576;
const CAMERA_PARAMS: usize = 16;
const MAX_STRESS_WORK_ITERS: u32 = 4_096;
const MODES: [&str; 4] = [
    "Rust rainbow kernel",
    "2D compute stress",
    "3D volume stress",
    "Raytraced world",
];

const PARTY_POST: &str = r#"
extern "C" {
__device__ float party_intensity = 0.0f;
}

extern "C" __global__
void party_post(unsigned int* frame, unsigned long long n, unsigned int frame_index, unsigned int mode) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int c = frame[i];
    unsigned int r = (c >> 16) & 255u;
    unsigned int g = (c >> 8) & 255u;
    unsigned int b = c & 255u;
    unsigned int x = i & 1023u;
    unsigned int y = i / 1024u;
    unsigned int wave = ((x * 2u + y * 3u + frame_index * 5u + mode * 41u) & 255u);
    unsigned int band = ((x / 32u + y / 24u + frame_index / 8u) & 1u) * 36u;
    float k = party_intensity;

    unsigned int tr = (wave + 80u + band) & 255u;
    unsigned int tg = ((255u - wave) + 48u) & 255u;
    unsigned int tb = ((wave >> 1u) + 160u + band) & 255u;
    unsigned int nr = (unsigned int)((float)r * (1.0f - k) + (float)tr * k);
    unsigned int ng = (unsigned int)((float)g * (1.0f - k) + (float)tg * k);
    unsigned int nb = (unsigned int)((float)b * (1.0f - k) + (float)tb * k);
    frame[i] = ((nr & 255u) << 16) | ((ng & 255u) << 8) | (nb & 255u);
}
"#;

#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len <= f32::EPSILON {
            self
        } else {
            self.scale(1.0 / len)
        }
    }

    fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _single_instance = InstanceGuard::acquire()?;
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let party_module = device.compile_hip_source(PARTY_POST)?;
    let party_kernel = party_module.kernel(c"party_post")?;
    let party_intensity = party_module.global::<f32>(c"party_intensity")?;

    let pixel_count = WIDTH * HEIGHT;
    let block_x = 256u32;
    let device_frame = DeviceBuffer::<u32>::new(pixel_count)?;
    let device_camera = DeviceBuffer::<f32>::new(CAMERA_PARAMS)?;
    let mut camera = vec![0.0f32; CAMERA_PARAMS];

    let d_affine_input = DeviceBuffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0])?;
    let d_affine_output = DeviceBuffer::<f32>::new(4)?;
    let d_affine_params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 2.5,
        bias: 10.0,
    }])?;
    let short = DeviceBuffer::<u32>::new(pixel_count / 2)?;
    let mut safety_status = "C: safety demo: reject a half-size frame before launch".to_string();

    let mut window = Window::new(
        "ROCm-Oxide Possibilities Window",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
        },
    )?;

    let max_frames = requested_frames("ROCM_OXIDE_WINDOW_MAX_FRAMES");
    let mut rendered_frames = 0u32;
    let mut frames = 0u32;
    let mut mode = 0usize;
    let mut work_iters = 96u32;
    let mut paused = false;
    let mut party = false;
    let mut shadows = true;
    let mut reflections = true;
    let mut fps = 0.0f64;
    let mut mouse_was_down = false;
    let mut frame_index = 0u32;
    let mut last_title = Instant::now();
    let start = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if mouse_down
            && !mouse_was_down
            && let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard)
            && let Some(clicked_mode) = mode_from_mouse(mx as usize, my as usize)
        {
            mode = clicked_mode;
        }
        mouse_was_down = mouse_down;

        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Key1 => mode = 0,
                Key::Key2 => mode = 1,
                Key::Key3 => mode = 2,
                Key::Key4 => mode = 3,
                Key::Right => mode = (mode + 1) % MODES.len(),
                Key::Left => mode = (mode + MODES.len() - 1) % MODES.len(),
                Key::Up => work_iters = clamp_work_iters(work_iters.saturating_add(32)),
                Key::Down => work_iters = work_iters.saturating_sub(32).max(1),
                Key::P => party = !party,
                Key::Space => paused = !paused,
                Key::S => shadows = !shadows,
                Key::R => reflections = !reflections,
                Key::C => {
                    let check = unsafe {
                        kernels.rainbow_geometry(
                            LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
                            &short,
                            WIDTH as u32,
                            HEIGHT as u32,
                            0,
                        )
                    };
                    safety_status = match check {
                        Err(rocm_oxide::Error::InvalidLaunch(_)) => {
                            format!(
                                "safety demo passed: rejected {} px buffer for {} px kernel",
                                pixel_count / 2,
                                pixel_count
                            )
                        }
                        Err(err) => format!("unexpected validation error: {err}"),
                        Ok(()) => "contract check unexpectedly launched".to_string(),
                    };
                }
                _ => {}
            }
        }
        work_iters = clamp_work_iters(work_iters);

        if !paused {
            frame_index = start.elapsed().as_millis() as u32 / 16;
        }

        render_mode(
            mode,
            &kernels,
            &device_frame,
            &device_camera,
            &mut camera,
            pixel_count,
            block_x,
            frame_index,
            work_iters,
            shadows,
            reflections,
        )?;

        if party {
            let intensity = 0.10 + (((frame_index as f32) * 0.035).sin() + 1.0) * 0.08;
            party_intensity.set(intensity)?;
            launch_party_post(
                &party_module,
                &party_kernel,
                &device_frame,
                pixel_count,
                frame_index,
                mode as u32,
            )?;
        } else {
            party_intensity.set(0.0)?;
        }

        unsafe {
            kernels.affine_transform(
                LaunchConfig::for_num_elems_with_block_size(4, block_x),
                &d_affine_output,
                &d_affine_input,
                &d_affine_params,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        let affine = d_affine_output.copy_to_vec()?;

        window.update_with_device_buffer_and_regions(
            &device_frame,
            WIDTH,
            HEIGHT,
            &overlay_regions(),
            |overlay| {
                draw_overlay(
                    overlay,
                    &device,
                    mode,
                    work_iters,
                    fps,
                    party,
                    paused,
                    shadows,
                    reflections,
                    &safety_status,
                    affine[2],
                );
            },
        )?;

        frames += 1;
        rendered_frames += 1;
        if last_title.elapsed() >= Duration::from_millis(500) {
            fps = frames as f64 / last_title.elapsed().as_secs_f64();
            frames = 0;
            last_title = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide Possibilities | {:.1} FPS | {} | party {} | {}",
                fps,
                MODES[mode],
                on_off(party),
                device.arch()
            ));
        }
        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    Ok(())
}

fn render_mode(
    mode: usize,
    kernels: &generated::DeviceKernels,
    device_frame: &DeviceBuffer<u32>,
    device_camera: &DeviceBuffer<f32>,
    camera: &mut [f32],
    pixel_count: usize,
    block_x: u32,
    frame_index: u32,
    work_iters: u32,
    shadows: bool,
    reflections: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        0 => unsafe {
            kernels.rainbow_geometry(
                LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
                device_frame,
                WIDTH as u32,
                HEIGHT as u32,
                frame_index,
            )?;
        },
        1 => unsafe {
            kernels.stress_pattern(
                LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
                device_frame,
                frame_index,
                (frame_index / 90) & 7,
                work_iters,
            )?;
        },
        2 => unsafe {
            kernels.stress_3d(
                LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
                device_frame,
                frame_index,
                (frame_index / 100) & 7,
                work_iters.max(32),
            )?;
        },
        _ => {
            write_orbit_camera(camera, frame_index, shadows, reflections);
            device_camera.copy_from_host(camera)?;
            unsafe {
                kernels.raytrace_world(
                    LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
                    device_frame,
                    device_camera,
                    frame_index,
                )?;
            }
        }
    }
    rocm_oxide::hip::synchronize()?;
    Ok(())
}

fn clamp_work_iters(value: u32) -> u32 {
    value.clamp(1, MAX_STRESS_WORK_ITERS)
}

fn launch_party_post(
    _module: &Module,
    kernel: &rocm_oxide::Kernel,
    frame: &DeviceBuffer<u32>,
    pixel_count: usize,
    frame_index: u32,
    mode: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame_ptr = frame.as_mut_ptr();
    let mut n_arg = pixel_count as u64;
    let mut frame_arg = frame_index;
    let mut mode_arg = mode;
    let mut params = [
        rocm_oxide::__private::arg_ptr(&mut frame_ptr),
        rocm_oxide::__private::arg_ptr(&mut n_arg),
        rocm_oxide::__private::arg_ptr(&mut frame_arg),
        rocm_oxide::__private::arg_ptr(&mut mode_arg),
    ];
    unsafe {
        kernel.launch_raw(LaunchConfig::for_num_elems(pixel_count), &mut params)?;
    }
    Ok(())
}

fn write_orbit_camera(camera: &mut [f32], frame_index: u32, shadows: bool, reflections: bool) {
    let t = frame_index as f32 * 0.018;
    let pos = Vec3::new(
        t.sin() * 2.15,
        0.42 + (t * 0.7).sin() * 0.15,
        -1.3 + t.cos() * 0.7,
    );
    let yaw = t.sin() * 0.45;
    let pitch = 0.08 + (t * 0.5).sin() * 0.08;
    let cp = pitch.cos();
    let forward = Vec3::new(yaw.sin() * cp, pitch.sin(), yaw.cos() * cp).normalize();
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = world_up.cross(forward).normalize();
    let up = forward.cross(right).normalize();
    let flags = (shadows as u32) | ((reflections as u32) << 1);

    camera[0] = pos.x;
    camera[1] = pos.y;
    camera[2] = pos.z;
    camera[3] = right.x;
    camera[4] = right.y;
    camera[5] = right.z;
    camera[6] = up.x;
    camera[7] = up.y;
    camera[8] = up.z;
    camera[9] = forward.x;
    camera[10] = forward.y;
    camera[11] = forward.z;
    camera[12] = flags as f32;
}

fn draw_overlay(
    frame: &mut [u32],
    device: &Device,
    mode: usize,
    work_iters: u32,
    fps: f64,
    party: bool,
    paused: bool,
    shadows: bool,
    reflections: bool,
    safety_status: &str,
    affine_probe: f32,
) {
    draw_rect(frame, 18, 18, 500, 216, 0x111820);
    draw_rect_outline(frame, 18, 18, 500, 216, 0x5cc8ff);
    draw_rect(frame, 22, 22, 492, 3, 0x2fb8ff);
    draw_text(frame, 34, 34, "ROCm-Oxide Possibilities", 0xffffff);
    draw_text(
        frame,
        34,
        56,
        "Rust GPU kernels -> AMD HSACO -> safe typed host calls",
        0x9ddcff,
    );
    draw_text(frame, 34, 82, &format!("GPU: {}", device.arch()), 0xd8f3ff);
    draw_text(
        frame,
        34,
        104,
        &format!("Mode: {}  {}", mode + 1, MODES[mode]),
        0xffe680,
    );
    draw_text(
        frame,
        34,
        126,
        &format!("Render FPS: {:>5.1}", fps),
        0x90ffb8,
    );
    draw_text(
        frame,
        34,
        148,
        &format!(
            "module-global post effect: {}  paused: {}",
            on_off(party),
            on_off(paused)
        ),
        0xf0d0ff,
    );
    draw_text(
        frame,
        34,
        170,
        &mode_status(mode, work_iters, shadows, reflections),
        0xd4d8e2,
    );
    draw_text(
        frame,
        34,
        192,
        &format!("repr(C) env: affine params on GPU -> {:.1}", affine_probe),
        0xd4d8e2,
    );
    draw_text(frame, 34, 214, "click top tabs or use keyboard", 0x9aa8b8);

    draw_rect(frame, 18, HEIGHT - 126, WIDTH - 36, 96, 0x141a22);
    draw_rect_outline(frame, 18, HEIGHT - 126, WIDTH - 36, 96, 0x7b90a4);
    draw_text_clipped(
        frame,
        34,
        HEIGHT - 110,
        "1 rainbow  2 2D stress  3 3D volume  4 raytrace",
        0xffffff,
        WIDTH - 52,
    );
    draw_text_clipped(
        frame,
        34,
        HEIGHT - 88,
        "P optional global post   C safety demo   arrows mode/work   S/R ray flags in mode 4   Space pause   Esc exit",
        0xc8d8ff,
        WIDTH - 52,
    );
    draw_text_clipped(frame, 34, HEIGHT - 64, safety_status, 0xffc080, WIDTH - 52);

    for i in 0..MODES.len() {
        let (x, y, w, h) = mode_button_rect(i);
        let color = if i == mode { 0x2f78a0 } else { 0x202a34 };
        draw_rect(frame, x, y, w, h, color);
        draw_rect_outline(frame, x, y, w, h, 0x88b8d8);
        draw_text(
            frame,
            x + 14,
            y + 12,
            &format!("{} {}", i + 1, short_mode_name(i)),
            0xffffff,
        );
        draw_text(frame, x + 14, y + 32, mode_caption(i), 0xb8d8f0);
    }
}

fn overlay_regions() -> [CopyRegion; 6] {
    [
        CopyRegion::new(18, 18, 500, 216),
        CopyRegion::new(18, HEIGHT - 126, WIDTH - 36, 96),
        CopyRegion::new(548, 32, 104, 60),
        CopyRegion::new(666, 32, 104, 60),
        CopyRegion::new(784, 32, 104, 60),
        CopyRegion::new(902, 32, 104, 60),
    ]
}

fn mode_button_rect(index: usize) -> (usize, usize, usize, usize) {
    (548 + index * 118, 32, 104, 60)
}

fn mode_from_mouse(x: usize, y: usize) -> Option<usize> {
    (0..MODES.len()).find(|&index| {
        let (bx, by, bw, bh) = mode_button_rect(index);
        x >= bx && x < bx + bw && y >= by && y < by + bh
    })
}

fn short_mode_name(index: usize) -> &'static str {
    match index {
        0 => "Rainbow",
        1 => "2D",
        2 => "3D",
        _ => "Ray",
    }
}

fn mode_caption(index: usize) -> &'static str {
    match index {
        0 => "Rust kernel",
        1 => "ALU load",
        2 => "3D objects",
        _ => "camera scene",
    }
}

fn mode_status(mode: usize, work_iters: u32, shadows: bool, reflections: bool) -> String {
    match mode {
        0 => "active: generated Rust rainbow kernel only".to_string(),
        1 => format!("active: generated Rust 2D ALU stress, work={work_iters}"),
        2 => format!(
            "active: generated Rust 3D volume stress, steps={}",
            work_iters.max(32)
        ),
        _ => format!(
            "active: raytraced world, shadows={} reflections={}",
            on_off(shadows),
            on_off(reflections)
        ),
    }
}

fn draw_rect(frame: &mut [u32], x: usize, y: usize, w: usize, h: usize, color: u32) {
    let x_end = (x + w).min(WIDTH);
    let y_end = (y + h).min(HEIGHT);
    for py in y.min(HEIGHT)..y_end {
        let row = py * WIDTH;
        for px in x.min(WIDTH)..x_end {
            frame[row + px] = color;
        }
    }
}

fn draw_rect_outline(frame: &mut [u32], x: usize, y: usize, w: usize, h: usize, color: u32) {
    draw_rect(frame, x, y, w, 1, color);
    draw_rect(frame, x, y + h.saturating_sub(1), w, 1, color);
    draw_rect(frame, x, y, 1, h, color);
    draw_rect(frame, x + w.saturating_sub(1), y, 1, h, color);
}

fn draw_text(frame: &mut [u32], x: usize, y: usize, text: &str, color: u32) {
    draw_text_clipped(frame, x, y, text, color, WIDTH - x);
}

fn draw_text_clipped(
    frame: &mut [u32],
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    max_width: usize,
) {
    let mut cx = x;
    let max_x = x.saturating_add(max_width).min(WIDTH);
    for ch in text.chars() {
        if cx + 8 > max_x {
            break;
        }
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits >> col) & 1 == 1 {
                        let px = cx + col;
                        let py = y + row;
                        if px < WIDTH && py < HEIGHT {
                            frame[py * WIDTH + px] = color;
                        }
                    }
                }
            }
        }
        cx += 8;
    }
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

struct InstanceGuard {
    path: PathBuf,
}

impl InstanceGuard {
    fn acquire() -> Result<Self, Box<dyn std::error::Error>> {
        let path = std::env::temp_dir().join("rocm_oxide_possibilities_window.pid");
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                Ok(Self { path })
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if stale_pid_file(&path) {
                    let _ = fs::remove_file(&path);
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)?;
                    writeln!(file, "{}", std::process::id())?;
                    Ok(Self { path })
                } else {
                    Err(format!(
                        "another possibilities_window instance is already running; close it first or remove {} if it is stale",
                        path.display()
                    )
                    .into())
                }
            }
            Err(err) => Err(Box::new(err)),
        }
    }
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn stale_pid_file(path: &PathBuf) -> bool {
    let mut text = String::new();
    let Ok(mut file) = File::open(path) else {
        return true;
    };
    if file.read_to_string(&mut text).is_err() {
        return true;
    }
    let Ok(pid) = text.trim().parse::<u32>() else {
        return true;
    };
    !PathBuf::from(format!("/proc/{pid}")).exists()
}
