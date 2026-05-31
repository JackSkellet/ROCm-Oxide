use minifb::{Key, Scale, Window, WindowOptions};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::{Duration, Instant};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1024;
const HEIGHT: usize = 576;
const CAMERA_PARAMS: usize = 16;

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

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
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
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let pixel_count = WIDTH * HEIGHT;
    let block_x = 256u32;
    let device_frame = DeviceBuffer::<u32>::new(pixel_count)?;
    let device_camera = DeviceBuffer::<f32>::new(CAMERA_PARAMS)?;
    let mut host_frame = vec![0u32; pixel_count];
    let mut camera_params = vec![0.0f32; CAMERA_PARAMS];

    let mut window = Window::new(
        "ROCm-Oxide Raytraced World",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;

    let mut pos = Vec3::new(0.0, 0.28, -1.6);
    let mut yaw = 0.0f32;
    let mut pitch = 0.04f32;
    let mut shadows = true;
    let mut reflections = true;
    let mut paused = false;
    let mut frame_index = 0u32;
    let mut frames = 0u32;
    let mut rendered_frames = 0u32;
    let mut last_title = Instant::now();
    let mut last_frame = Instant::now();
    let start = Instant::now();
    let max_frames = std::env::var("ROCM_OXIDE_RAYTRACE_MAX_FRAMES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok());
    let toggle_reflections_at = std::env::var("ROCM_OXIDE_RAYTRACE_TOGGLE_REFLECTIONS_AT")
        .ok()
        .and_then(|value| value.parse::<u32>().ok());

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32().min(0.05);
        last_frame = now;

        if window.is_key_down(Key::Left) {
            yaw -= 1.8 * dt;
        }
        if window.is_key_down(Key::Right) {
            yaw += 1.8 * dt;
        }
        if window.is_key_down(Key::Up) {
            pitch = (pitch + 1.2 * dt).min(0.78);
        }
        if window.is_key_down(Key::Down) {
            pitch = (pitch - 1.2 * dt).max(-0.55);
        }
        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            shadows = !shadows;
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            reflections = !reflections;
        }
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            paused = !paused;
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            pos = Vec3::new(0.0, 0.28, -1.6);
            yaw = 0.0;
            pitch = 0.04;
        }
        if toggle_reflections_at == Some(rendered_frames) {
            reflections = !reflections;
        }

        let forward_flat = Vec3::new(yaw.sin(), 0.0, yaw.cos()).normalize();
        let right_flat = Vec3::new(forward_flat.z, 0.0, -forward_flat.x);
        let mut delta = Vec3::new(0.0, 0.0, 0.0);
        let speed = if window.is_key_down(Key::LeftShift) {
            3.0
        } else {
            1.45
        };
        if window.is_key_down(Key::W) {
            delta = delta.add(forward_flat.scale(speed * dt));
        }
        if window.is_key_down(Key::S) {
            delta = delta.add(forward_flat.scale(-speed * dt));
        }
        if window.is_key_down(Key::A) {
            delta = delta.add(right_flat.scale(-speed * dt));
        }
        if window.is_key_down(Key::D) {
            delta = delta.add(right_flat.scale(speed * dt));
        }
        pos = collide_move(pos, delta);

        if !paused {
            frame_index = start.elapsed().as_millis() as u32 / 16;
        }

        let cp = pitch.cos();
        let forward = Vec3::new(yaw.sin() * cp, pitch.sin(), yaw.cos() * cp).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let flags = (shadows as u32) | ((reflections as u32) << 1);

        camera_params[0] = pos.x;
        camera_params[1] = pos.y;
        camera_params[2] = pos.z;
        camera_params[3] = right.x;
        camera_params[4] = right.y;
        camera_params[5] = right.z;
        camera_params[6] = up.x;
        camera_params[7] = up.y;
        camera_params[8] = up.z;
        camera_params[9] = forward.x;
        camera_params[10] = forward.y;
        camera_params[11] = forward.z;
        camera_params[12] = flags as f32;
        device_camera.copy_from_host(&camera_params)?;

        let render_result: Result<(), Box<dyn std::error::Error>> = (|| {
            unsafe {
                kernels.raytrace_world(
                    LaunchConfig::for_num_elems_with_block_size(pixel_count, block_x),
                    &device_frame,
                    &device_camera,
                    frame_index,
                )?;
            }
            rocm_oxide::hip::synchronize()?;
            device_frame.copy_to_host(&mut host_frame)?;
            Ok(())
        })();

        if let Err(err) = render_result {
            window.set_title(&format!(
                "ROCm-Oxide Raytraced World | render error: {err} | Esc exit"
            ));
            window.update();
            std::thread::sleep(Duration::from_millis(120));
            continue;
        }

        window.update_with_buffer(&host_frame, WIDTH, HEIGHT)?;
        frames += 1;
        rendered_frames += 1;
        let elapsed = last_title.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let fps = frames as f64 / elapsed.as_secs_f64();
            frames = 0;
            last_title = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide Raytraced World | {:.1} FPS | pos {:.1},{:.1} | shadows {} | reflections {} | WASD move, arrows look, 1/2 toggle, R reset, Esc exit | {}",
                fps,
                pos.x,
                pos.z,
                on_off(shadows),
                on_off(reflections),
                device.arch(),
            ));
        }
        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    Ok(())
}

fn collide_move(pos: Vec3, delta: Vec3) -> Vec3 {
    let radius = 0.32;
    let mut next = pos;

    let try_x = Vec3::new(pos.x + delta.x, pos.y, pos.z);
    if !blocked(try_x, radius) {
        next.x = try_x.x;
    }

    let try_z = Vec3::new(next.x, pos.y, pos.z + delta.z);
    if !blocked(try_z, radius) {
        next.z = try_z.z;
    }

    next
}

fn blocked(pos: Vec3, radius: f32) -> bool {
    if pos.x < -4.4 || pos.x > 4.4 || pos.z < -2.4 || pos.z > 8.8 {
        return true;
    }

    sphere_block(pos, -1.65, 3.2, 0.58 + radius)
        || sphere_block(pos, 1.25, 2.25, 0.42 + radius)
        || box_block(pos, 1.65, 3.45, 2.55, 4.2, radius)
        || box_block(pos, -2.75, 5.05, -1.95, 5.85, radius)
        || box_block(pos, -0.45, 6.8, 0.45, 7.7, radius)
}

fn sphere_block(pos: Vec3, cx: f32, cz: f32, radius: f32) -> bool {
    let dx = pos.x - cx;
    let dz = pos.z - cz;
    dx * dx + dz * dz < radius * radius
}

fn box_block(pos: Vec3, min_x: f32, min_z: f32, max_x: f32, max_z: f32, radius: f32) -> bool {
    pos.x > min_x - radius
        && pos.x < max_x + radius
        && pos.z > min_z - radius
        && pos.z < max_z + radius
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}
