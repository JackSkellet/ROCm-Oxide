use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::{Duration, Instant};

#[path = "shared/visual_presenter.rs"]
mod visual_presenter;
use visual_presenter::{Key, KeyRepeat, Scale, Window, WindowOptions, requested_frames};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1024;
const HEIGHT: usize = 512;
const MAX_STRESS_WORK_ITERS: u32 = 4_096;
const MODES: [&str; 8] = [
    "plasma xor",
    "radial tunnel",
    "bitwise lattice",
    "diamond fields",
    "moire grid",
    "hash noise",
    "multiply warp",
    "distance heat",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let n = WIDTH * HEIGHT;
    let block_x = 256u32;
    let device_frame = DeviceBuffer::<u32>::new(n)?;
    let mut host_frame = vec![0u32; n];

    let mut window = Window::new(
        "ROCm-Oxide Stress Test",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
        },
    )?;

    let mut mode = 0usize;
    let mut work_iters = 16u32;
    let mut paused = false;
    let mut frame_index = 0u32;
    let mut frames = 0u32;
    let mut rendered_frames = 0u32;
    let mut last_title = Instant::now();
    let start = Instant::now();
    let max_frames = requested_frames("ROCM_OXIDE_STRESS_TEST_MAX_FRAMES");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for key in window.get_keys_pressed(KeyRepeat::Yes) {
            match key {
                Key::Right => mode = (mode + 1) % MODES.len(),
                Key::Left => mode = (mode + MODES.len() - 1) % MODES.len(),
                Key::Up => work_iters = clamp_work_iters(work_iters.saturating_add(8)),
                Key::Down => work_iters = work_iters.saturating_sub(8),
                Key::PageUp => work_iters = clamp_work_iters(work_iters.saturating_add(64)),
                Key::PageDown => work_iters = work_iters.saturating_sub(64),
                Key::Space => paused = !paused,
                Key::Key0 => work_iters = 0,
                Key::Key1 => work_iters = 16,
                Key::Key2 => work_iters = 64,
                Key::Key3 => work_iters = 256,
                _ => {}
            }
        }
        work_iters = clamp_work_iters(work_iters);

        if !paused {
            frame_index = start.elapsed().as_millis() as u32 / 16;
        }

        unsafe {
            kernels.stress_pattern(
                LaunchConfig::for_num_elems_with_block_size(n, block_x),
                &device_frame,
                frame_index,
                mode as u32,
                work_iters,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;
        window.update_with_buffer(&host_frame, WIDTH, HEIGHT)?;

        frames += 1;
        rendered_frames += 1;
        let elapsed = last_title.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let fps = frames as f64 / elapsed.as_secs_f64();
            frames = 0;
            last_title = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide Stress | {:.1} FPS | {} | work {} | {} | arrows mode/work, PgUp/PgDn heavy, Esc exit",
                fps,
                MODES[mode],
                work_iters,
                device.arch(),
            ));
        }
        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    Ok(())
}

fn clamp_work_iters(value: u32) -> u32 {
    value.min(MAX_STRESS_WORK_ITERS)
}
