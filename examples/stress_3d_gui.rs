use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::{Duration, Instant};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1024;
const HEIGHT: usize = 512;
const MODES: [&str; 8] = [
    "cube lattice",
    "ray tunnel",
    "woven field",
    "manhattan volume",
    "bitwise hull",
    "hash fog",
    "shell warp",
    "mixed volume",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let n = WIDTH * HEIGHT;
    let block_x = 256u32;
    let device_frame = DeviceBuffer::<u32>::new(n)?;
    let mut host_frame = vec![0u32; n];

    let mut window = Window::new(
        "ROCm-Oxide 3D Stress",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;

    let mut mode = 0usize;
    let mut work_iters = 96u32;
    let mut paused = false;
    let mut frame_index = 0u32;
    let mut frames = 0u32;
    let mut last_title = Instant::now();
    let start = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for key in window.get_keys_pressed(KeyRepeat::Yes) {
            match key {
                Key::Right => mode = (mode + 1) % MODES.len(),
                Key::Left => mode = (mode + MODES.len() - 1) % MODES.len(),
                Key::Up => work_iters = work_iters.saturating_add(16),
                Key::Down => work_iters = work_iters.saturating_sub(16).max(1),
                Key::PageUp => work_iters = work_iters.saturating_add(128),
                Key::PageDown => work_iters = work_iters.saturating_sub(128).max(1),
                Key::Space => paused = !paused,
                Key::Key1 => work_iters = 32,
                Key::Key2 => work_iters = 96,
                Key::Key3 => work_iters = 256,
                Key::Key4 => work_iters = 512,
                _ => {}
            }
        }

        if !paused {
            frame_index = start.elapsed().as_millis() as u32 / 16;
        }

        unsafe {
            kernels.stress_3d(
                LaunchConfig::for_num_elems_with_block_size(n, block_x),
                &device_frame,
                n,
                frame_index,
                mode as u32,
                work_iters,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;
        window.update_with_buffer(&host_frame, WIDTH, HEIGHT)?;

        frames += 1;
        let elapsed = last_title.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let fps = frames as f64 / elapsed.as_secs_f64();
            frames = 0;
            last_title = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide 3D Stress | {:.1} FPS | {} | steps {} | {} | arrows mode/steps, 1-4 presets, Esc exit",
                fps,
                MODES[mode],
                work_iters,
                device.arch(),
            ));
        }
    }

    Ok(())
}
