use minifb::{Key, Scale, Window, WindowOptions};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::Instant;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1024;
const HEIGHT: usize = 512;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let n = WIDTH * HEIGHT;
    let block_x = 256u32;
    let device_frame = DeviceBuffer::<u32>::new(n)?;
    let mut host_frame = vec![0u32; n];

    let mut window = Window::new(
        "ROCm-Oxide Rainbow Geometry - Rust kernel on AMD GPU",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(60);

    let start = Instant::now();
    let mut frame_index = 0u32;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        unsafe {
            kernels.rainbow_geometry(
                LaunchConfig::for_num_elems_with_block_size(n, block_x),
                &device_frame,
                WIDTH as u32,
                HEIGHT as u32,
                frame_index,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;
        window.update_with_buffer(&host_frame, WIDTH, HEIGHT)?;

        frame_index = start.elapsed().as_millis() as u32 / 16;
    }

    Ok(())
}
