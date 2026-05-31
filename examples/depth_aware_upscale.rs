use image::{Rgb, RgbImage};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const SRC_W: usize = 512;
const SRC_H: usize = 288;
const DST_W: usize = 1024;
const DST_H: usize = 576;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let (color, depth) = make_low_res_scene();
    let device_color = DeviceBuffer::<u32>::from_slice(&color)?;
    let device_depth = DeviceBuffer::<f32>::from_slice(&depth)?;
    let device_frame = DeviceBuffer::<u32>::new(DST_W * DST_H)?;
    let mut host_frame = vec![0u32; DST_W * DST_H];

    for (name, mode) in [("color", 4u32 << 4), ("depth", 1u32), ("edge_mask", 2u32)] {
        unsafe {
            kernels.depth_aware_upscale(
                LaunchConfig::for_num_elems(DST_W * DST_H, 256),
                &device_frame,
                &device_color,
                &device_depth,
                DST_W * DST_H,
                mode,
                256,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;
        save_rgb(
            &host_frame,
            &format!("target/depth_aware_upscale_{name}.png"),
        )?;
    }

    let center = host_frame[(DST_H / 2) * DST_W + (DST_W / 2)];
    assert_ne!(center, 0, "upscaler produced an empty frame");
    println!(
        "Depth-aware Rust GPU upscaler wrote target/depth_aware_upscale_color.png, _depth.png, and _edge_mask.png on {}",
        device.arch()
    );
    Ok(())
}

fn make_low_res_scene() -> (Vec<u32>, Vec<f32>) {
    let mut color = vec![0u32; SRC_W * SRC_H];
    let mut depth = vec![1.0f32; SRC_W * SRC_H];

    for y in 0..SRC_H {
        for x in 0..SRC_W {
            let fx = x as f32 / SRC_W as f32;
            let fy = y as f32 / SRC_H as f32;
            let mut r = (24.0 + fx * 36.0) as u32;
            let mut g = (34.0 + fy * 70.0) as u32;
            let mut b = (58.0 + (1.0 - fy) * 90.0) as u32;
            let mut z = 0.82 + fy * 0.14;

            let dx = x as i32 - 172;
            let dy = y as i32 - 143;
            if dx * dx + dy * dy < 74 * 74 {
                r = 230;
                g = 78;
                b = 58;
                z = 0.32;
            }

            if x > 248 && x < 420 && y > 76 && y < 202 {
                r = 58;
                g = 190;
                b = 245;
                z = 0.48;
            }

            if ((x / 16) ^ (y / 16)) & 1 == 0 {
                r = (r + 12).min(255);
                g = (g + 12).min(255);
                b = (b + 12).min(255);
            }

            color[y * SRC_W + x] = (r << 16) | (g << 8) | b;
            depth[y * SRC_W + x] = z;
        }
    }

    (color, depth)
}

fn save_rgb(pixels: &[u32], path: &str) -> Result<(), image::ImageError> {
    let mut image = RgbImage::new(DST_W as u32, DST_H as u32);
    for y in 0..DST_H {
        for x in 0..DST_W {
            let rgb = pixels[y * DST_W + x];
            image.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    ((rgb >> 16) & 255) as u8,
                    ((rgb >> 8) & 255) as u8,
                    (rgb & 255) as u8,
                ]),
            );
        }
    }
    image.save(path)
}
