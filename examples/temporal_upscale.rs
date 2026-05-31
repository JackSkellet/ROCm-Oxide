use image::{Rgb, RgbImage};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::Instant;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const SRC_W: usize = 512;
const SRC_H: usize = 288;
const DST_W: usize = 1024;
const DST_H: usize = 576;
const PIXELS: usize = DST_W * DST_H;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let (prev_color, _, _) = make_scene(0);
    let (color, depth, motion_reactive) = make_scene(1);
    let prev_history = cpu_bilinear_upscale(&prev_color);

    let device_color = DeviceBuffer::<u32>::from_slice(&color)?;
    let device_depth = DeviceBuffer::<f32>::from_slice(&depth)?;
    let device_motion_reactive = DeviceBuffer::<f32>::from_slice(&motion_reactive)?;
    let device_prev_history = DeviceBuffer::<u32>::from_slice(&prev_history)?;
    let device_frame = DeviceBuffer::<u32>::new(PIXELS)?;
    let device_history_out = DeviceBuffer::<u32>::new(PIXELS)?;
    let config = LaunchConfig::for_num_elems(PIXELS, 256);

    let mut host_frame = vec![0u32; PIXELS];
    unsafe {
        kernels.depth_aware_upscale(
            config,
            &device_frame,
            &device_color,
            &device_depth,
            PIXELS,
            4u32 << 4,
            256,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    device_frame.copy_to_host(&mut host_frame)?;
    let current_only = host_frame.clone();
    save_rgb(&current_only, "target/temporal_upscale_current_only.png")?;

    let mut temporal = vec![0u32; PIXELS];
    for (name, mode) in [
        ("temporal", 4u32 << 4),
        ("motion", 1u32),
        ("reactive", 2u32),
        ("depth_edges", 3u32),
        ("history_weight", 4u32),
    ] {
        let start = Instant::now();
        unsafe {
            kernels.temporal_reconstruct_upscale(
                config,
                &device_frame,
                &device_history_out,
                &device_color,
                &device_depth,
                &device_motion_reactive,
                &device_prev_history,
                PIXELS,
                mode,
                256,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        let elapsed = start.elapsed();
        device_frame.copy_to_host(&mut host_frame)?;
        save_rgb(&host_frame, &format!("target/temporal_upscale_{name}.png"))?;
        if name == "temporal" {
            temporal.copy_from_slice(&host_frame);
            println!(
                "temporal upscale GPU pass: {:.3} ms",
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    let changed = pixel_delta_score(&current_only, &temporal);
    assert!(changed > 100_000, "temporal pass did not affect the frame");
    println!(
        "AMD-guided temporal Rust GPU upscaler wrote target/temporal_upscale_*.png on {} (delta score {changed})",
        device.arch()
    );
    Ok(())
}

fn make_scene(frame: u32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut color = vec![0u32; SRC_W * SRC_H];
    let mut depth = vec![1.0f32; SRC_W * SRC_H];
    let mut motion_reactive = vec![0.0f32; SRC_W * SRC_H * 3];

    let prev_cx = 174.0f32;
    let curr_cx = 214.0f32;
    let cx = if frame == 0 { prev_cx } else { curr_cx };
    let cy = 142.0f32;
    let motion_x = (prev_cx - curr_cx) * 2.0;

    for y in 0..SRC_H {
        for x in 0..SRC_W {
            let i = y * SRC_W + x;
            let fx = x as f32 / SRC_W as f32;
            let fy = y as f32 / SRC_H as f32;
            let mut r = (18.0 + fx * 28.0) as u32;
            let mut g = (30.0 + fy * 72.0) as u32;
            let mut b = (52.0 + (1.0 - fy) * 112.0) as u32;
            let mut z = 0.88 + fy * 0.08;
            let mut reactive = 0.0f32;
            let mut mvx = 0.0f32;
            let mvy = 0.0f32;

            if x > 280 && x < 438 && y > 70 && y < 205 {
                r = 44;
                g = 184;
                b = 235;
                z = 0.50;
            }

            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < 69.0 * 69.0 {
                let rim = (d2.sqrt() / 69.0).clamp(0.0, 1.0);
                r = (240.0 - rim * 30.0) as u32;
                g = (66.0 + rim * 46.0) as u32;
                b = (48.0 + rim * 28.0) as u32;
                z = 0.30;
                reactive = 0.22 + rim * 0.42;
                mvx = motion_x;
            }

            let glow_dx = x as f32 - (cx + 72.0);
            let glow_dy = y as f32 - (cy - 34.0);
            let glow_d2 = glow_dx * glow_dx + glow_dy * glow_dy;
            if glow_d2 < 31.0 * 31.0 {
                let a = 1.0 - (glow_d2.sqrt() / 31.0).clamp(0.0, 1.0);
                r = ((r as f32) * (1.0 - a) + 255.0 * a) as u32;
                g = ((g as f32) * (1.0 - a) + 200.0 * a) as u32;
                b = ((b as f32) * (1.0 - a) + 48.0 * a) as u32;
                reactive = reactive.max(0.82 * a);
                mvx = motion_x;
            }

            if frame != 0 {
                let old_dx = x as f32 - prev_cx;
                let old_dy = y as f32 - cy;
                let was_covered = old_dx * old_dx + old_dy * old_dy < 72.0 * 72.0;
                let now_covered = dx * dx + dy * dy < 72.0 * 72.0;
                if was_covered && !now_covered {
                    reactive = 1.0;
                }
            }

            if ((x / 16) ^ (y / 16)) & 1 == 0 {
                r = (r + 10).min(255);
                g = (g + 10).min(255);
                b = (b + 10).min(255);
            }

            color[i] = (r << 16) | (g << 8) | b;
            depth[i] = z;
            motion_reactive[i * 3] = if frame == 0 { 0.0 } else { mvx };
            motion_reactive[i * 3 + 1] = mvy;
            motion_reactive[i * 3 + 2] = if frame == 0 { 0.0 } else { reactive };
        }
    }

    (color, depth, motion_reactive)
}

fn cpu_bilinear_upscale(color: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; PIXELS];
    for y in 0..DST_H {
        for x in 0..DST_W {
            let sx = (((x as f32) + 0.5) * 0.5 - 0.5).clamp(0.0, 511.0);
            let sy = (((y as f32) + 0.5) * 0.5 - 0.5).clamp(0.0, 287.0);
            let x0 = sx as usize;
            let y0 = sy as usize;
            let x1 = (x0 + 1).min(SRC_W - 1);
            let y1 = (y0 + 1).min(SRC_H - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            out[y * DST_W + x] = bilinear(
                color[y0 * SRC_W + x0],
                color[y0 * SRC_W + x1],
                color[y1 * SRC_W + x0],
                color[y1 * SRC_W + x1],
                fx,
                fy,
            );
        }
    }
    out
}

fn bilinear(c00: u32, c10: u32, c01: u32, c11: u32, fx: f32, fy: f32) -> u32 {
    let r = channel(c00 >> 16, c10 >> 16, c01 >> 16, c11 >> 16, fx, fy);
    let g = channel(c00 >> 8, c10 >> 8, c01 >> 8, c11 >> 8, fx, fy);
    let b = channel(c00, c10, c01, c11, fx, fy);
    (r << 16) | (g << 8) | b
}

fn channel(c00: u32, c10: u32, c01: u32, c11: u32, fx: f32, fy: f32) -> u32 {
    let top = (c00 & 255) as f32 + ((c10 & 255) as f32 - (c00 & 255) as f32) * fx;
    let bottom = (c01 & 255) as f32 + ((c11 & 255) as f32 - (c01 & 255) as f32) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 255.0) as u32
}

fn pixel_delta_score(a: &[u32], b: &[u32]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let ar = ((*a >> 16) & 255) as i32;
            let ag = ((*a >> 8) & 255) as i32;
            let ab = (*a & 255) as i32;
            let br = ((*b >> 16) & 255) as i32;
            let bg = ((*b >> 8) & 255) as i32;
            let bb = (*b & 255) as i32;
            (ar - br).unsigned_abs() as u64
                + (ag - bg).unsigned_abs() as u64
                + (ab - bb).unsigned_abs() as u64
        })
        .sum()
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
