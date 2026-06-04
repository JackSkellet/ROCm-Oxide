//! Vulkan plasma demo for ROCm-Oxide.
//!
//! Drop this file into:
//!
//! ```text
//! examples/vulkan_plasma_demo.rs
//! ```
//!
//! Run with the Vulkan presenter:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_plasma_demo
//! ```
//!
//! Optional bounded run:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_plasma_demo -- --frames 300
//! ```
//!
//! This demo intentionally does not add a new Vulkan stack. It reuses the
//! repository's existing `examples/shared/visual_presenter.rs`, which already
//! owns the SDL2 + ash Vulkan swapchain path. The frame itself is generated on
//! the CPU so this is a small, standalone Vulkan-presentation smoke test.
//!
//! Controls:
//!
//! - `Esc` closes the window.
//! - `W` / `S` increase/decrease animation speed.
//! - `A` / `D` change the pattern scale.
//! - `Up` / `Down` change color cycling.

use std::time::Instant;

#[path = "shared/visual_presenter.rs"]
mod visual_presenter;

use visual_presenter::{requested_frames, Key, Scale, Window, WindowOptions};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ROCM_OXIDE_VISUAL_PRESENT").as_deref() != Ok("vulkan") {
        eprintln!(
            "note: run with `ROCM_OXIDE_VISUAL_PRESENT=vulkan` to force the Vulkan presenter"
        );
    }

    let mut window = Window::new(
        "ROCm-Oxide Vulkan Plasma Demo",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
        },
    )?;
    window.set_target_fps(60);

    let max_frames = requested_frames("ROCM_OXIDE_VULKAN_PLASMA_FRAMES");
    let start = Instant::now();
    let mut rendered_frames = 0u32;

    let mut speed = 1.0f32;
    let mut scale = 1.0f32;
    let mut color_rate = 1.0f32;

    println!("vulkan_plasma_demo: {}x{}", WIDTH, HEIGHT);
    println!("controls: Esc quit | W/S speed | A/D scale | Up/Down color");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_down(Key::W) {
            speed = (speed * 1.015).min(4.0);
        }
        if window.is_key_down(Key::S) {
            speed = (speed * 0.985).max(0.15);
        }
        if window.is_key_down(Key::D) {
            scale = (scale * 1.01).min(3.0);
        }
        if window.is_key_down(Key::A) {
            scale = (scale * 0.99).max(0.35);
        }
        if window.is_key_down(Key::Up) {
            color_rate = (color_rate * 1.01).min(3.0);
        }
        if window.is_key_down(Key::Down) {
            color_rate = (color_rate * 0.99).max(0.25);
        }

        let elapsed = start.elapsed().as_secs_f32();
        let t = elapsed * speed;

        window.update_with_frame(WIDTH, HEIGHT, |frame| {
            draw_plasma(frame, WIDTH, HEIGHT, t, scale, color_rate);
        })?;

        rendered_frames = rendered_frames.wrapping_add(1);
        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    println!(
        "vulkan_plasma_demo: rendered {} frame(s), speed={:.2}, scale={:.2}, color_rate={:.2}",
        rendered_frames, speed, scale, color_rate
    );

    Ok(())
}

fn draw_plasma(frame: &mut [u32], width: usize, height: usize, t: f32, scale: f32, color_rate: f32) {
    let inv_w = 1.0 / width as f32;
    let inv_h = 1.0 / height as f32;
    let aspect = width as f32 / height as f32;

    let cx = 0.5 + 0.18 * (t * 0.27).sin();
    let cy = 0.5 + 0.16 * (t * 0.21).cos();

    for y in 0..height {
        let fy = y as f32 * inv_h;
        let sy = (fy - 0.5) * 2.0;

        for x in 0..width {
            let fx = x as f32 * inv_w;
            let sx = (fx - 0.5) * 2.0 * aspect;

            let dx = fx - cx;
            let dy = fy - cy;
            let r = (dx * dx + dy * dy).sqrt();

            let wave_a = ((sx * 10.0 * scale) + t * 1.35).sin();
            let wave_b = ((sy * 12.0 * scale) - t * 1.10).cos();
            let wave_c = ((r * 38.0 * scale) - t * 2.10).sin();
            let twist = ((sx * sy * 9.0 * scale) + t * 0.85).cos();

            let grid = soft_grid(fx, fy, t);
            let vignette = (1.45 - (sx * sx + sy * sy) * 0.33).clamp(0.25, 1.25);

            let v = (wave_a + wave_b + wave_c + twist) * 0.25;
            let hue = fract(0.58 + v * 0.16 + t * 0.035 * color_rate + grid * 0.04);
            let sat = (0.70 + 0.22 * wave_c.abs()).clamp(0.0, 1.0);
            let val = ((0.58 + 0.35 * v.abs() + grid * 0.18) * vignette).clamp(0.0, 1.0);

            frame[y * width + x] = hsv_to_rgb_u32(hue, sat, val);
        }
    }

    draw_border(frame, width, height, t);
}

fn soft_grid(x: f32, y: f32, t: f32) -> f32 {
    let gx = ((x * 24.0 + t * 0.18).fract() - 0.5).abs();
    let gy = ((y * 14.0 - t * 0.12).fract() - 0.5).abs();

    let line_x = smoothstep(0.040, 0.000, gx);
    let line_y = smoothstep(0.040, 0.000, gy);

    (line_x.max(line_y) * 0.65).clamp(0.0, 1.0)
}

fn draw_border(frame: &mut [u32], width: usize, height: usize, t: f32) {
    let border_color = hsv_to_rgb_u32(fract(t * 0.05), 0.8, 1.0);
    let inset = 12usize;
    let thickness = 3usize;

    for y in inset..(height - inset) {
        for dx in 0..thickness {
            frame[y * width + inset + dx] = border_color;
            frame[y * width + (width - inset - 1 - dx)] = border_color;
        }
    }

    for x in inset..(width - inset) {
        for dy in 0..thickness {
            frame[(inset + dy) * width + x] = border_color;
            frame[(height - inset - 1 - dy) * width + x] = border_color;
        }
    }
}

fn hsv_to_rgb_u32(h: f32, s: f32, v: f32) -> u32 {
    let h = fract(h) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;

    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    pack_rgb(r, g, b)
}

fn pack_rgb(r: f32, g: f32, b: f32) -> u32 {
    let r = (r.clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (g.clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (b.clamp(0.0, 1.0) * 255.0).round() as u32;

    (r << 16) | (g << 8) | b
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
