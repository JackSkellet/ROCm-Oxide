//! Vulkan + HIPRTC orbit-field demo for ROCm-Oxide.
//!
//! Drop this file into:
//!
//! ```text
//! examples/vulkan_orbit_field.rs
//! ```
//!
//! Run with Vulkan presentation:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_orbit_field
//! ```
//!
//! Optional bounded run:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_orbit_field -- --frames 300
//! ```
//!
//! This is intentionally self-contained:
//!
//! - no new device-spike kernel required
//! - no generated bindings required
//! - compiles a HIP C++ pixel kernel at runtime with HIPRTC
//! - writes directly into `DeviceBuffer<u32>`
//! - presents that GPU buffer through the existing Vulkan presenter
//!
//! Visual idea:
//!
//! A procedural "neon gravity field" made from moving attractors, orbit traps,
//! distance folds, and fake glow. It should look different from the existing
//! rainbow/stress demos.
//!
//! Controls:
//!
//! - `Esc` closes the window.
//! - `Left` / `Right` changes palette.
//! - `Up` / `Down` changes field strength.
//! - `Space` toggles auto-palette cycling.

use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::ffi::c_void;
use std::time::Instant;

#[path = "shared/visual_presenter.rs"]
mod visual_presenter;

use visual_presenter::{requested_frames, Key, KeyRepeat, Scale, Window, WindowOptions};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const PIXELS: usize = WIDTH * HEIGHT;

const ORBIT_FIELD_KERNEL: &str = r#"
extern "C" __global__ void orbit_field(
    unsigned int* frame,
    unsigned int width,
    unsigned int height,
    unsigned int frame_index,
    unsigned int palette,
    float field_strength
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = width * height;
    if (i >= n) {
        return;
    }

    unsigned int x = i % width;
    unsigned int y = i / width;

    float aspect = (float)width / (float)height;
    float uvx = ((float)x + 0.5f) / (float)width;
    float uvy = ((float)y + 0.5f) / (float)height;

    float px = (uvx * 2.0f - 1.0f) * aspect;
    float py = uvy * 2.0f - 1.0f;

    float t = (float)frame_index * 0.016f;

    // Moving attractors.
    float ax0 = 0.55f * sinf(t * 0.73f);
    float ay0 = 0.42f * cosf(t * 0.61f);
    float ax1 = 0.63f * cosf(t * 0.37f + 2.0f);
    float ay1 = 0.48f * sinf(t * 0.79f + 1.1f);
    float ax2 = 0.38f * sinf(t * 1.17f + 4.0f);
    float ay2 = 0.58f * sinf(t * 0.49f + 2.7f);

    float qx = px;
    float qy = py;

    float glow = 0.0f;
    float filaments = 0.0f;
    float orbit_min = 10.0f;

    // Domain folding/orbit-trap iterations.
    for (int k = 0; k < 9; ++k) {
        float dx0 = qx - ax0;
        float dy0 = qy - ay0;
        float dx1 = qx - ax1;
        float dy1 = qy - ay1;
        float dx2 = qx - ax2;
        float dy2 = qy - ay2;

        float d0 = sqrtf(dx0 * dx0 + dy0 * dy0) + 0.018f;
        float d1 = sqrtf(dx1 * dx1 + dy1 * dy1) + 0.018f;
        float d2 = sqrtf(dx2 * dx2 + dy2 * dy2) + 0.018f;

        orbit_min = fminf(orbit_min, fminf(d0, fminf(d1, d2)));

        float pull0 = 0.012f * field_strength / (d0 * d0 + 0.08f);
        float pull1 = 0.010f * field_strength / (d1 * d1 + 0.07f);
        float pull2 = 0.009f * field_strength / (d2 * d2 + 0.06f);

        qx += dx0 * pull0 - dx1 * pull1 + dy2 * pull2;
        qy += dy0 * pull0 + dy1 * pull1 - dx2 * pull2;

        // Fold and rotate a little each step.
        qx = fabsf(qx) - 0.72f;
        qy = fabsf(qy) - 0.47f;

        float a = 0.19f + 0.035f * sinf(t + (float)k);
        float ca = cosf(a);
        float sa = sinf(a);
        float rx = qx * ca - qy * sa;
        float ry = qx * sa + qy * ca;
        qx = rx;
        qy = ry;

        float stripe = sinf((qx * 19.0f + qy * 23.0f) - t * 3.0f + (float)k);
        filaments += 0.5f + 0.5f * stripe;

        glow += 0.035f / (orbit_min + 0.020f);
    }

    float radial = sqrtf(px * px + py * py);
    float rings = 0.5f + 0.5f * sinf(42.0f * orbit_min - t * 4.0f);
    float beams = powf(fmaxf(0.0f, filaments / 9.0f), 2.4f);
    float core = expf(-orbit_min * 8.0f);
    float vignette = fmaxf(0.0f, 1.25f - radial * 0.55f);

    float value = (0.25f * rings + 0.45f * beams + 0.85f * core + 0.10f * glow) * vignette;
    value = fminf(1.0f, value);

    float hue = fmodf(
        0.62f
        + 0.11f * (float)(palette & 7)
        + 0.12f * sinf(t * 0.21f)
        + 0.18f * rings
        + 0.10f * beams,
        1.0f
    );
    if (hue < 0.0f) {
        hue += 1.0f;
    }

    float sat = fminf(1.0f, 0.72f + 0.25f * core + 0.12f * beams);
    float val = fminf(1.0f, 0.10f + value * 1.35f);

    // HSV to RGB.
    float h6 = hue * 6.0f;
    int hi = (int)floorf(h6);
    float f = h6 - (float)hi;
    float p = val * (1.0f - sat);
    float q = val * (1.0f - f * sat);
    float u = val * (1.0f - (1.0f - f) * sat);

    float r, g, b;
    int sector = hi % 6;
    if (sector == 0) {
        r = val; g = u; b = p;
    } else if (sector == 1) {
        r = q; g = val; b = p;
    } else if (sector == 2) {
        r = p; g = val; b = u;
    } else if (sector == 3) {
        r = p; g = q; b = val;
    } else if (sector == 4) {
        r = u; g = p; b = val;
    } else {
        r = val; g = p; b = q;
    }

    // Add a subtle scanline and animated border.
    float scan = 0.92f + 0.08f * sinf((float)y * 0.75f);
    r *= scan;
    g *= scan;
    b *= scan;

    unsigned int border = 0;
    if (x < 5 || y < 5 || x >= width - 5 || y >= height - 5) {
        border = 1;
    }

    unsigned int ir = (unsigned int)(fminf(1.0f, r) * 255.0f);
    unsigned int ig = (unsigned int)(fminf(1.0f, g) * 255.0f);
    unsigned int ib = (unsigned int)(fminf(1.0f, b) * 255.0f);

    if (border) {
        float border_phase = 0.5f + 0.5f * sinf(t * 3.0f + (float)(palette & 7));
        ir = (unsigned int)(180.0f + 75.0f * border_phase);
        ig = (unsigned int)(50.0f + 120.0f * (1.0f - border_phase));
        ib = 255;
    }

    frame[i] = (ir << 16) | (ig << 8) | ib;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ROCM_OXIDE_VISUAL_PRESENT").as_deref() != Ok("vulkan") {
        eprintln!(
            "note: run with `ROCM_OXIDE_VISUAL_PRESENT=vulkan` to exercise Vulkan presentation"
        );
    }

    let device = Device::first()?;
    let module = device.compile_hip_source(ORBIT_FIELD_KERNEL)?;
    let kernel = module.kernel(c"orbit_field")?;

    let frame = DeviceBuffer::<u32>::new(PIXELS)?;

    let mut window = Window::new(
        "ROCm-Oxide Vulkan Orbit Field",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
        },
    )?;
    window.set_target_fps(60);

    let max_frames = requested_frames("ROCM_OXIDE_VULKAN_ORBIT_FIELD_FRAMES");
    let start = Instant::now();

    let mut rendered_frames = 0u32;
    let mut palette = 0u32;
    let mut auto_palette = true;
    let mut field_strength = 1.0f32;

    println!("vulkan_orbit_field: HIPRTC kernel -> DeviceBuffer -> Vulkan presenter");
    println!("controls: Esc quit | Left/Right palette | Up/Down field | Space auto-palette");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            auto_palette = !auto_palette;
        }

        if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
            palette = palette.wrapping_add(1) & 7;
            auto_palette = false;
        }
        if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
            palette = palette.wrapping_sub(1) & 7;
            auto_palette = false;
        }

        if window.is_key_down(Key::Up) {
            field_strength = (field_strength * 1.012).min(3.0);
        }
        if window.is_key_down(Key::Down) {
            field_strength = (field_strength * 0.988).max(0.25);
        }

        let elapsed_ms = start.elapsed().as_millis() as u32;
        let frame_index = elapsed_ms / 16;

        let active_palette = if auto_palette {
            (elapsed_ms / 2400) & 7
        } else {
            palette
        };

        unsafe {
            rocm_oxide::launch!(
                kernel,
                LaunchConfig::for_num_elems_with_block_size(PIXELS, 256),
                frame.as_mut_ptr(),
                WIDTH as u32,
                HEIGHT as u32,
                frame_index,
                active_palette,
                field_strength,
            )?;
        }

        rocm_oxide::hip::synchronize()?;
        window.update_with_device_buffer(&frame, WIDTH, HEIGHT)?;

        if rendered_frames % 90 == 0 {
            window.set_title(&format!(
                "ROCm-Oxide Vulkan Orbit Field | palette={} field={:.2} auto={}",
                active_palette, field_strength, auto_palette
            ));
        }

        rendered_frames = rendered_frames.wrapping_add(1);
        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    println!(
        "vulkan_orbit_field: rendered {} frame(s), palette={}, field_strength={:.2}",
        rendered_frames, palette, field_strength
    );

    Ok(())
}
