use font8x8::{BASIC_FONTS, UnicodeFonts};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window as MiniWindow, WindowOptions};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::io::{self, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use xcap::Window as CaptureWindow;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const OUT_WIDTH: usize = 1024;
const OUT_HEIGHT: usize = 576;
const IN_WIDTH: usize = 1024;
const IN_HEIGHT: usize = 576;
const PANEL_W: usize = 292;
const UPSCALE_PRESETS: [(&str, usize, usize); 4] = [
    ("native", 1024, 576),
    ("quality", 768, 432),
    ("balanced", 640, 360),
    ("perf", 512, 288),
];
const MODES: [&str; 7] = [
    "native sharp",
    "party neon",
    "scanline bloom",
    "pulse tint",
    "normal/depth edges",
    "3D card",
    "depth heat",
];

struct PickedWindow {
    window: CaptureWindow,
    label: String,
}

struct SharedCapture {
    pixels: Vec<u32>,
    sequence: u64,
    captures: u64,
    errors: u64,
}

#[derive(Clone, Copy)]
struct Button {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    action: Action,
}

#[derive(Clone, Copy)]
enum Action {
    Mode(usize),
    Upscale(u32),
    SharpnessDown,
    SharpnessUp,
    CaptureMs(u32),
    Freeze,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(picked) = pick_window()? else {
        return Ok(());
    };

    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let shared = Arc::new(Mutex::new(SharedCapture {
        pixels: vec![0u32; IN_WIDTH * IN_HEIGHT],
        sequence: 0,
        captures: 0,
        errors: 0,
    }));
    let capture_running = Arc::new(AtomicBool::new(true));
    let capture_frozen = Arc::new(AtomicBool::new(false));
    let capture_interval_ms = Arc::new(AtomicU32::new(16));
    let upscale_level = Arc::new(AtomicU32::new(0));
    let capture_thread = spawn_capture_thread(
        picked.window,
        Arc::clone(&shared),
        Arc::clone(&capture_running),
        Arc::clone(&capture_frozen),
        Arc::clone(&capture_interval_ms),
        Arc::clone(&upscale_level),
    );

    let output_pixels = OUT_WIDTH * OUT_HEIGHT;
    let device_input = DeviceBuffer::<u32>::new(IN_WIDTH * IN_HEIGHT)?;
    let device_frame = DeviceBuffer::<u32>::new(output_pixels)?;
    let mut host_input = vec![0u32; IN_WIDTH * IN_HEIGHT];
    let mut host_frame = vec![0u32; output_pixels];

    let mut window = MiniWindow::new(
        "ROCm-Oxide Window Effects Lab",
        OUT_WIDTH,
        OUT_HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;

    let mut mode = 0usize;
    let mut sharpness = 4u32;
    let mut frames = 0u32;
    let mut rendered_frames = 0u32;
    let mut last_sequence = u64::MAX;
    let mut last_capture_count = 0u64;
    let mut capture_fps = 0.0f64;
    let mut render_fps = 0.0f64;
    let mut previous_mouse = false;
    let mut last_title = Instant::now();
    let start = Instant::now();
    let max_frames = std::env::var("ROCM_OXIDE_WINDOW_FX_MAX_FRAMES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok());

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Right => mode = (mode + 1) % MODES.len(),
                Key::Left => mode = (mode + MODES.len() - 1) % MODES.len(),
                Key::Up => sharpness = (sharpness + 1).min(7),
                Key::Down => sharpness = sharpness.saturating_sub(1),
                Key::Space => toggle_freeze(&capture_frozen),
                _ => {}
            }
        }

        let buttons = panel_buttons();
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if mouse_down
            && !previous_mouse
            && let Some((mx, my)) = buffer_mouse_pos(&window)
        {
            handle_click(
                mx as usize,
                my as usize,
                &buttons,
                &mut mode,
                &mut sharpness,
                &capture_interval_ms,
                &capture_frozen,
                &upscale_level,
            );
        }
        previous_mouse = mouse_down;

        let (sequence, captures, errors) = {
            let shared = shared.lock().expect("capture mutex poisoned");
            if shared.sequence != last_sequence {
                host_input.copy_from_slice(&shared.pixels);
                last_sequence = shared.sequence;
                device_input.copy_from_host(&host_input)?;
            }
            (shared.sequence, shared.captures, shared.errors)
        };

        let frame_index = start.elapsed().as_millis() as u32 / 16;
        let current_upscale = upscale_level.load(Ordering::Relaxed).min(3);
        let packed_mode = mode as u32 | (sharpness << 4) | (current_upscale << 8);
        unsafe {
            kernels.window_fx(
                LaunchConfig::for_num_elems(output_pixels),
                &device_frame,
                &device_input,
                output_pixels,
                frame_index,
                packed_mode,
            )?;
        }
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;

        frames += 1;
        rendered_frames += 1;
        let elapsed = last_title.elapsed();
        if elapsed >= Duration::from_millis(500) {
            render_fps = frames as f64 / elapsed.as_secs_f64();
            capture_fps =
                (captures.saturating_sub(last_capture_count)) as f64 / elapsed.as_secs_f64();
            frames = 0;
            last_capture_count = captures;
            last_title = Instant::now();
            window.set_title(&format!(
                "ROCm-Oxide Window Effects Lab | render {:.1} FPS | capture {:.1} FPS | {} | {}",
                render_fps, capture_fps, MODES[mode], UPSCALE_PRESETS[current_upscale as usize].0,
            ));
        }

        render_panel(
            &mut host_frame,
            mode,
            sharpness,
            current_upscale,
            capture_interval_ms.load(Ordering::Relaxed),
            capture_frozen.load(Ordering::Relaxed),
            render_fps,
            capture_fps,
            sequence,
            errors,
            &picked.label,
            &buttons,
        );
        window.update_with_buffer(&host_frame, OUT_WIDTH, OUT_HEIGHT)?;

        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    capture_running.store(false, Ordering::Relaxed);
    let _ = capture_thread.join();
    Ok(())
}

fn spawn_capture_thread(
    window: CaptureWindow,
    shared: Arc<Mutex<SharedCapture>>,
    running: Arc<AtomicBool>,
    frozen: Arc<AtomicBool>,
    interval_ms: Arc<AtomicU32>,
    upscale_level: Arc<AtomicU32>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut local = vec![0u32; IN_WIDTH * IN_HEIGHT];
        while running.load(Ordering::Relaxed) {
            if frozen.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(16));
                continue;
            }

            match window.capture_image() {
                Ok(image) => {
                    downsample_to_u32(
                        &image,
                        &mut local,
                        upscale_level.load(Ordering::Relaxed).min(3),
                    );
                    let mut shared = shared.lock().expect("capture mutex poisoned");
                    shared.pixels.copy_from_slice(&local);
                    shared.sequence = shared.sequence.wrapping_add(1);
                    shared.captures = shared.captures.wrapping_add(1);
                }
                Err(_) => {
                    let mut shared = shared.lock().expect("capture mutex poisoned");
                    shared.errors = shared.errors.wrapping_add(1);
                }
            }

            let delay = interval_ms.load(Ordering::Relaxed);
            if delay > 0 {
                thread::sleep(Duration::from_millis(delay as u64));
            }
        }
    })
}

fn panel_buttons() -> Vec<Button> {
    let mut buttons = Vec::new();
    let mut y = 76;
    for index in 0..MODES.len() {
        buttons.push(Button {
            x: 14,
            y,
            w: 260,
            h: 24,
            action: Action::Mode(index),
        });
        y += 30;
    }
    buttons.push(Button {
        x: 14,
        y: 312,
        w: 58,
        h: 28,
        action: Action::Upscale(0),
    });
    buttons.push(Button {
        x: 80,
        y: 312,
        w: 58,
        h: 28,
        action: Action::Upscale(1),
    });
    buttons.push(Button {
        x: 146,
        y: 312,
        w: 58,
        h: 28,
        action: Action::Upscale(2),
    });
    buttons.push(Button {
        x: 212,
        y: 312,
        w: 58,
        h: 28,
        action: Action::Upscale(3),
    });
    buttons.push(Button {
        x: 14,
        y: 374,
        w: 82,
        h: 28,
        action: Action::SharpnessDown,
    });
    buttons.push(Button {
        x: 104,
        y: 374,
        w: 82,
        h: 28,
        action: Action::SharpnessUp,
    });
    buttons.push(Button {
        x: 194,
        y: 374,
        w: 80,
        h: 28,
        action: Action::Freeze,
    });
    for (x, ms) in [(14, 0), (80, 16), (146, 33), (212, 66)] {
        buttons.push(Button {
            x,
            y: 436,
            w: 58,
            h: 28,
            action: Action::CaptureMs(ms),
        });
    }
    buttons
}

fn handle_click(
    x: usize,
    y: usize,
    buttons: &[Button],
    mode: &mut usize,
    sharpness: &mut u32,
    interval_ms: &AtomicU32,
    frozen: &AtomicBool,
    upscale_level: &AtomicU32,
) {
    for button in buttons {
        if x >= button.x && x < button.x + button.w && y >= button.y && y < button.y + button.h {
            match button.action {
                Action::Mode(next) => *mode = next,
                Action::Upscale(level) => upscale_level.store(level.min(3), Ordering::Relaxed),
                Action::SharpnessDown => *sharpness = sharpness.saturating_sub(1),
                Action::SharpnessUp => *sharpness = (*sharpness + 1).min(7),
                Action::CaptureMs(ms) => interval_ms.store(ms, Ordering::Relaxed),
                Action::Freeze => toggle_freeze(frozen),
            }
        }
    }
}

fn buffer_mouse_pos(window: &MiniWindow) -> Option<(usize, usize)> {
    let (mx, my) = window.get_unscaled_mouse_pos(MouseMode::Discard)?;
    let (win_w, win_h) = window.get_size();
    if win_w == 0 || win_h == 0 {
        return None;
    }

    let x = ((mx.max(0.0) as f64) * OUT_WIDTH as f64 / win_w as f64)
        .floor()
        .clamp(0.0, (OUT_WIDTH - 1) as f64) as usize;
    let y = ((my.max(0.0) as f64) * OUT_HEIGHT as f64 / win_h as f64)
        .floor()
        .clamp(0.0, (OUT_HEIGHT - 1) as f64) as usize;
    Some((x, y))
}

fn toggle_freeze(frozen: &AtomicBool) {
    let current = frozen.load(Ordering::Relaxed);
    frozen.store(!current, Ordering::Relaxed);
}

#[allow(clippy::too_many_arguments)]
fn render_panel(
    frame: &mut [u32],
    mode: usize,
    sharpness: u32,
    upscale_level: u32,
    capture_ms: u32,
    frozen: bool,
    render_fps: f64,
    capture_fps: f64,
    sequence: u64,
    errors: u64,
    label: &str,
    buttons: &[Button],
) {
    draw_rect(frame, 0, 0, PANEL_W, OUT_HEIGHT, 0x15191f);
    draw_rect(frame, PANEL_W - 2, 0, 2, OUT_HEIGHT, 0x50d0ff);
    draw_text(frame, 14, 14, "Window FX Lab", 0xffffff);
    draw_text(frame, 14, 34, "ROCm Rust kernel preview", 0x9fd7ff);
    draw_text(frame, 14, 54, &truncate(label, 32), 0xd0d4dc);

    for button in buttons {
        let (label, active) = match button.action {
            Action::Mode(index) => (MODES[index].to_string(), index == mode),
            Action::Upscale(level) => (
                UPSCALE_PRESETS[level as usize].0.to_string(),
                level == upscale_level,
            ),
            Action::SharpnessDown => ("Sharp -".to_string(), false),
            Action::SharpnessUp => ("Sharp +".to_string(), false),
            Action::CaptureMs(ms) => (capture_label(ms).to_string(), ms == capture_ms),
            Action::Freeze => {
                if frozen {
                    ("Resume".to_string(), true)
                } else {
                    ("Freeze".to_string(), false)
                }
            }
        };
        let fill = if active { 0x30556e } else { 0x252b34 };
        draw_rect(frame, button.x, button.y, button.w, button.h, fill);
        draw_rect_outline(frame, button.x, button.y, button.w, button.h, 0x5d7180);
        draw_text(frame, button.x + 8, button.y + 8, &label, 0xf2f6fb);
    }

    draw_text(
        frame,
        14,
        292,
        &format!(
            "Upscaler: {} {}x{} -> 1024x576",
            UPSCALE_PRESETS[upscale_level as usize].0,
            UPSCALE_PRESETS[upscale_level as usize].1,
            UPSCALE_PRESETS[upscale_level as usize].2
        ),
        0xd0d4dc,
    );
    draw_text(
        frame,
        14,
        354,
        &format!("Sharpness: {sharpness}/7"),
        0xd0d4dc,
    );
    draw_text(frame, 14, 416, "Capture pacing", 0xd0d4dc);
    draw_text(
        frame,
        14,
        488,
        &format!("Render FPS: {:>5.1}", render_fps),
        0x9df7b3,
    );
    draw_text(
        frame,
        14,
        508,
        &format!("Capture FPS:{:>5.1}", capture_fps),
        0xffdc80,
    );
    draw_text(frame, 14, 528, &format!("Frames: {sequence}"), 0xd0d4dc);
    draw_text(
        frame,
        14,
        548,
        &format!("Capture errors: {errors}"),
        0xd0d4dc,
    );
}

fn capture_label(ms: u32) -> &'static str {
    match ms {
        0 => "max",
        16 => "60",
        33 => "30",
        66 => "15",
        _ => "?",
    }
}

fn draw_rect(frame: &mut [u32], x: usize, y: usize, w: usize, h: usize, color: u32) {
    let x_end = (x + w).min(OUT_WIDTH);
    let y_end = (y + h).min(OUT_HEIGHT);
    for py in y.min(OUT_HEIGHT)..y_end {
        let row = py * OUT_WIDTH;
        for px in x.min(OUT_WIDTH)..x_end {
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
    let mut cx = x;
    for ch in text.chars() {
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits >> col) & 1 == 1 {
                        let px = cx + col;
                        let py = y + row;
                        if px < OUT_WIDTH && py < OUT_HEIGHT {
                            frame[py * OUT_WIDTH + px] = color;
                        }
                    }
                }
            }
        }
        cx += 8;
        if cx >= OUT_WIDTH - 8 {
            break;
        }
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut out = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn pick_window() -> Result<Option<PickedWindow>, Box<dyn std::error::Error>> {
    let windows = CaptureWindow::all()?;
    let candidates = windows
        .into_iter()
        .filter_map(|window| {
            let title = window.title().ok()?;
            let app = window.app_name().unwrap_or_else(|_| "unknown".to_string());
            let width = window.width().ok()?;
            let height = window.height().ok()?;
            if title.trim().is_empty() || width < 160 || height < 100 {
                return None;
            }
            Some((window, app, title, width, height))
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        eprintln!(
            "No capturable windows found. On Wayland, some native windows may be hidden from normal clients."
        );
        return Ok(None);
    }

    let requested = std::env::args().nth(1);

    println!("Choose a window to mirror/effect:");
    for (index, (_, app, title, width, height)) in candidates.iter().enumerate() {
        println!(
            "  {:>2}: {:<24} {:>4}x{:<4} {}",
            index, app, width, height, title
        );
    }

    let selection = if let Some(requested) = requested {
        requested
    } else {
        print!("Window number or app/title text: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input
    };

    let selection = selection.trim();
    let index = if let Ok(index) = selection.parse::<usize>() {
        index
    } else {
        let needle = selection.to_lowercase();
        candidates
            .iter()
            .position(|(_, app, title, _, _)| {
                app.to_lowercase().contains(&needle) || title.to_lowercase().contains(&needle)
            })
            .unwrap_or(usize::MAX)
    };
    let Some((window, app, title, _, _)) = candidates.into_iter().nth(index) else {
        eprintln!("Invalid selection.");
        return Ok(None);
    };

    Ok(Some(PickedWindow {
        window,
        label: format!("{app}: {title}"),
    }))
}

fn downsample_to_u32(image: &image::RgbaImage, output: &mut [u32], upscale_level: u32) {
    let (_, dst_w, dst_h) = UPSCALE_PRESETS[upscale_level as usize];
    let src_w = image.width().max(1);
    let src_h = image.height().max(1);
    for y in 0..dst_h {
        let src_y = ((y as u32) * src_h / dst_h as u32).min(src_h - 1);
        for x in 0..dst_w {
            let src_x = ((x as u32) * src_w / dst_w as u32).min(src_w - 1);
            let px = image.get_pixel(src_x, src_y).0;
            output[y * IN_WIDTH + x] =
                ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32;
        }
    }
}
