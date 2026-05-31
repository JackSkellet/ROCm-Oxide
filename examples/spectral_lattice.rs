use image::{Rgb, RgbImage};
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig, RocBlas, SgemmLayout};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 960;
const HEIGHT: usize = 540;
const BLOCK_X: u32 = 256;
const DEFAULT_OUTPUT: &str = "target/spectral_lattice.png";

struct DemoArgs {
    frames: Option<u32>,
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let pixel_count = WIDTH * HEIGHT;
    let device_frame = DeviceBuffer::<u32>::new(pixel_count)?;
    let mut host_frame = vec![0u32; pixel_count];
    let palette = derive_palette();

    if let Some(frames) = args.frames {
        let frames = frames.max(1);
        for frame_index in 0..frames {
            render_frame(&kernels, &device_frame, frame_index, palette)?;
        }
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;
        save_png(&args.output, &host_frame)?;
        println!(
            "saved Spectral Lattice preview after {frames} frame(s): {}",
            args.output.display()
        );
        return Ok(());
    }

    let mut window = Window::new(
        "ROCm-Oxide Spectral Lattice",
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
    let mut saved_once = false;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_index = (start.elapsed().as_millis() / 16) as u32;
        render_frame(&kernels, &device_frame, frame_index, palette)?;
        rocm_oxide::hip::synchronize()?;
        device_frame.copy_to_host(&mut host_frame)?;
        window.update_with_buffer(&host_frame, WIDTH, HEIGHT)?;

        if !saved_once || window.is_key_pressed(Key::S, KeyRepeat::No) {
            save_png(&args.output, &host_frame)?;
            saved_once = true;
        }
    }

    Ok(())
}

fn render_frame(
    kernels: &generated::DeviceKernels,
    device_frame: &DeviceBuffer<u32>,
    frame_index: u32,
    palette: [f32; 4],
) -> rocm_oxide::Result<()> {
    unsafe {
        kernels.spectral_lattice(
            LaunchConfig::for_num_elems_with_block_size(WIDTH * HEIGHT, BLOCK_X),
            device_frame,
            WIDTH as u32,
            HEIGHT as u32,
            WIDTH * HEIGHT,
            frame_index,
            palette[0],
            palette[1],
            palette[2],
            palette[3],
        )
    }
}

fn derive_palette() -> [f32; 4] {
    const FALLBACK: [f32; 4] = [1.1, 2.8, 4.6, 0.95];
    match derive_palette_with_rocblas() {
        Ok(palette) => palette,
        Err(err) => {
            eprintln!("rocBLAS palette seed skipped: {err}");
            FALLBACK
        }
    }
}

fn derive_palette_with_rocblas() -> rocm_oxide::Result<[f32; 4]> {
    let blas = RocBlas::open()?;
    let handle = blas.create_handle()?;
    let a = DeviceBuffer::from_slice(&[0.82f32, 0.27, 0.16, 0.91])?;
    let b = DeviceBuffer::from_slice(&[1.03f32, 0.21, 0.34, 0.79])?;
    let c = DeviceBuffer::<f32>::new(4)?;
    handle.sgemm_nn(SgemmLayout::column_major(2, 2, 2)?, 1.0, &a, &b, 0.0, &c)?;
    rocm_oxide::hip::synchronize()?;
    let out = c.copy_to_vec()?;
    Ok([
        phase(out[0] * 1.7),
        phase(out[1] * 2.1 + 0.4),
        phase(out[2] * 2.6 + 1.2),
        0.7 + out[3].abs().fract() * 0.9,
    ])
}

fn phase(value: f32) -> f32 {
    value.abs().fract() * std::f32::consts::TAU
}

fn save_png(path: &Path, frame: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut image = RgbImage::new(WIDTH as u32, HEIGHT as u32);
    for (index, pixel) in frame.iter().copied().enumerate() {
        let x = (index % WIDTH) as u32;
        let y = (index / WIDTH) as u32;
        image.put_pixel(
            x,
            y,
            Rgb([
                ((pixel >> 16) & 255) as u8,
                ((pixel >> 8) & 255) as u8,
                (pixel & 255) as u8,
            ]),
        );
    }
    image.save(path)?;
    Ok(())
}

fn parse_args() -> Result<DemoArgs, Box<dyn std::error::Error>> {
    let mut frames = None;
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--frames requires a frame count".to_string())?;
                frames = Some(value.parse::<u32>()?);
            }
            "--output" => {
                output = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--output requires a path".to_string())?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --example spectral_lattice -- [--frames N] [--output PATH]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument `{arg}`").into()),
        }
    }
    Ok(DemoArgs { frames, output })
}
