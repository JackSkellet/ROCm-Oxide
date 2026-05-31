use image::{Rgb, RgbImage};
use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::Instant;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const WIDTH: usize = 1024;
const HEIGHT: usize = 576;
const SPHERE_STRIDE: usize = 8;
const NODE_STRIDE: usize = 8;
const HEADER_LEN: usize = 8;

#[derive(Clone, Copy, Debug)]
struct Sphere {
    center: [f32; 3],
    radius: f32,
    color: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    left_or_first: u32,
    right_or_neg_count: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load(&device, env!("ROCM_OXIDE_DEVICE_HSACO"))?;

    let source_spheres = make_spheres();
    let (ordered_spheres, nodes) = build_bvh(&source_spheres);
    let scene = pack_scene(&ordered_spheres, &nodes);

    let device_scene = DeviceBuffer::<f32>::from_slice(&scene)?;
    let device_frame = DeviceBuffer::<u32>::new(WIDTH * HEIGHT)?;
    let mut brute = vec![0u32; WIDTH * HEIGHT];
    let mut bvh = vec![0u32; WIDTH * HEIGHT];
    let config = LaunchConfig::for_num_elems(WIDTH * HEIGHT, 256);

    let brute_ms = benchmark_mode(
        &kernels,
        config,
        &device_frame,
        &device_scene,
        0,
        &mut brute,
    )?;
    let bvh_ms = benchmark_mode(&kernels, config, &device_frame, &device_scene, 1, &mut bvh)?;

    let mismatches = brute
        .iter()
        .zip(&bvh)
        .filter(|(a, b)| a.abs_diff(**b) > 0)
        .count();
    assert_eq!(mismatches, 0, "BVH and brute-force images diverged");

    save_rgb(&brute, "target/bvh_raytrace_brute.png")?;
    save_rgb(&bvh, "target/bvh_raytrace_bvh.png")?;

    println!(
        "BVH raytrace benchmark on {}: {} spheres, {} nodes, brute {:.3} ms, bvh {:.3} ms, speedup {:.2}x",
        device.arch(),
        ordered_spheres.len(),
        nodes.len(),
        brute_ms,
        bvh_ms,
        brute_ms / bvh_ms.max(0.001),
    );
    Ok(())
}

fn benchmark_mode(
    kernels: &generated::DeviceKernels,
    config: LaunchConfig,
    frame: &DeviceBuffer<u32>,
    scene: &DeviceBuffer<f32>,
    mode: u32,
    output: &mut [u32],
) -> rocm_oxide::Result<f64> {
    const ITERS: usize = 12;
    unsafe {
        kernels.bvh_raytrace(config, frame, scene, WIDTH * HEIGHT, mode, 256)?;
    }
    rocm_oxide::hip::synchronize()?;

    let start = Instant::now();
    for _ in 0..ITERS {
        unsafe {
            kernels.bvh_raytrace(config, frame, scene, WIDTH * HEIGHT, mode, 256)?;
        }
    }
    rocm_oxide::hip::synchronize()?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    frame.copy_to_host(output)?;
    Ok(elapsed)
}

fn make_spheres() -> Vec<Sphere> {
    let mut spheres = Vec::new();
    for z in 0..8 {
        for x in 0..16 {
            let fx = x as f32 - 7.5;
            let fz = z as f32;
            let h = hash32((x as u32).wrapping_mul(977) ^ (z as u32).wrapping_mul(131));
            let radius = 0.16 + ((h & 31) as f32) * 0.003;
            let y = -0.92 + ((h >> 5) & 31) as f32 * 0.018;
            spheres.push(Sphere {
                center: [fx * 0.42, y, 1.5 + fz * 0.7],
                radius,
                color: [
                    0.18 + ((h >> 10) & 127) as f32 / 180.0,
                    0.18 + ((h >> 17) & 127) as f32 / 180.0,
                    0.25 + ((h >> 24) & 127) as f32 / 170.0,
                ],
            });
        }
    }

    for i in 0..24 {
        let h = hash32(9_999 + i as u32 * 771);
        spheres.push(Sphere {
            center: [
                -2.8 + (i % 8) as f32 * 0.8,
                -0.25 + ((h >> 8) & 15) as f32 * 0.06,
                3.0 + (i / 8) as f32 * 1.4,
            ],
            radius: 0.28 + ((h & 7) as f32) * 0.025,
            color: [
                0.45 + ((h >> 11) & 63) as f32 / 160.0,
                0.2 + ((h >> 18) & 63) as f32 / 180.0,
                0.25 + ((h >> 25) & 63) as f32 / 150.0,
            ],
        });
    }

    spheres
}

fn build_bvh(spheres: &[Sphere]) -> (Vec<Sphere>, Vec<Node>) {
    let mut indices = (0..spheres.len()).collect::<Vec<_>>();
    let mut ordered = Vec::with_capacity(spheres.len());
    let mut nodes = Vec::new();
    build_node(&mut indices, spheres, &mut ordered, &mut nodes);
    (ordered, nodes)
}

fn build_node(
    indices: &mut [usize],
    spheres: &[Sphere],
    ordered: &mut Vec<Sphere>,
    nodes: &mut Vec<Node>,
) -> usize {
    let node_index = nodes.len();
    nodes.push(Node {
        min: [0.0; 3],
        max: [0.0; 3],
        left_or_first: 0,
        right_or_neg_count: 0.0,
    });

    let (min, max) = bounds(indices, spheres);
    if indices.len() <= 4 {
        let first = ordered.len() as u32;
        for &index in indices.iter() {
            ordered.push(spheres[index]);
        }
        nodes[node_index] = Node {
            min,
            max,
            left_or_first: first,
            right_or_neg_count: -(indices.len() as f32),
        };
        return node_index;
    }

    let axis = longest_axis(indices, spheres);
    indices.sort_by(|a, b| spheres[*a].center[axis].total_cmp(&spheres[*b].center[axis]));
    let mid = indices.len() / 2;
    let (left_indices, right_indices) = indices.split_at_mut(mid);
    let left = build_node(left_indices, spheres, ordered, nodes);
    let right = build_node(right_indices, spheres, ordered, nodes);
    nodes[node_index] = Node {
        min,
        max,
        left_or_first: left as u32,
        right_or_neg_count: right as f32,
    };
    node_index
}

fn bounds(indices: &[usize], spheres: &[Sphere]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &index in indices {
        let sphere = spheres[index];
        for axis in 0..3 {
            min[axis] = min[axis].min(sphere.center[axis] - sphere.radius);
            max[axis] = max[axis].max(sphere.center[axis] + sphere.radius);
        }
    }
    (min, max)
}

fn longest_axis(indices: &[usize], spheres: &[Sphere]) -> usize {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &index in indices {
        for axis in 0..3 {
            min[axis] = min[axis].min(spheres[index].center[axis]);
            max[axis] = max[axis].max(spheres[index].center[axis]);
        }
    }
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    }
}

fn pack_scene(spheres: &[Sphere], nodes: &[Node]) -> Vec<f32> {
    let node_offset = HEADER_LEN + spheres.len() * SPHERE_STRIDE;
    let mut out = vec![0.0f32; node_offset + nodes.len() * NODE_STRIDE];
    out[0] = spheres.len() as f32;
    out[1] = nodes.len() as f32;
    out[2] = node_offset as f32;

    for (index, sphere) in spheres.iter().enumerate() {
        let base = HEADER_LEN + index * SPHERE_STRIDE;
        out[base] = sphere.center[0];
        out[base + 1] = sphere.center[1];
        out[base + 2] = sphere.center[2];
        out[base + 3] = sphere.radius;
        out[base + 4] = sphere.color[0];
        out[base + 5] = sphere.color[1];
        out[base + 6] = sphere.color[2];
    }

    for (index, node) in nodes.iter().enumerate() {
        let base = node_offset + index * NODE_STRIDE;
        out[base] = node.min[0];
        out[base + 1] = node.min[1];
        out[base + 2] = node.min[2];
        out[base + 3] = node.max[0];
        out[base + 4] = node.max[1];
        out[base + 5] = node.max[2];
        out[base + 6] = node.left_or_first as f32;
        out[base + 7] = node.right_or_neg_count;
    }
    out
}

fn save_rgb(pixels: &[u32], path: &str) -> Result<(), image::ImageError> {
    let mut image = RgbImage::new(WIDTH as u32, HEIGHT as u32);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let rgb = pixels[y * WIDTH + x];
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

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
