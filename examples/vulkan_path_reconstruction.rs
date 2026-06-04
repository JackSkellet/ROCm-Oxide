//! Vulkan path tracing + reconstruction demo for ROCm-Oxide.
//!
//! Drop this file into:
//!
//! ```text
//! examples/vulkan_path_reconstruction.rs
//! ```
//!
//! Run with Vulkan presentation:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_path_reconstruction
//! ```
//!
//! Optional bounded run:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_path_reconstruction -- --frames 300
//! ```
//!
//! What this demonstrates:
//!
//! ```text
//! HIPRTC path tracing kernel
//!         ↓
//! progressive GPU accumulation buffer
//!         ↓
//! GPU reconstruction / denoise / tonemap pass
//!         ↓
//! DeviceBuffer<u32>
//!         ↓
//! Vulkan presenter
//!         ↓
//! window
//! ```
//!
//! This is not NVIDIA/DLSS-style neural ray reconstruction. It is a compact
//! real-time reconstruction demo: progressive path tracing plus an edge-aware
//! spatial filter and filmic tonemap pass, all running on the GPU.
//!
//! Controls:
//!
//! - `Esc` closes the window.
//! - `Space` toggles reconstruction on/off.
//! - `R` resets accumulation.
//! - `Up` / `Down` changes exposure.
//! - `A` / `D` changes aperture and resets accumulation.
//! - `Left` / `Right` changes focus distance and resets accumulation.

use rocm_oxide::{Device, DeviceBuffer, LaunchConfig};
use std::time::Instant;

#[path = "shared/visual_presenter.rs"]
mod visual_presenter;

use visual_presenter::{requested_frames, Key, KeyRepeat, Scale, Window, WindowOptions};

const WIDTH: usize = 960;
const HEIGHT: usize = 540;
const PIXELS: usize = WIDTH * HEIGHT;
const ACCUM_FLOATS: usize = PIXELS * 4;

const PATH_RECONSTRUCTION_KERNELS: &str = r#"
struct V3 {
    float x;
    float y;
    float z;
};

struct Ray {
    V3 o;
    V3 d;
};

struct Hit {
    float t;
    V3 p;
    V3 n;
    V3 color;
    V3 emission;
    float roughness;
    int mat;
};

__device__ V3 v3(float x, float y, float z) {
    V3 r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

__device__ V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ V3 mul(V3 a, V3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ V3 muls(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ V3 divs(V3 a, float s) { return v3(a.x / s, a.y / s, a.z / s); }

__device__ float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ V3 cross3(V3 a, V3 b) {
    return v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float len3(V3 a) { return sqrtf(dot3(a, a)); }

__device__ V3 normalize3(V3 a) {
    float l = len3(a);
    if (l <= 1.0e-20f) {
        return v3(0.0f, 1.0f, 0.0f);
    }
    return divs(a, l);
}

__device__ V3 clamp3(V3 a, float lo, float hi) {
    return v3(
        fminf(hi, fmaxf(lo, a.x)),
        fminf(hi, fmaxf(lo, a.y)),
        fminf(hi, fmaxf(lo, a.z))
    );
}

__device__ V3 reflect3(V3 d, V3 n) {
    return sub(d, muls(n, 2.0f * dot3(d, n)));
}

__device__ unsigned int wang_hash(unsigned int x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

__device__ float rand01(unsigned int* state) {
    *state = wang_hash(*state);
    return (float)(*state & 0x00ffffffu) / 16777216.0f;
}

__device__ V3 random_in_unit_sphere(unsigned int* state) {
    for (int i = 0; i < 16; ++i) {
        V3 p = v3(
            rand01(state) * 2.0f - 1.0f,
            rand01(state) * 2.0f - 1.0f,
            rand01(state) * 2.0f - 1.0f
        );
        if (dot3(p, p) < 1.0f) {
            return p;
        }
    }
    return v3(0.0f, 1.0f, 0.0f);
}

__device__ V3 random_cosine_hemisphere(V3 n, unsigned int* state) {
    V3 p = normalize3(random_in_unit_sphere(state));
    if (dot3(p, n) < 0.0f) {
        p = muls(p, -1.0f);
    }
    return normalize3(add(n, p));
}

__device__ V3 sky_color(V3 d) {
    float t = 0.5f * (d.y + 1.0f);
    V3 horizon = v3(0.78f, 0.86f, 1.0f);
    V3 zenith = v3(0.06f, 0.10f, 0.20f);
    V3 c = add(muls(horizon, 1.0f - t), muls(zenith, t));

    float sun = fmaxf(0.0f, dot3(d, normalize3(v3(-0.35f, 0.58f, 0.72f))));
    c = add(c, muls(v3(1.0f, 0.82f, 0.42f), powf(sun, 320.0f) * 6.0f));
    c = add(c, muls(v3(1.0f, 0.58f, 0.22f), powf(sun, 16.0f) * 0.22f));
    return c;
}

__device__ bool hit_sphere(
    Ray ray,
    V3 center,
    float radius,
    V3 color,
    V3 emission,
    float roughness,
    int mat,
    float t_min,
    float t_max,
    Hit* best
) {
    V3 oc = sub(ray.o, center);
    float a = dot3(ray.d, ray.d);
    float half_b = dot3(oc, ray.d);
    float c = dot3(oc, oc) - radius * radius;
    float disc = half_b * half_b - a * c;
    if (disc < 0.0f) {
        return false;
    }

    float s = sqrtf(disc);
    float root = (-half_b - s) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + s) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    best->t = root;
    best->p = add(ray.o, muls(ray.d, root));
    best->n = divs(sub(best->p, center), radius);
    best->color = color;
    best->emission = emission;
    best->roughness = roughness;
    best->mat = mat;
    return true;
}

__device__ bool hit_plane(
    Ray ray,
    V3 point,
    V3 normal,
    V3 color,
    float roughness,
    int mat,
    float t_min,
    float t_max,
    Hit* best
) {
    float denom = dot3(ray.d, normal);
    if (fabsf(denom) < 1.0e-5f) {
        return false;
    }

    float t = dot3(sub(point, ray.o), normal) / denom;
    if (t < t_min || t > t_max) {
        return false;
    }

    V3 p = add(ray.o, muls(ray.d, t));

    // Checker for floor.
    if (normal.y > 0.8f) {
        int ix = (int)floorf(p.x * 1.25f);
        int iz = (int)floorf(p.z * 1.25f);
        int checker = (ix ^ iz) & 1;
        color = checker ? v3(0.78f, 0.76f, 0.68f) : v3(0.18f, 0.18f, 0.20f);
    }

    best->t = t;
    best->p = p;
    best->n = normal;
    best->color = color;
    best->emission = v3(0.0f, 0.0f, 0.0f);
    best->roughness = roughness;
    best->mat = mat;
    return true;
}

__device__ bool scene_intersect(Ray ray, Hit* hit) {
    bool any = false;
    float closest = 1.0e20f;
    Hit h;

    if (hit_plane(ray, v3(0.0f, -0.75f, 0.0f), v3(0.0f, 1.0f, 0.0f),
                  v3(0.75f, 0.75f, 0.72f), 0.88f, 0, 0.001f, closest, &h)) {
        any = true;
        closest = h.t;
        *hit = h;
    }

    if (hit_plane(ray, v3(0.0f, 0.0f, -3.45f), v3(0.0f, 0.0f, 1.0f),
                  v3(0.50f, 0.57f, 0.70f), 0.82f, 0, 0.001f, closest, &h)) {
        any = true;
        closest = h.t;
        *hit = h;
    }

    if (hit_sphere(ray, v3(-1.05f, -0.10f, -0.55f), 0.63f,
                   v3(0.95f, 0.62f, 0.38f), v3(0.0f, 0.0f, 0.0f), 0.045f, 1,
                   0.001f, closest, &h)) {
        any = true;
        closest = h.t;
        *hit = h;
    }

    if (hit_sphere(ray, v3(0.72f, -0.18f, -0.35f), 0.55f,
                   v3(0.62f, 0.82f, 1.00f), v3(0.0f, 0.0f, 0.0f), 0.018f, 2,
                   0.001f, closest, &h)) {
        any = true;
        closest = h.t;
        *hit = h;
    }

    if (hit_sphere(ray, v3(0.10f, -0.43f, 0.84f), 0.32f,
                   v3(0.72f, 0.12f, 0.09f), v3(0.0f, 0.0f, 0.0f), 0.76f, 0,
                   0.001f, closest, &h)) {
        any = true;
        closest = h.t;
        *hit = h;
    }

    // Emissive soft light sphere.
    if (hit_sphere(ray, v3(0.0f, 2.65f, -1.25f), 0.62f,
                   v3(1.0f, 0.86f, 0.55f), v3(8.0f, 6.2f, 3.5f), 0.0f, 3,
                   0.001f, closest, &h)) {
        any = true;
        closest = h.t;
        *hit = h;
    }

    return any;
}

__device__ V3 direct_light(Hit hit, unsigned int* rng) {
    V3 light_center = v3(0.0f, 2.65f, -1.25f);
    float light_radius = 0.62f;

    V3 jitter = muls(random_in_unit_sphere(rng), light_radius);
    V3 target = add(light_center, jitter);
    V3 to_light = sub(target, hit.p);
    float dist2 = fmaxf(0.001f, dot3(to_light, to_light));
    float dist = sqrtf(dist2);
    V3 ldir = divs(to_light, dist);

    float n_dot_l = fmaxf(0.0f, dot3(hit.n, ldir));
    if (n_dot_l <= 0.0f) {
        return v3(0.0f, 0.0f, 0.0f);
    }

    Ray shadow;
    shadow.o = add(hit.p, muls(hit.n, 0.003f));
    shadow.d = ldir;

    Hit blocker;
    if (scene_intersect(shadow, &blocker)) {
        if (blocker.t < dist - 0.04f && dot3(blocker.emission, blocker.emission) <= 0.0f) {
            return v3(0.0f, 0.0f, 0.0f);
        }
    }

    V3 light_power = v3(8.0f, 6.2f, 3.5f);
    float attenuation = (light_radius * light_radius * 5.0f) / dist2;
    return muls(light_power, n_dot_l * attenuation);
}

__device__ V3 trace_path(Ray ray, unsigned int* rng) {
    V3 radiance = v3(0.0f, 0.0f, 0.0f);
    V3 throughput = v3(1.0f, 1.0f, 1.0f);

    for (int bounce = 0; bounce < 5; ++bounce) {
        Hit hit;
        if (!scene_intersect(ray, &hit)) {
            radiance = add(radiance, mul(throughput, sky_color(ray.d)));
            break;
        }

        if (dot3(hit.emission, hit.emission) > 0.0f) {
            radiance = add(radiance, mul(throughput, hit.emission));
            break;
        }

        V3 dl = direct_light(hit, rng);
        radiance = add(radiance, mul(throughput, mul(hit.color, dl)));

        if (hit.mat == 1) {
            // Rough metal.
            V3 refl = reflect3(ray.d, hit.n);
            V3 fuzz = muls(random_in_unit_sphere(rng), hit.roughness);
            ray.o = add(hit.p, muls(hit.n, 0.003f));
            ray.d = normalize3(add(refl, fuzz));
            throughput = mul(throughput, hit.color);
        } else if (hit.mat == 2) {
            // Fake glass/clearcoat: mostly reflection, sometimes diffuse caustic tint.
            V3 refl = reflect3(ray.d, hit.n);
            float facing = fmaxf(0.0f, -dot3(ray.d, hit.n));
            float fresnel = 0.04f + 0.96f * powf(1.0f - facing, 5.0f);
            if (rand01(rng) < fresnel + 0.22f) {
                ray.o = add(hit.p, muls(hit.n, 0.003f));
                ray.d = normalize3(add(refl, muls(random_in_unit_sphere(rng), hit.roughness)));
                throughput = mul(throughput, v3(0.82f, 0.93f, 1.0f));
            } else {
                ray.o = add(hit.p, muls(hit.n, 0.003f));
                ray.d = random_cosine_hemisphere(hit.n, rng);
                throughput = mul(throughput, muls(hit.color, 0.72f));
            }
        } else {
            // Diffuse.
            ray.o = add(hit.p, muls(hit.n, 0.003f));
            ray.d = random_cosine_hemisphere(hit.n, rng);
            throughput = mul(throughput, muls(hit.color, 0.82f));
        }

        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (bounce >= 3) {
            if (rand01(rng) > p) {
                break;
            }
            throughput = divs(throughput, fmaxf(p, 0.05f));
        }
    }

    return radiance;
}

extern "C" __global__ void clear_accum(float* accum, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        accum[i] = 0.0f;
    }
}

extern "C" __global__ void trace_sample(
    float* accum,
    unsigned int width,
    unsigned int height,
    unsigned int sample_index,
    float aperture,
    float focus_dist
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = width * height;
    if (i >= n) {
        return;
    }

    unsigned int x = i % width;
    unsigned int y = i / width;

    unsigned int rng = wang_hash(i ^ (sample_index * 747796405u) ^ 0x9e3779b9u);

    float jx = rand01(&rng) - 0.5f;
    float jy = rand01(&rng) - 0.5f;

    float aspect = (float)width / (float)height;
    float fov = 42.0f * 0.017453292519943295f;
    float scale = tanf(fov * 0.5f);

    float sx = (((float)x + 0.5f + jx) / (float)width * 2.0f - 1.0f) * aspect * scale;
    float sy = (1.0f - ((float)y + 0.5f + jy) / (float)height * 2.0f) * scale;

    V3 cam_pos = v3(0.0f, 0.50f, 4.45f);
    V3 target = v3(-0.08f, 0.05f, -0.75f);
    V3 forward = normalize3(sub(target, cam_pos));
    V3 right = normalize3(cross3(forward, v3(0.0f, 1.0f, 0.0f)));
    V3 up = normalize3(cross3(right, forward));

    V3 sensor_dir = normalize3(add(forward, add(muls(right, sx), muls(up, sy))));

    // Depth of field.
    float lens_r = aperture * sqrtf(rand01(&rng));
    float lens_a = 6.28318530718f * rand01(&rng);
    V3 lens_offset = add(
        muls(right, cosf(lens_a) * lens_r),
        muls(up, sinf(lens_a) * lens_r)
    );

    V3 focal_point = add(cam_pos, muls(sensor_dir, focus_dist / fmaxf(0.05f, dot3(sensor_dir, forward))));

    Ray ray;
    ray.o = add(cam_pos, lens_offset);
    ray.d = normalize3(sub(focal_point, ray.o));

    V3 color = trace_path(ray, &rng);

    // Firefly clamp for stable reconstruction.
    color = clamp3(color, 0.0f, 12.0f);

    unsigned int base = i * 4u;
    accum[base + 0u] += color.x;
    accum[base + 1u] += color.y;
    accum[base + 2u] += color.z;
    accum[base + 3u] += 1.0f;
}

__device__ V3 accum_color(const float* accum, int x, int y, int width, int height) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    unsigned int i = (unsigned int)y * (unsigned int)width + (unsigned int)x;
    unsigned int base = i * 4u;
    float samples = fmaxf(1.0f, accum[base + 3u]);
    return v3(accum[base + 0u] / samples, accum[base + 1u] / samples, accum[base + 2u] / samples);
}

__device__ float luma(V3 c) {
    return dot3(c, v3(0.2126f, 0.7152f, 0.0722f));
}

__device__ V3 tonemap(V3 c, float exposure) {
    c = muls(c, exposure);

    // ACES-ish fit.
    V3 a = muls(c, 2.51f);
    V3 b = add(muls(c, 0.03f), v3(0.0f, 0.0f, 0.0f));
    V3 numerator = mul(c, add(a, b));

    V3 denominator = add(mul(c, add(muls(c, 2.43f), v3(0.59f, 0.59f, 0.59f))), v3(0.14f, 0.14f, 0.14f));
    c = v3(numerator.x / denominator.x, numerator.y / denominator.y, numerator.z / denominator.z);

    c = clamp3(c, 0.0f, 1.0f);
    c.x = powf(c.x, 1.0f / 2.2f);
    c.y = powf(c.y, 1.0f / 2.2f);
    c.z = powf(c.z, 1.0f / 2.2f);
    return c;
}

extern "C" __global__ void reconstruct_frame(
    unsigned int* frame,
    const float* accum,
    unsigned int width,
    unsigned int height,
    unsigned int reconstruction_enabled,
    float exposure,
    unsigned int frame_index
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = width * height;
    if (i >= n) {
        return;
    }

    int x = (int)(i % width);
    int y = (int)(i / width);

    V3 center = accum_color(accum, x, y, (int)width, (int)height);
    V3 color = center;

    if (reconstruction_enabled != 0u) {
        float center_luma = luma(center);
        V3 sum = muls(center, 1.20f);
        float wsum = 1.20f;

        // Edge-aware 5x5 reconstruction. This is deliberately simple: it is not
        // neural, but it demonstrates the ray-reconstruction idea in a compact way.
        for (int oy = -2; oy <= 2; ++oy) {
            for (int ox = -2; ox <= 2; ++ox) {
                if (ox == 0 && oy == 0) {
                    continue;
                }

                V3 c = accum_color(accum, x + ox, y + oy, (int)width, (int)height);
                float dl = fabsf(luma(c) - center_luma);
                float spatial = 1.0f / (1.0f + (float)(ox * ox + oy * oy));
                float edge = 1.0f / (1.0f + 18.0f * dl);
                float w = spatial * edge;

                sum = add(sum, muls(c, w));
                wsum += w;
            }
        }

        V3 filtered = divs(sum, wsum);

        // Preserve crisp specular/high-frequency detail while reducing noise.
        float detail = fminf(1.0f, fabsf(luma(center) - luma(filtered)) * 3.5f);
        color = add(muls(filtered, 0.78f - 0.28f * detail), muls(center, 0.22f + 0.28f * detail));
    }

    color = tonemap(color, exposure);

    // Subtle film grain after reconstruction so the output stays alive.
    unsigned int rng = wang_hash(i ^ (frame_index * 9781u));
    float grain = (rand01(&rng) - 0.5f) * 0.018f;
    color = clamp3(add(color, v3(grain, grain, grain)), 0.0f, 1.0f);

    // Small HUD strips: left = raw/reconstructed state, bottom = sample progress.
    float samples = fmaxf(1.0f, accum[i * 4u + 3u]);
    if (x < 8) {
        if (reconstruction_enabled != 0u) {
            color = v3(0.2f, 0.9f, 1.0f);
        } else {
            color = v3(1.0f, 0.45f, 0.10f);
        }
    }
    if (y > (int)height - 9) {
        float progress = fminf(1.0f, samples / 240.0f);
        if ((float)x / (float)width < progress) {
            color = v3(0.95f, 0.80f, 0.35f);
        }
    }

    unsigned int r = (unsigned int)(fminf(1.0f, color.x) * 255.0f);
    unsigned int g = (unsigned int)(fminf(1.0f, color.y) * 255.0f);
    unsigned int b = (unsigned int)(fminf(1.0f, color.z) * 255.0f);

    frame[i] = (r << 16) | (g << 8) | b;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ROCM_OXIDE_VISUAL_PRESENT").as_deref() != Ok("vulkan") {
        eprintln!(
            "note: run with `ROCM_OXIDE_VISUAL_PRESENT=vulkan` to exercise Vulkan presentation"
        );
    }

    let device = Device::first()?;
    let module = device.compile_hip_source(PATH_RECONSTRUCTION_KERNELS)?;
    let clear_accum = module.kernel(c"clear_accum")?;
    let trace_sample = module.kernel(c"trace_sample")?;
    let reconstruct_frame = module.kernel(c"reconstruct_frame")?;

    let accum = DeviceBuffer::<f32>::new(ACCUM_FLOATS)?;
    let frame = DeviceBuffer::<u32>::new(PIXELS)?;

    unsafe {
        rocm_oxide::launch!(
            clear_accum,
            LaunchConfig::for_num_elems_with_block_size(ACCUM_FLOATS, 256),
            accum.as_mut_ptr(),
            ACCUM_FLOATS as u32,
        )?;
    }
    rocm_oxide::hip::synchronize()?;

    let mut window = Window::new(
        "ROCm-Oxide Vulkan Path Reconstruction",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
        },
    )?;
    window.set_target_fps(60);

    let max_frames = requested_frames("ROCM_OXIDE_PATH_RECONSTRUCTION_FRAMES");

    let mut rendered_frames = 0u32;
    let mut sample_index = 0u32;
    let mut reconstruction = true;
    let mut exposure = 1.15f32;
    let mut aperture = 0.030f32;
    let mut focus_dist = 4.55f32;

    println!("vulkan_path_reconstruction: HIPRTC path tracer + GPU reconstruction");
    println!("controls: Esc quit | Space reconstruction | R reset | Up/Down exposure | A/D aperture | Left/Right focus");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut reset = false;

        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            reconstruction = !reconstruction;
        }

        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            reset = true;
        }

        if window.is_key_down(Key::Up) {
            exposure = (exposure * 1.012).min(3.0);
        }
        if window.is_key_down(Key::Down) {
            exposure = (exposure * 0.988).max(0.25);
        }

        if window.is_key_down(Key::D) {
            aperture = (aperture * 1.020).min(0.12);
            reset = true;
        }
        if window.is_key_down(Key::A) {
            aperture = (aperture * 0.980).max(0.001);
            reset = true;
        }

        if window.is_key_down(Key::Right) {
            focus_dist = (focus_dist * 1.010).min(7.0);
            reset = true;
        }
        if window.is_key_down(Key::Left) {
            focus_dist = (focus_dist * 0.990).max(2.0);
            reset = true;
        }

        if reset {
            unsafe {
                rocm_oxide::launch!(
                    clear_accum,
                    LaunchConfig::for_num_elems_with_block_size(ACCUM_FLOATS, 256),
                    accum.as_mut_ptr(),
                    ACCUM_FLOATS as u32,
                )?;
            }
            sample_index = 0;
        }

        unsafe {
            rocm_oxide::launch!(
                trace_sample,
                LaunchConfig::for_num_elems_with_block_size(PIXELS, 256),
                accum.as_mut_ptr(),
                WIDTH as u32,
                HEIGHT as u32,
                sample_index,
                aperture,
                focus_dist,
            )?;

            rocm_oxide::launch!(
                reconstruct_frame,
                LaunchConfig::for_num_elems_with_block_size(PIXELS, 256),
                frame.as_mut_ptr(),
                accum.as_ptr(),
                WIDTH as u32,
                HEIGHT as u32,
                reconstruction as u32,
                exposure,
                rendered_frames,
            )?;
        }

        rocm_oxide::hip::synchronize()?;
        window.update_with_device_buffer(&frame, WIDTH, HEIGHT)?;

        if rendered_frames % 30 == 0 {
            window.set_title(&format!(
                "ROCm-Oxide Vulkan Path Reconstruction | samples={} recon={} exposure={:.2} aperture={:.3} focus={:.2}",
                sample_index + 1,
                reconstruction,
                exposure,
                aperture,
                focus_dist,
            ));
        }

        rendered_frames = rendered_frames.wrapping_add(1);
        sample_index = sample_index.wrapping_add(1);

        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    println!(
        "vulkan_path_reconstruction: rendered {} frame(s), accumulated {} sample(s)",
        rendered_frames, sample_index
    );

    Ok(())
}
