//! Interactive Vulkan path tracing + reconstruction demo for ROCm-Oxide.
//!
//! Drop this file into:
//!
//! ```text
//! examples/vulkan_interactive_path_reconstruction.rs
//! ```
//!
//! Run:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_interactive_path_reconstruction
//! ```
//!
//! Bounded run:
//!
//! ```sh
//! ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run --example vulkan_interactive_path_reconstruction -- --frames 300
//! ```
//!
//! What this demonstrates:
//!
//! ```text
//! interactive camera / object controls
//!         ↓
//! HIPRTC path tracing kernel
//!         ↓
//! progressive GPU accumulation buffer
//!         ↓
//! GPU edge-aware reconstruction / denoise / tonemap pass
//!         ↓
//! DeviceBuffer<u32>
//!         ↓
//! Vulkan presenter
//! ```
//!
//! This is not neural/DLSS-style ray reconstruction. It is a compact GPU
//! reconstruction demo: progressive path tracing plus an edge-aware spatial
//! reconstruction pass and filmic tonemapping.
//!
//! This version also reduces motion noise by tracing extra samples per frame
//! while the camera or scene is changing, then switching back to normal
//! progressive accumulation once the view is stable.
//!
//! Controls:
//!
//! Camera:
//! - `W` / `S` move forward/back
//! - `A` / `D` strafe left/right
//! - `Left` / `Right` yaw camera
//! - `Up` / `Down` pitch camera
//!
//! Scene:
//! - `1`, `2`, `3`, `4` select object/light
//! - `5` / `6` move selected object/light left/right
//! - `7` / `8` move selected object/light forward/back
//! - `PageUp` / `PageDown` move selected object/light up/down
//!
//! Render:
//! - `Space` toggle reconstruction
//! - `R` reset accumulation
//! - `P` pause/resume animation
//! - `C` cycle material preset
//! - `9` / `0` exposure down/up
//! - `LeftShift` hold for faster movement/adjustments
//! - `Esc` quit

use rocm_oxide::{Device, DeviceBuffer, DevicePod, LaunchConfig};
use std::f32::consts::PI;
use std::time::Instant;

#[path = "shared/visual_presenter.rs"]
mod visual_presenter;

use visual_presenter::{requested_frames, Key, KeyRepeat, Scale, Window, WindowOptions};

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const PIXELS: usize = WIDTH * HEIGHT;
const ACCUM_FLOATS: usize = PIXELS * 4;

const KERNELS: &str = r#"
struct V3 { float x; float y; float z; };
struct Ray { V3 o; V3 d; };

struct Scene {
    V3 s0;
    V3 s1;
    V3 s2;
    V3 light;
    unsigned int material_preset;
};

struct RenderParams {
    unsigned int width;
    unsigned int height;
    unsigned int sample_index;
    unsigned int material_preset;
    unsigned int moving_preview;
    V3 cam;
    V3 forward;
    V3 right;
    V3 up;
    float aperture;
    float focus_dist;
    V3 s0;
    V3 s1;
    V3 s2;
    V3 light;
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

__device__ V3 v3(float x, float y, float z) { V3 r; r.x=x; r.y=y; r.z=z; return r; }
__device__ V3 add(V3 a,V3 b){return v3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ V3 sub(V3 a,V3 b){return v3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ V3 mul(V3 a,V3 b){return v3(a.x*b.x,a.y*b.y,a.z*b.z);}
__device__ V3 muls(V3 a,float s){return v3(a.x*s,a.y*s,a.z*s);}
__device__ V3 divs(V3 a,float s){return v3(a.x/s,a.y/s,a.z/s);}
__device__ float dot3(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ V3 cross3(V3 a,V3 b){return v3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float len3(V3 a){return sqrtf(dot3(a,a));}
__device__ V3 normalize3(V3 a){float l=len3(a); return l>1e-20f?divs(a,l):v3(0,1,0);}
__device__ V3 clamp3(V3 a,float lo,float hi){return v3(fminf(hi,fmaxf(lo,a.x)),fminf(hi,fmaxf(lo,a.y)),fminf(hi,fmaxf(lo,a.z)));}
__device__ V3 reflect3(V3 d,V3 n){return sub(d,muls(n,2.0f*dot3(d,n)));}

__device__ unsigned int wang_hash(unsigned int x){
    x=(x^61u)^(x>>16); x*=9u; x=x^(x>>4); x*=0x27d4eb2du; x=x^(x>>15); return x;
}
__device__ float rand01(unsigned int* s){*s=wang_hash(*s); return (float)(*s&0x00ffffffu)/16777216.0f;}

__device__ V3 random_in_unit_sphere(unsigned int* s){
    for(int i=0;i<16;++i){
        V3 p=v3(rand01(s)*2.0f-1.0f,rand01(s)*2.0f-1.0f,rand01(s)*2.0f-1.0f);
        if(dot3(p,p)<1.0f) return p;
    }
    return v3(0,1,0);
}

__device__ V3 random_cosine_hemisphere(V3 n,unsigned int* s){
    V3 p=normalize3(random_in_unit_sphere(s));
    if(dot3(p,n)<0.0f) p=muls(p,-1.0f);
    return normalize3(add(n,p));
}

__device__ V3 sky_color(V3 d){
    float t=0.5f*(d.y+1.0f);
    V3 horizon=v3(0.78f,0.86f,1.0f);
    V3 zenith=v3(0.045f,0.070f,0.145f);
    V3 c=add(muls(horizon,1.0f-t),muls(zenith,t));
    float sun=fmaxf(0.0f,dot3(d,normalize3(v3(-0.35f,0.58f,0.72f))));
    c=add(c,muls(v3(1.0f,0.82f,0.42f),powf(sun,360.0f)*7.0f));
    c=add(c,muls(v3(1.0f,0.58f,0.22f),powf(sun,16.0f)*0.22f));
    return c;
}

__device__ bool hit_sphere(Ray ray,V3 center,float radius,V3 color,V3 emission,float roughness,int mat,float tmin,float tmax,Hit* best){
    V3 oc=sub(ray.o,center);
    float a=dot3(ray.d,ray.d);
    float hb=dot3(oc,ray.d);
    float c=dot3(oc,oc)-radius*radius;
    float disc=hb*hb-a*c;
    if(disc<0.0f) return false;
    float sq=sqrtf(disc);
    float root=(-hb-sq)/a;
    if(root<tmin||root>tmax){
        root=(-hb+sq)/a;
        if(root<tmin||root>tmax) return false;
    }
    best->t=root;
    best->p=add(ray.o,muls(ray.d,root));
    best->n=divs(sub(best->p,center),radius);
    best->color=color;
    best->emission=emission;
    best->roughness=roughness;
    best->mat=mat;
    return true;
}

__device__ bool hit_plane(Ray ray,V3 point,V3 normal,V3 color,float roughness,int mat,float tmin,float tmax,Hit* best){
    float denom=dot3(ray.d,normal);
    if(fabsf(denom)<1e-5f) return false;
    float t=dot3(sub(point,ray.o),normal)/denom;
    if(t<tmin||t>tmax) return false;
    V3 p=add(ray.o,muls(ray.d,t));

    if(normal.y>0.8f){
        int ix=(int)floorf(p.x*1.25f);
        int iz=(int)floorf(p.z*1.25f);
        int checker=(ix^iz)&1;
        color=checker?v3(0.78f,0.76f,0.68f):v3(0.18f,0.18f,0.20f);
    }

    best->t=t; best->p=p; best->n=normal; best->color=color; best->emission=v3(0,0,0); best->roughness=roughness; best->mat=mat;
    return true;
}

__device__ bool scene_intersect(Ray ray,Scene scene,Hit* hit){
    bool any=false;
    float closest=1e20f;
    Hit h;

    if(hit_plane(ray,v3(0,-0.75f,0),v3(0,1,0),v3(0.75f,0.75f,0.72f),0.88f,0,0.001f,closest,&h)){any=true;closest=h.t;*hit=h;}
    if(hit_plane(ray,v3(0,0,-3.75f),v3(0,0,1),v3(0.50f,0.57f,0.70f),0.82f,0,0.001f,closest,&h)){any=true;closest=h.t;*hit=h;}

    unsigned int p=scene.material_preset&3u;
    V3 c0 = p==0u ? v3(0.95f,0.62f,0.38f) : p==1u ? v3(1.0f,0.72f,0.55f) : p==2u ? v3(0.90f,0.85f,0.72f) : v3(0.75f,0.95f,0.95f);
    V3 c1 = p==0u ? v3(0.62f,0.82f,1.00f) : p==1u ? v3(0.60f,1.00f,0.82f) : p==2u ? v3(0.95f,0.95f,1.00f) : v3(1.00f,0.50f,0.80f);
    V3 c2 = p==0u ? v3(0.72f,0.12f,0.09f) : p==1u ? v3(0.85f,0.20f,1.00f) : p==2u ? v3(0.70f,0.35f,0.12f) : v3(0.25f,1.00f,0.45f);

    if(hit_sphere(ray,scene.s0,0.63f,c0,v3(0,0,0),0.045f,1,0.001f,closest,&h)){any=true;closest=h.t;*hit=h;}
    if(hit_sphere(ray,scene.s1,0.55f,c1,v3(0,0,0),0.018f,2,0.001f,closest,&h)){any=true;closest=h.t;*hit=h;}
    if(hit_sphere(ray,scene.s2,0.32f,c2,v3(0,0,0),0.76f,0,0.001f,closest,&h)){any=true;closest=h.t;*hit=h;}
    if(hit_sphere(ray,scene.light,0.58f,v3(1.0f,0.86f,0.55f),v3(8.5f,6.4f,3.6f),0.0f,3,0.001f,closest,&h)){any=true;closest=h.t;*hit=h;}

    return any;
}

__device__ V3 direct_light(Hit hit,Scene scene,unsigned int* rng){
    float light_radius=0.58f;
    V3 jitter=muls(random_in_unit_sphere(rng),light_radius);
    V3 target=add(scene.light,jitter);
    V3 to_light=sub(target,hit.p);
    float dist2=fmaxf(0.001f,dot3(to_light,to_light));
    float dist=sqrtf(dist2);
    V3 ldir=divs(to_light,dist);
    float ndl=fmaxf(0.0f,dot3(hit.n,ldir));
    if(ndl<=0.0f) return v3(0,0,0);

    Ray shadow; shadow.o=add(hit.p,muls(hit.n,0.003f)); shadow.d=ldir;
    Hit blocker;
    if(scene_intersect(shadow,scene,&blocker)){
        if(blocker.t<dist-0.04f && dot3(blocker.emission,blocker.emission)<=0.0f) return v3(0,0,0);
    }

    float attenuation=(light_radius*light_radius*5.0f)/dist2;
    return muls(v3(8.5f,6.4f,3.6f),ndl*attenuation);
}

__device__ V3 trace_path(Ray ray,Scene scene,unsigned int* rng){
    V3 radiance=v3(0,0,0);
    V3 throughput=v3(1,1,1);

    for(int bounce=0;bounce<5;++bounce){
        Hit hit;
        if(!scene_intersect(ray,scene,&hit)){
            radiance=add(radiance,mul(throughput,sky_color(ray.d)));
            break;
        }

        if(dot3(hit.emission,hit.emission)>0.0f){
            radiance=add(radiance,mul(throughput,hit.emission));
            break;
        }

        radiance=add(radiance,mul(throughput,mul(hit.color,direct_light(hit,scene,rng))));

        if(hit.mat==1){
            V3 refl=reflect3(ray.d,hit.n);
            V3 fuzz=muls(random_in_unit_sphere(rng),hit.roughness);
            ray.o=add(hit.p,muls(hit.n,0.003f));
            ray.d=normalize3(add(refl,fuzz));
            throughput=mul(throughput,hit.color);
        } else if(hit.mat==2){
            V3 refl=reflect3(ray.d,hit.n);
            float facing=fmaxf(0.0f,-dot3(ray.d,hit.n));
            float fresnel=0.04f+0.96f*powf(1.0f-facing,5.0f);
            if(rand01(rng)<fresnel+0.22f){
                ray.o=add(hit.p,muls(hit.n,0.003f));
                ray.d=normalize3(add(refl,muls(random_in_unit_sphere(rng),hit.roughness)));
                throughput=mul(throughput,v3(0.82f,0.93f,1.0f));
            } else {
                ray.o=add(hit.p,muls(hit.n,0.003f));
                ray.d=random_cosine_hemisphere(hit.n,rng);
                throughput=mul(throughput,muls(hit.color,0.72f));
            }
        } else {
            ray.o=add(hit.p,muls(hit.n,0.003f));
            ray.d=random_cosine_hemisphere(hit.n,rng);
            throughput=mul(throughput,muls(hit.color,0.82f));
        }

        float p=fmaxf(throughput.x,fmaxf(throughput.y,throughput.z));
        if(bounce>=3){
            if(rand01(rng)>p) break;
            throughput=divs(throughput,fmaxf(p,0.05f));
        }
    }

    return radiance;
}

extern "C" __global__ void clear_accum(float* accum,unsigned int n){
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) accum[i]=0.0f;
}

extern "C" __global__ void trace_sample(
    float* accum,
    const RenderParams* params
){
    RenderParams p = params[0];
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int n=p.width*p.height;
    if(i>=n) return;

    Scene scene;
    scene.s0=p.s0;
    scene.s1=p.s1;
    scene.s2=p.s2;
    scene.light=p.light;
    scene.material_preset=p.material_preset;

    unsigned int x=i%p.width;
    unsigned int y=i/p.width;
    unsigned int rng=wang_hash(i^(p.sample_index*747796405u)^0x9e3779b9u);

    float jx = p.moving_preview != 0u ? 0.0f : rand01(&rng)-0.5f;
    float jy = p.moving_preview != 0u ? 0.0f : rand01(&rng)-0.5f;

    float aspect=(float)p.width/(float)p.height;
    float fov=42.0f*0.017453292519943295f;
    float scale=tanf(fov*0.5f);
    float sx=(((float)x+0.5f+jx)/(float)p.width*2.0f-1.0f)*aspect*scale;
    float sy=(1.0f-((float)y+0.5f+jy)/(float)p.height*2.0f)*scale;

    V3 cam=p.cam;
    V3 forward=normalize3(p.forward);
    V3 right=normalize3(p.right);
    V3 up=normalize3(p.up);

    V3 sensor_dir=normalize3(add(forward,add(muls(right,sx),muls(up,sy))));

    float lens_r=p.aperture*sqrtf(rand01(&rng));
    float lens_a=6.28318530718f*rand01(&rng);
    V3 lens_offset=add(muls(right,cosf(lens_a)*lens_r),muls(up,sinf(lens_a)*lens_r));
    V3 focal_point=add(cam,muls(sensor_dir,p.focus_dist/fmaxf(0.05f,dot3(sensor_dir,forward))));

    Ray ray;
    ray.o=add(cam,lens_offset);
    ray.d=normalize3(sub(focal_point,ray.o));

    V3 color=clamp3(trace_path(ray,scene,&rng),0.0f,12.0f);
    unsigned int base=i*4u;
    accum[base+0u]+=color.x;
    accum[base+1u]+=color.y;
    accum[base+2u]+=color.z;
    accum[base+3u]+=1.0f;
}

__device__ V3 accum_color(const float* accum,int x,int y,int width,int height){
    x=max(0,min(width-1,x)); y=max(0,min(height-1,y));
    unsigned int i=(unsigned int)y*(unsigned int)width+(unsigned int)x;
    unsigned int base=i*4u;
    float samples=fmaxf(1.0f,accum[base+3u]);
    return v3(accum[base+0u]/samples,accum[base+1u]/samples,accum[base+2u]/samples);
}

__device__ float luma(V3 c){return dot3(c,v3(0.2126f,0.7152f,0.0722f));}

__device__ V3 tonemap(V3 c,float exposure){
    c=muls(c,exposure);
    V3 numerator=mul(c,add(muls(c,2.51f),v3(0.03f,0.03f,0.03f)));
    V3 denominator=add(mul(c,add(muls(c,2.43f),v3(0.59f,0.59f,0.59f))),v3(0.14f,0.14f,0.14f));
    c=v3(numerator.x/denominator.x,numerator.y/denominator.y,numerator.z/denominator.z);
    c=clamp3(c,0.0f,1.0f);
    c.x=powf(c.x,1.0f/2.2f); c.y=powf(c.y,1.0f/2.2f); c.z=powf(c.z,1.0f/2.2f);
    return c;
}

extern "C" __global__ void reconstruct_frame(
    unsigned int* frame,
    const float* accum,
    unsigned int width,
    unsigned int height,
    unsigned int reconstruction_enabled,
    float exposure,
    unsigned int frame_index,
    unsigned int selected
){
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int n=width*height;
    if(i>=n) return;

    int x=(int)(i%width);
    int y=(int)(i/width);
    V3 center=accum_color(accum,x,y,(int)width,(int)height);
    V3 color=center;

    if(reconstruction_enabled!=0u){
        float center_luma=luma(center);
        V3 sum=muls(center,1.20f);
        float wsum=1.20f;

        for(int oy=-2;oy<=2;++oy){
            for(int ox=-2;ox<=2;++ox){
                if(ox==0&&oy==0) continue;
                V3 c=accum_color(accum,x+ox,y+oy,(int)width,(int)height);
                float dl=fabsf(luma(c)-center_luma);
                float spatial=1.0f/(1.0f+(float)(ox*ox+oy*oy));
                float edge=1.0f/(1.0f+18.0f*dl);
                float w=spatial*edge;
                sum=add(sum,muls(c,w));
                wsum+=w;
            }
        }

        V3 filtered=divs(sum,wsum);
        float detail=fminf(1.0f,fabsf(luma(center)-luma(filtered))*3.5f);

        // Stronger reconstruction when sample count is low, especially while
        // moving. As samples accumulate, preserve more raw path-traced detail.
        float samples_here = fmaxf(1.0f, accum[i * 4u + 3u]);
        float low_sample = 1.0f - fminf(1.0f, samples_here / 20.0f);
        float filtered_weight = 0.78f + 0.16f * low_sample - 0.28f * detail;
        filtered_weight = fminf(0.96f, fmaxf(0.42f, filtered_weight));
        color=add(muls(filtered,filtered_weight),muls(center,1.0f-filtered_weight));
    }

    color=tonemap(color,exposure);

    color=clamp3(color,0.0f,1.0f);

    float samples=fmaxf(1.0f,accum[i*4u+3u]);
    if(x<8){
        color=reconstruction_enabled!=0u?v3(0.2f,0.9f,1.0f):v3(1.0f,0.45f,0.10f);
    }
    if(y>(int)height-9){
        float progress=fminf(1.0f,samples/240.0f);
        if((float)x/(float)width<progress) color=v3(0.95f,0.80f,0.35f);
    }

    // Corner indicator for selected object/light.
    if(x<80&&y<26){
        if(selected==0u) color=v3(1.0f,0.65f,0.32f);
        else if(selected==1u) color=v3(0.40f,0.80f,1.0f);
        else if(selected==2u) color=v3(0.95f,0.20f,0.15f);
        else color=v3(1.0f,0.92f,0.38f);
    }

    unsigned int r=(unsigned int)(fminf(1.0f,color.x)*255.0f);
    unsigned int g=(unsigned int)(fminf(1.0f,color.y)*255.0f);
    unsigned int b=(unsigned int)(fminf(1.0f,color.z)*255.0f);
    frame[i]=(r<<16)|(g<<8)|b;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len <= 1.0e-20 {
            Self::new(0.0, 1.0, 0.0)
        } else {
            self.mul(1.0 / len)
        }
    }
}


unsafe impl DevicePod for Vec3 {}

#[repr(C)]
#[derive(Clone, Copy)]
struct RenderParams {
    width: u32,
    height: u32,
    sample_index: u32,
    material_preset: u32,
    moving_preview: u32,
    cam: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    aperture: f32,
    focus_dist: f32,
    s0: Vec3,
    s1: Vec3,
    s2: Vec3,
    light: Vec3,
}

unsafe impl DevicePod for RenderParams {}

fn camera_basis(yaw: f32, pitch: f32) -> (Vec3, Vec3, Vec3) {
    let cp = pitch.cos();
    let forward = Vec3::new(yaw.sin() * cp, pitch.sin(), -yaw.cos() * cp).normalize();
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = forward.cross(world_up).normalize();
    let up = right.cross(forward).normalize();
    (forward, right, up)
}

fn clear_accum(
    kernel: &rocm_oxide::Kernel,
    accum: &DeviceBuffer<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        rocm_oxide::launch!(
            kernel,
            LaunchConfig::for_num_elems_with_block_size(ACCUM_FLOATS, 256),
            accum.as_mut_ptr(),
            ACCUM_FLOATS as u32,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ROCM_OXIDE_VISUAL_PRESENT").as_deref() != Ok("vulkan") {
        eprintln!("note: run with `ROCM_OXIDE_VISUAL_PRESENT=vulkan` to exercise Vulkan presentation");
    }

    let device = Device::first()?;
    let module = device.compile_hip_source(KERNELS)?;
    let clear_kernel = module.kernel(c"clear_accum")?;
    let trace_kernel = module.kernel(c"trace_sample")?;
    let recon_kernel = module.kernel(c"reconstruct_frame")?;

    let accum = DeviceBuffer::<f32>::new(ACCUM_FLOATS)?;
    let frame = DeviceBuffer::<u32>::new(PIXELS)?;
    let params = DeviceBuffer::<RenderParams>::new(1)?;
    clear_accum(&clear_kernel, &accum)?;

    let mut window = Window::new(
        "ROCm-Oxide Interactive Path Reconstruction",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
        },
    )?;
    window.set_target_fps(144);

    let max_frames = requested_frames("ROCM_OXIDE_INTERACTIVE_PATH_RECON_FRAMES");
    let mut last_tick = Instant::now();

    let mut cam = Vec3::new(0.0, 0.50, 4.45);
    let mut yaw = 0.0f32;
    let mut pitch = -0.03f32;

    let mut s0 = Vec3::new(-1.05, -0.10, -0.55);
    let mut s1 = Vec3::new(0.72, -0.18, -0.35);
    let mut s2 = Vec3::new(0.10, -0.43, 0.84);
    let mut light = Vec3::new(0.0, 2.65, -1.25);

    let mut sample_index = 0u32;
    let mut rendered_frames = 0u32;
    let mut reconstruction = true;
    let mut exposure = 1.15f32;
    let mut aperture = 0.030f32;
    let mut focus_dist = 4.55f32;
    let mut selected = 0u32;
    let mut material_preset = 0u32;
    let mut animate = true;
    let mut time = 0.0f32;
    let mut still_frames = 0u32;

    println!("interactive_path_reconstruction: WASD camera, 1-4 select objects/light, 5/6/7/8/PageUp/PageDown move selection");
    println!("render: Space reconstruction | R reset | C material | P pause animation | 9/0 exposure | LeftShift fast");
    println!("controls are time-scaled, so they should feel similar at high or low FPS");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = (now - last_tick).as_secs_f32().clamp(0.001, 0.050);
        last_tick = now;

        let fast = window.is_key_down(Key::LeftShift);

        // Speeds are now expressed per second instead of per frame. The Vulkan
        // presenter can run well above 60 FPS, so frame-based increments feel
        // wildly too fast on fast GPUs.
        let move_speed = if fast { 1.90 } else { 0.62 } * dt;
        let look_speed = if fast { 0.95 } else { 0.32 } * dt;
        let object_speed = if fast { 1.20 } else { 0.38 } * dt;
        let exposure_rate = if fast { 1.20 } else { 0.45 } * dt;

        let (forward, right, up) = camera_basis(yaw, pitch);
        let mut reset = false;
        let mut view_changed = false;

        if window.is_key_down(Key::W) {
            cam = cam.add(forward.mul(move_speed));
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::S) {
            cam = cam.add(forward.mul(-move_speed));
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::A) {
            cam = cam.add(right.mul(-move_speed));
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::D) {
            cam = cam.add(right.mul(move_speed));
            reset = true;
            view_changed = true;
        }

        if window.is_key_down(Key::Left) {
            yaw -= look_speed;
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::Right) {
            yaw += look_speed;
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::Up) && !window.is_key_down(Key::LeftShift) {
            pitch = (pitch + look_speed).clamp(-PI * 0.42, PI * 0.42);
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::Down) && !window.is_key_down(Key::LeftShift) {
            pitch = (pitch - look_speed).clamp(-PI * 0.42, PI * 0.42);
            reset = true;
            view_changed = true;
        }

        if window.is_key_down(Key::Down) && window.is_key_down(Key::LeftShift) {
            focus_dist = (focus_dist - look_speed).clamp(0.1, 10.0);
            reset = true;
            view_changed = true;
        }
        if window.is_key_down(Key::Up) && window.is_key_down(Key::LeftShift) {
            focus_dist = (focus_dist + look_speed).clamp(0.1, 10.0);
            reset = true;
            view_changed = true;
        }

        if window.is_key_pressed(Key::Key1, KeyRepeat::No) {
            selected = 0;
        }
        if window.is_key_pressed(Key::Key2, KeyRepeat::No) {
            selected = 1;
        }
        if window.is_key_pressed(Key::Key3, KeyRepeat::No) {
            selected = 2;
        }
        if window.is_key_pressed(Key::Key4, KeyRepeat::No) {
            selected = 3;
        }

        let mut object_delta = Vec3::new(0.0, 0.0, 0.0);
        if window.is_key_down(Key::Key5) {
            object_delta = object_delta.add(Vec3::new(-object_speed, 0.0, 0.0));
        }
        if window.is_key_down(Key::Key6) {
            object_delta = object_delta.add(Vec3::new(object_speed, 0.0, 0.0));
        }
        if window.is_key_down(Key::Key7) {
            object_delta = object_delta.add(Vec3::new(0.0, 0.0, -object_speed));
        }
        if window.is_key_down(Key::Key8) {
            object_delta = object_delta.add(Vec3::new(0.0, 0.0, object_speed));
        }
        if window.is_key_down(Key::PageUp) {
            object_delta = object_delta.add(Vec3::new(0.0, object_speed, 0.0));
        }
        if window.is_key_down(Key::PageDown) {
            object_delta = object_delta.add(Vec3::new(0.0, -object_speed, 0.0));
        }

        if object_delta.x != 0.0 || object_delta.y != 0.0 || object_delta.z != 0.0 {
            match selected {
                0 => s0 = s0.add(object_delta),
                1 => s1 = s1.add(object_delta),
                2 => s2 = s2.add(object_delta),
                _ => light = light.add(object_delta),
            }
            reset = true;
            view_changed = true;
        }

        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            reconstruction = !reconstruction;
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            reset = true;
            view_changed = true;
        }
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            animate = !animate;
        }
        if window.is_key_pressed(Key::C, KeyRepeat::No) {
            material_preset = material_preset.wrapping_add(1) & 3;
            reset = true;
            view_changed = true;
        }

        // Exposure controls. Focus/aperture stay stable so the numbered keys
        // can be used for object manipulation with the current shared presenter key set.
        if window.is_key_down(Key::Key9) {
            exposure = (exposure - exposure_rate).max(0.1);
        }
        if window.is_key_down(Key::Key0) {
            exposure = (exposure + exposure_rate).min(3.0);
        }

        if reset {
            clear_accum(&clear_kernel, &accum)?;
            sample_index = 0;
            still_frames = 0;
        } else {
            still_frames = still_frames.saturating_add(1);
        }

        if animate {
            time += dt;
        }

        // While moving, the accumulation buffer has just been reset, so a single
        // path-tracing sample per pixel is very noisy. Spend a little more GPU
        // work during motion and the first few stable frames to hide that noise.
        let samples_this_frame = if view_changed {
            16
        } else if still_frames < 8 {
            8
        } else if still_frames < 32 {
            4
        } else {
            1
        };

        let animated_light = Vec3::new(
            light.x + 0.16 * (time * 0.77).sin(),
            light.y + 0.08 * (time * 1.17).cos(),
            light.z + 0.16 * (time * 0.53).cos(),
        );

        let (forward, right, up) = camera_basis(yaw, pitch);
        let effective_reconstruction = reconstruction || view_changed || still_frames < 32;
        let moving_preview = view_changed || still_frames < 8;
        let effective_aperture = if moving_preview { 0.0 } else { aperture };

        unsafe {
            for _ in 0..samples_this_frame {
                let render_params = RenderParams {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                    sample_index,
                    material_preset,
                    moving_preview: moving_preview as u32,
                    cam,
                    forward,
                    right,
                    up,
                    aperture: effective_aperture,
                    focus_dist,
                    s0,
                    s1,
                    s2,
                    light: animated_light,
                };
                params.copy_from_host(&[render_params])?;

                rocm_oxide::launch!(
                    trace_kernel,
                    LaunchConfig::for_num_elems_with_block_size(PIXELS, 256),
                    accum.as_mut_ptr(),
                    params.as_ptr(),
                )?;

                sample_index = sample_index.wrapping_add(1);
            }

            rocm_oxide::launch!(
                recon_kernel,
                LaunchConfig::for_num_elems_with_block_size(PIXELS, 256),
                frame.as_mut_ptr(),
                accum.as_ptr(),
                WIDTH as u32,
                HEIGHT as u32,
                effective_reconstruction as u32,
                exposure,
                rendered_frames,
                selected,
            )?;
        }

        rocm_oxide::hip::synchronize()?;
        window.update_with_device_buffer(&frame, WIDTH, HEIGHT)?;

        if rendered_frames % 30 == 0 {
            let selected_name = match selected {
                0 => "metal",
                1 => "glass",
                2 => "diffuse",
                _ => "light",
            };
            window.set_title(&format!(
                "Interactive Path Reconstruction | samples={} spp={} recon={} selected={} material={} aperture={:.3} focus={:.2}",
                sample_index,
                samples_this_frame,
                reconstruction,
                selected_name,
                material_preset,
                aperture,
                focus_dist
            ));
        }

        rendered_frames = rendered_frames.wrapping_add(1);

        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            break;
        }
    }

    println!(
        "interactive_path_reconstruction: rendered {} frame(s), accumulated {} sample(s)",
        rendered_frames, sample_index
    );

    Ok(())
}
