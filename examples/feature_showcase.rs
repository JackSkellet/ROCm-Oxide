use rocm_oxide::{
    AtomicMemoryKind, Device, DeviceBuffer, DeviceOperation, Dim3, ExecutionContext, LaunchConfig,
    ManagedBuffer, ManagedMemoryKind, PinnedHostBuffer, Result, StreamPool,
};
use std::sync::{Arc, mpsc};
use std::time::Duration;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

const VECTOR_ADD_HIPRTC: &str = r#"
extern "C" __global__
void vector_add(float* out, const float* a, const float* b, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

const MODULE_GLOBAL_HIPRTC: &str = r#"
extern "C" {
__device__ float scale_factor = 1.0f;
}

extern "C" __global__
void scale_vec(float* out, const float* input, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] * scale_factor;
    }
}
"#;

fn vector_add_operation(
    device: Device,
    a: Vec<f32>,
    b: Vec<f32>,
) -> impl DeviceOperation<Output = Vec<f32>> {
    move |context: &ExecutionContext| -> Result<Vec<f32>> {
        let n = a.len();
        let module = device.compile_hip_source(VECTOR_ADD_HIPRTC)?;
        let kernel = module.kernel(c"vector_add")?;

        let d_a = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        let d_b = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        let d_out = DeviceBuffer::<f32>::new_async(context.stream(), n)?;
        d_a.copy_from_host_async(context.stream(), &a)?;
        d_b.copy_from_host_async(context.stream(), &b)?;

        let config = LaunchConfig::for_num_elems(n);
        let mut out_ptr = d_out.as_mut_ptr();
        let mut a_ptr = d_a.as_ptr();
        let mut b_ptr = d_b.as_ptr();
        let mut n_arg = n as u64;
        let mut params = [
            rocm_oxide::__private::arg_ptr(&mut out_ptr),
            rocm_oxide::__private::arg_ptr(&mut a_ptr),
            rocm_oxide::__private::arg_ptr(&mut b_ptr),
            rocm_oxide::__private::arg_ptr(&mut n_arg),
        ];
        unsafe {
            kernel.launch_raw_on_stream(context.stream(), config, &mut params)?;
        }

        let mut out = vec![0.0f32; n];
        d_out.copy_to_host_async(context.stream(), &mut out)?;
        Ok(out)
    }
}

fn scoped_atomic_config() -> LaunchConfig {
    LaunchConfig::new(Dim3::x(1), Dim3::x(256))
}

fn verify_scoped_atomic_outputs(out: &[u32], counters: &[u32]) {
    assert_eq!(out, &[0, 1, 2, 0]);
    assert_eq!(counters, &[256, 256, 256]);
}

fn launch_scoped_atomics_raw(
    kernels: &generated::DeviceKernels,
    out_ptr: *mut u32,
    out_len: usize,
    counters_ptr: *mut u32,
    counters_len: usize,
) -> Result<()> {
    let resource = kernels
        .resource("scoped_atomics")
        .expect("generated resource metadata should include scoped_atomics");
    let kernel = kernels
        .module()
        .kernel_with_metadata(c"scoped_atomics", resource.launch_metadata())?;
    unsafe {
        rocm_oxide::launch!(
            kernel,
            scoped_atomic_config(),
            out_ptr,
            out_len,
            counters_ptr,
            counters_len,
        )
    }
}

fn main() -> Result<()> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;
    let n = 1 << 16;
    let block_x = 256;
    let lds_resource = kernels
        .resource("lds_block_sum")
        .expect("generated resource metadata should include lds_block_sum");
    assert!(lds_resource.uses_dynamic_shared_mem);
    assert_eq!(
        kernels.resources().len(),
        generated::DEVICE_KERNEL_RESOURCES.len()
    );
    let vector_kernel = kernels.module().kernel(c"vector_add")?;
    let potential = vector_kernel.occupancy_max_potential_block_size(0, 0)?;
    assert!(potential.min_grid_size > 0);
    assert!(potential.block_size > 0);
    let active = vector_kernel.occupancy_max_active_blocks_per_multiprocessor(block_x, 0)?;
    assert!(active.blocks_per_multiprocessor > 0);
    let recommended = kernels.recommend_1d_launch("vector_add", n, 0, 0)?;
    assert!(recommended.config.grid.x > 0);
    assert_eq!(recommended.config.block.x, recommended.block_size);
    assert_eq!(recommended.dynamic_shared_mem_bytes, 0);
    assert!(recommended.active_blocks_per_multiprocessor > 0);
    assert!(recommended.waves_per_block.unwrap_or(0) > 0);
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (i as f32) * 0.5).collect::<Vec<_>>();

    println!("ROCm-Oxide feature showcase on {}", device.arch());
    println!("ok: HIP occupancy wrappers reported launch guidance");
    println!("ok: generated launch recommendation produced a 1D vector_add shape");

    let d_a = Arc::new(DeviceBuffer::from_slice(&a)?);
    let d_b = Arc::new(DeviceBuffer::from_slice(&b)?);
    let d_out = Arc::new(DeviceBuffer::<f32>::new(n)?);

    let short = Arc::new(DeviceBuffer::from_slice(&a[..n / 2])?);
    let rejected = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &short,
            &d_b,
        )
    };
    assert!(matches!(rejected, Err(rocm_oxide::Error::InvalidLaunch(_))));
    println!("ok: generated bindings rejected a short buffer before launch");

    unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &d_a,
            &d_b,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let out = d_out.copy_to_vec()?;
    assert_eq!(out[4096], a[4096] + b[4096]);
    println!("ok: Rust-authored AMDGPU vector_add launched from generated host bindings");

    let generic_input = (0..n)
        .map(|i| (i as u32).rotate_left(5))
        .collect::<Vec<_>>();
    let d_generic_input = DeviceBuffer::from_slice(&generic_input)?;
    let d_generic_out = DeviceBuffer::<u32>::new(n)?;
    unsafe {
        kernels.generic_copy_u32(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_generic_out,
            &d_generic_input,
            n,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let generic_out = d_generic_out.copy_to_vec()?;
    assert_eq!(generic_out[2048], generic_input[2048]);
    println!("ok: generic #[kernel] monomorphized without a handwritten wrapper");

    let completion = unsafe {
        kernels.vector_add_operation(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            Arc::clone(&d_out),
            Arc::clone(&d_a),
            Arc::clone(&d_b),
        )?
    }
    .sync_on(&device.execution_context()?)?;
    assert_eq!(completion.retained_count(), 4);
    let lazy_out = d_out.copy_to_vec()?;
    assert_eq!(lazy_out[8192], a[8192] + b[8192]);
    println!("ok: generated DeviceOperation binding launched on an execution stream");

    let graph_context = device.execution_context()?;
    let d_graph_out = Arc::new(DeviceBuffer::<f32>::new(n)?);
    let graph = unsafe {
        kernels.vector_add_operation(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            Arc::clone(&d_graph_out),
            Arc::clone(&d_a),
            Arc::clone(&d_b),
        )?
    }
    .capture_graph_on(&graph_context)?;
    assert_eq!(graph.capture_output().retained_count(), 4);
    graph.launch_and_sync_on(&graph_context)?;
    let graph_out = d_graph_out.copy_to_vec()?;
    assert_eq!(graph_out[16384], a[16384] + b[16384]);
    graph.launch_and_sync_on(&graph_context)?;
    println!("ok: captured and replayed a generated DeviceOperation HIP graph");

    let default_mem_pool = device.default_mem_pool()?;
    device.set_mem_pool(default_mem_pool)?;
    let _current_mem_pool = device.current_mem_pool()?;
    let release_threshold = default_mem_pool.release_threshold()?;
    default_mem_pool.set_release_threshold(release_threshold)?;
    let follow_events = default_mem_pool.reuse_follow_event_dependencies()?;
    default_mem_pool.set_reuse_follow_event_dependencies(follow_events)?;
    let _reserved_before = default_mem_pool.reserved_mem_current()?;
    let _used_before = default_mem_pool.used_mem_current()?;
    let pool_context = device.execution_context()?;
    let pooled =
        DeviceBuffer::<f32>::new_from_pool_async(pool_context.stream(), default_mem_pool, 32)?;
    let pooled_input = (0..32).map(|i| i as f32).collect::<Vec<_>>();
    let mut pooled_output = vec![0.0f32; 32];
    pooled.copy_from_host_async(pool_context.stream(), &pooled_input)?;
    pooled.copy_to_host_async(pool_context.stream(), &mut pooled_output)?;
    unsafe {
        pooled.free_async(pool_context.stream())?;
    }
    pool_context.synchronize()?;
    assert_eq!(pooled_output, pooled_input);
    default_mem_pool.trim_to(0)?;
    println!("ok: HIP stream-ordered memory pool controls handled async allocation");

    let properties = device.properties()?;
    assert!(properties.warp_size > 0);
    assert!(properties.multiprocessor_count > 0);
    if properties.can_map_host_memory {
        assert_eq!(
            properties.mapped_host_memory_kind(),
            Some(AtomicMemoryKind::MappedCoherentHost)
        );
    }
    let visible_devices = Device::all()?;
    assert!(!visible_devices.is_empty());
    if visible_devices.len() > 1 {
        let can_peer = visible_devices[0].can_access_peer(&visible_devices[1])?;
        if can_peer
            && visible_devices[0]
                .enable_peer_access(&visible_devices[1])
                .is_ok()
        {
            visible_devices[0].disable_peer_access(&visible_devices[1])?;
        }
    }
    rocm_oxide::hip::set_device(device.ordinal())?;
    println!("ok: device properties and peer-access probes described host memory support");

    let atomic_out = DeviceBuffer::<u32>::new(4)?;
    let atomic_counters = DeviceBuffer::from_slice(&[0u32; 3])?;
    unsafe {
        kernels.scoped_atomics(scoped_atomic_config(), &atomic_out, &atomic_counters)?;
    }
    rocm_oxide::hip::synchronize()?;
    verify_scoped_atomic_outputs(&atomic_out.copy_to_vec()?, &atomic_counters.copy_to_vec()?);
    println!("ok: scoped atomic kernel updated device-memory counters");

    let fine_atomic_out = DeviceBuffer::<u32>::new_fine_grained(4)?;
    let fine_atomic_counters = DeviceBuffer::<u32>::new_fine_grained(3)?;
    fine_atomic_counters.copy_from_host(&[0u32; 3])?;
    unsafe {
        kernels.scoped_atomics(
            scoped_atomic_config(),
            &fine_atomic_out,
            &fine_atomic_counters,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    verify_scoped_atomic_outputs(
        &fine_atomic_out.copy_to_vec()?,
        &fine_atomic_counters.copy_to_vec()?,
    );
    println!("ok: scoped atomic kernel updated fine-grained device counters");

    let host_atomic_out = PinnedHostBuffer::<u32>::new_zeroed_mapped_coherent(4)?;
    let host_atomic_counters = PinnedHostBuffer::<u32>::new_zeroed_mapped_coherent(3)?;
    launch_scoped_atomics_raw(
        &kernels,
        host_atomic_out.device_ptr()?,
        host_atomic_out.len(),
        host_atomic_counters.device_ptr()?,
        host_atomic_counters.len(),
    )?;
    rocm_oxide::hip::synchronize()?;
    verify_scoped_atomic_outputs(host_atomic_out.as_slice(), host_atomic_counters.as_slice());
    println!("ok: scoped atomic kernel updated mapped coherent host-visible counters");

    if properties.managed_memory {
        let managed_kind = properties
            .managed_memory_kind(ManagedMemoryKind::FineGrain)
            .expect("managed memory property should classify managed allocations");
        assert!(matches!(
            managed_kind,
            AtomicMemoryKind::ManagedFineGrain | AtomicMemoryKind::ManagedCoarseGrain
        ));
        let managed_atomic_out = ManagedBuffer::<u32>::new_zeroed(4)?;
        let managed_atomic_counters = ManagedBuffer::<u32>::new_zeroed(3)?;
        launch_scoped_atomics_raw(
            &kernels,
            managed_atomic_out.as_mut_ptr(),
            managed_atomic_out.len(),
            managed_atomic_counters.as_mut_ptr(),
            managed_atomic_counters.len(),
        )?;
        rocm_oxide::hip::synchronize()?;
        verify_scoped_atomic_outputs(
            managed_atomic_out.as_slice(),
            managed_atomic_counters.as_slice(),
        );

        let coarse_atomic_out = ManagedBuffer::<u32>::new_zeroed_coarse_grained(4)?;
        let coarse_atomic_counters = ManagedBuffer::<u32>::new_zeroed_coarse_grained(3)?;
        assert_eq!(
            properties.managed_memory_kind(coarse_atomic_counters.kind()),
            Some(AtomicMemoryKind::ManagedCoarseGrain)
        );
        launch_scoped_atomics_raw(
            &kernels,
            coarse_atomic_out.as_mut_ptr(),
            coarse_atomic_out.len(),
            coarse_atomic_counters.as_mut_ptr(),
            coarse_atomic_counters.len(),
        )?;
        rocm_oxide::hip::synchronize()?;
        verify_scoped_atomic_outputs(
            coarse_atomic_out.as_slice(),
            coarse_atomic_counters.as_slice(),
        );
        println!("ok: scoped atomic kernel updated managed fine/coarse host-visible counters");
    }

    let reduce_n = 768usize;
    let reduce_block_x = 128u32;
    let partial_count = reduce_n.div_ceil(reduce_block_x as usize);
    let reduce_config = LaunchConfig::for_num_elems_with_block_size(reduce_n, reduce_block_x)
        .try_with_dynamic_shared_mem::<f32>(reduce_block_x as usize)?;
    let lds_kernel = kernels
        .module()
        .kernel_with_metadata(c"lds_block_sum", lds_resource.launch_metadata())?;
    let lds_active = lds_kernel.occupancy_for_config(reduce_config)?;
    assert!(lds_active.blocks_per_multiprocessor > 0);
    let reduce_input = (0..reduce_n).map(|i| (i % 5) as f32).collect::<Vec<_>>();
    let d_reduce_input = DeviceBuffer::from_slice(&reduce_input)?;
    let d_partials = DeviceBuffer::<f32>::new(partial_count)?;
    unsafe {
        kernels.lds_block_sum(
            reduce_config,
            &d_partials,
            &d_reduce_input,
            reduce_n,
            partial_count,
            reduce_block_x,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let partials = d_partials.copy_to_vec()?;
    let expected = reduce_input
        .chunks(reduce_block_x as usize)
        .map(|chunk| chunk.iter().sum::<f32>())
        .collect::<Vec<_>>();
    assert_eq!(partials, expected);
    println!("ok: dynamic launch-sized LDS handled a block reduction kernel");

    let static_lds_resource = kernels
        .resource("static_lds_reverse")
        .expect("generated resource metadata should include static_lds_reverse");
    assert_eq!(static_lds_resource.group_segment_fixed_size, Some(1024));
    assert!(!static_lds_resource.uses_dynamic_shared_mem);
    let static_n = 512usize;
    let static_config = LaunchConfig::for_num_elems_with_block_size(static_n, 256);
    let static_kernel = kernels
        .module()
        .kernel_with_metadata(c"static_lds_reverse", static_lds_resource.launch_metadata())?;
    let static_active = static_kernel.occupancy_for_config(static_config)?;
    assert!(static_active.blocks_per_multiprocessor > 0);
    let static_input = (0..static_n as u32).collect::<Vec<_>>();
    let d_static_input = DeviceBuffer::from_slice(&static_input)?;
    let d_static_out = DeviceBuffer::<u32>::new(static_n)?;
    unsafe {
        kernels.static_lds_reverse(static_config, &d_static_out, &d_static_input, static_n)?;
    }
    rocm_oxide::hip::synchronize()?;
    let static_out = d_static_out.copy_to_vec()?;
    let expected_static = static_input
        .chunks(256)
        .flat_map(|chunk| chunk.iter().rev().copied())
        .collect::<Vec<_>>();
    assert_eq!(static_out, expected_static);
    println!("ok: static LDS marker produced a block-local reverse kernel");

    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 3.0,
        bias: -7.0,
    }])?;
    unsafe {
        kernels.affine_transform(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &d_a,
            &params,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let affine = d_out.copy_to_vec()?;
    assert_eq!(affine[1234], a[1234] * 3.0 - 7.0);
    println!("ok: mirrored repr(C) environment struct reached GPU code");

    let module = device.compile_hip_source(MODULE_GLOBAL_HIPRTC)?;
    let kernel = module.kernel(c"scale_vec")?;
    let scale = module.global::<f32>(c"scale_factor")?;
    scale.set(4.0)?;
    unsafe {
        rocm_oxide::launch!(
            kernel,
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            d_out.as_mut_ptr(),
            d_a.as_ptr(),
            n as u64,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let scaled = d_out.copy_to_vec()?;
    assert_eq!(scaled[321], a[321] * 4.0);
    assert_eq!(scale.copy_to_vec()?, vec![4.0]);
    println!("ok: HIP module global lookup/set/get updated GPU behavior");

    let pool = StreamPool::new(&device, 2)?;
    let future_a = vector_add_operation(device.clone(), a.clone(), b.clone()).async_in(&pool);
    let future_b = vector_add_operation(device.clone(), b.clone(), b.clone()).async_in(&pool);
    let async_a = future_a.wait()?;
    let async_b = future_b.wait()?;
    assert_eq!(async_a[n - 1], a[n - 1] + b[n - 1]);
    assert_eq!(async_b[n - 1], b[n - 1] * 2.0);
    println!("ok: lazy DeviceOperation jobs completed through a 2-stream pool");

    let (sent, received) = mpsc::channel();
    let dropped = (move |_context: &ExecutionContext| -> Result<()> {
        let _ = sent.send(());
        Ok(())
    })
    .async_in(&pool);
    drop(dropped);
    received
        .recv_timeout(Duration::from_secs(2))
        .expect("dropped DeviceFuture should not cancel already-started work");
    println!("ok: dropping DeviceFuture did not cancel in-flight work");

    println!("feature showcase passed");
    Ok(())
}
