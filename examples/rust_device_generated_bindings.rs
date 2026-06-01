use rocm_oxide::{Device, DeviceBuffer, DeviceOperation, Dim3, LaunchConfig, StreamPool};
use std::sync::Arc;

mod generated {
    include!(env!("ROCM_OXIDE_DEVICE_BINDINGS"));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let kernels = generated::DeviceKernels::load_embedded(&device)?;

    let n = 1 << 20;
    let block_x = 256u32;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (n - i) as f32).collect::<Vec<_>>();

    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_out = DeviceBuffer::<f32>::new(n)?;

    let delta = kernels.global_add_one_delta()?;
    assert_eq!(delta.copy_to_vec()?, vec![1.0]);
    delta.set(2.0)?;
    let add_input = DeviceBuffer::from_slice(&[1.0f32, 5.5, -3.0, 0.25])?;
    let add_out = DeviceBuffer::<f32>::new(add_input.len())?;
    unsafe {
        kernels.add_one(
            LaunchConfig::for_num_elems_with_block_size(add_input.len(), block_x),
            &add_out,
            &add_input,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(add_out.copy_to_vec()?, vec![3.0, 7.5, -1.0, 2.25]);

    let math_input = DeviceBuffer::from_slice(&[4.0f32, 0.0, 1.0, -1.0])?;
    let math_out = DeviceBuffer::<f32>::new(16)?;
    unsafe {
        kernels.math_intrinsics(LaunchConfig::for_num_elems(1), &math_out, &math_input)?;
    }
    rocm_oxide::hip::synchronize()?;
    let math = math_out.copy_to_vec()?;
    assert_close("sqrt_f32", math[0], 2.0, 0.0001)?;
    assert_close("rsqrt_f32", math[1], 0.5, 0.0001)?;
    assert_close("sin_f32", math[2], 0.0, 0.0001)?;
    assert_close("cos_f32", math[3], 1.0, 0.0001)?;
    assert_close("atan_f32", math[4], std::f32::consts::FRAC_PI_4, 0.002)?;
    assert_close("min_f32", math[5], -2.0, 0.0001)?;
    assert_close("max_f32", math[6], 3.0, 0.0001)?;
    assert_close("sqrt_f64", math[7], 2.0, 0.0001)?;
    assert_close("rsqrt_f64", math[8], 0.5, 0.0001)?;
    assert_close("sin_f64", math[9], 0.0, 0.0001)?;
    assert_close("cos_f64", math[10], 1.0, 0.0001)?;
    assert_close("atan_f64", math[11], std::f32::consts::FRAC_PI_4, 0.002)?;
    assert_eq!(math[12], 1.0, "sqrt_f32(-1) should produce NaN");
    assert_eq!(math[13], 1.0, "min_f32 should propagate NaN");
    assert_eq!(math[14], 1.0, "max_f32 should propagate NaN");
    assert_close("min_f64", math[15], -2.0, 0.0001)?;

    let atomic_scope_out = DeviceBuffer::<u32>::new(4)?;
    let atomic_counters = DeviceBuffer::from_slice(&[0u32; 3])?;
    unsafe {
        kernels.scoped_atomics(
            LaunchConfig::new(Dim3::x(1), Dim3::x(256)),
            &atomic_scope_out,
            &atomic_counters,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(atomic_scope_out.copy_to_vec()?, vec![0, 1, 2, 0]);
    assert_eq!(atomic_counters.copy_to_vec()?, vec![256, 256, 256]);

    let cooperative_out = DeviceBuffer::<u32>::new(12)?;
    unsafe {
        kernels.cooperative_groups_probe(
            LaunchConfig::new(Dim3::x(1), Dim3::x(block_x)),
            &cooperative_out,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let cooperative = cooperative_out.copy_to_vec()?;
    let wavefront_size = device.properties()?.warp_size;
    assert_eq!(
        cooperative,
        vec![
            block_x,
            0,
            0,
            wavefront_size,
            32,
            31,
            0,
            1,
            1,
            1,
            wavefront_size - 1,
            1
        ]
    );

    let api_out = DeviceBuffer::<u32>::new(24)?;
    let api_i32_counter = DeviceBuffer::from_slice(&[0i32])?;
    let api_u64_counter = DeviceBuffer::from_slice(&[0u64])?;
    let api_i64_counter = DeviceBuffer::from_slice(&[0i64])?;
    unsafe {
        kernels.device_api_breadth_probe(
            LaunchConfig::new(Dim3::x(1), Dim3::x(32)),
            &api_out,
            &api_i32_counter,
            &api_u64_counter,
            &api_i64_counter,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let api = api_out.copy_to_vec()?;
    let active_lanes = 32u32.min(wavefront_size);
    let expected_sum = active_lanes * (active_lanes + 1) / 2;
    let expected_or = (0..active_lanes).fold(0u32, |acc, lane| acc | (1u32 << (lane & 31)));
    let expected_xor = (0..active_lanes).fold(0u32, |acc, lane| acc ^ (1u32 << (lane & 31)));
    let expected_match_mask = (0..active_lanes).fold(0u64, |acc, lane| {
        if (lane & 3) == 0 {
            acc | (1u64 << lane)
        } else {
            acc
        }
    }) as u32;
    assert_eq!(api[0], 6, "shuffle lane 5 should read lane 5's value");
    assert_eq!(api[1], 2, "shuffle_down lane 0 should read lane 1");
    assert_eq!(api[2], 1, "shuffle_up lane 1 should read lane 0");
    assert_eq!(api[3], 2, "shuffle_xor lane 0 should read lane 1");
    assert_eq!(api[4], expected_sum);
    assert_eq!(api[5], -((active_lanes - 1) as i32) as u32);
    assert_eq!(api[6], 0);
    assert_eq!(api[7], expected_or);
    assert_eq!(api[8], expected_xor);
    assert_eq!(api[9], expected_match_mask);
    assert_eq!(api[10], 1);
    assert_eq!(api[11], 1);
    assert_eq!(api[12], 1);
    assert_eq!(api[13], 1);
    assert_eq!(api[14], 10);
    assert_eq!(api[15], -(active_lanes as i32) as u32);
    assert_eq!(api[16], active_lanes);
    assert_eq!(api[17], active_lanes * 2);
    assert_eq!(api[18], 100);
    assert_eq!(api[19], 1);
    assert_eq!(api[20], 1);
    assert_eq!(api[21], 1);
    assert_eq!(api_i32_counter.copy_to_vec()?, vec![-(active_lanes as i32)]);
    assert_eq!(api_u64_counter.copy_to_vec()?, vec![active_lanes as u64]);
    assert_eq!(
        api_i64_counter.copy_to_vec()?,
        vec![-((active_lanes * 2) as i64)]
    );

    let control_input = vec![0u32, 1, 2, 3, 7, 12, 15, 31];
    let control_values = DeviceBuffer::from_slice(&control_input)?;
    let control_out = DeviceBuffer::<u32>::new(control_input.len())?;
    let control_pairs = DeviceBuffer::<generated::ControlPair>::new(control_input.len())?;
    let control_params = generated::ControlParams { seed: 11, scale: 6 };
    unsafe {
        kernels.compiler_parity_matrix(
            LaunchConfig::for_num_elems_with_block_size(control_input.len(), 32),
            &control_out,
            &control_pairs,
            &control_values,
            control_params,
            control_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let control_scores = control_out.copy_to_vec()?;
    let control_pairs = control_pairs.copy_to_vec()?;
    for (index, value) in control_input.iter().copied().enumerate() {
        let expected_pair = control_pair_host(value, control_params);
        let expected_score = control_score_host(value, control_params, expected_pair);
        assert_eq!(
            (control_pairs[index].left, control_pairs[index].right),
            (expected_pair.left, expected_pair.right),
            "compiler_parity_matrix pair mismatch at {index}"
        );
        assert_eq!(
            control_scores[index], expected_score,
            "compiler_parity_matrix score mismatch at {index}"
        );
    }

    let layout_input = vec![2u32, 3, 5, 8, 13, 21];
    let layout_values = DeviceBuffer::from_slice(&layout_input)?;
    let layout_out = DeviceBuffer::<u32>::new(layout_input.len())?;
    let layout_params = generated::RustLayoutParams { base: 7, stride: 4 };
    unsafe {
        kernels.compiler_layout_probe(
            LaunchConfig::for_num_elems_with_block_size(layout_input.len(), 32),
            &layout_out,
            &layout_values,
            layout_params,
            layout_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(
        layout_out.copy_to_vec()?,
        layout_input
            .iter()
            .map(|value| value
                .wrapping_mul(layout_params.stride)
                .wrapping_add(layout_params.base))
            .collect::<Vec<_>>()
    );

    let closure_input = vec![1u32, 4, 9, 16, 25, 36, 49, 64];
    let closure_values = DeviceBuffer::from_slice(&closure_input)?;
    let closure_out = DeviceBuffer::<u32>::new(closure_input.len())?;
    let closure_params = generated::RustLayoutParams { base: 3, stride: 5 };
    unsafe {
        kernels.compiler_move_closure_probe_rust_layout_params(
            LaunchConfig::for_num_elems_with_block_size(closure_input.len(), 32),
            &closure_out,
            &closure_values,
            closure_params,
            closure_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(
        closure_out.copy_to_vec()?,
        closure_input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .wrapping_mul(closure_params.stride)
                    .wrapping_add(closure_params.base)
                    .wrapping_add((index as u32) & 1)
            })
            .collect::<Vec<_>>()
    );

    let flow_input = vec![0u32, 1, 2, 3, 4, 7, 9, 12, 15, 31, 42, 63];
    let flow_values = DeviceBuffer::from_slice(&flow_input)?;
    let flow_out = DeviceBuffer::<u32>::new(flow_input.len())?;
    unsafe {
        kernels.compiler_flow_cast_probe(
            LaunchConfig::for_num_elems_with_block_size(flow_input.len(), 32),
            &flow_out,
            &flow_values,
            flow_input.len(),
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    let flow_scores = flow_out.copy_to_vec()?;
    let expected_flow = flow_input
        .iter()
        .enumerate()
        .map(|(index, _)| flow_cast_score_host(&flow_input, index))
        .collect::<Vec<_>>();
    assert_eq!(
        flow_scores, expected_flow,
        "compiler_flow_cast_probe should match the host control-flow mirror"
    );

    let reduce_n = 1_000usize;
    let reduce_block_x = 128u32;
    let partial_count = reduce_n.div_ceil(reduce_block_x as usize);
    let reduce_input = (0..reduce_n).map(|i| (i % 7) as f32).collect::<Vec<_>>();
    let reduce_expected = reduce_input
        .chunks(reduce_block_x as usize)
        .map(|chunk| chunk.iter().sum::<f32>())
        .collect::<Vec<_>>();
    let d_reduce_input = DeviceBuffer::from_slice(&reduce_input)?;
    let d_partials = DeviceBuffer::<f32>::new(partial_count)?;
    let missing_shared_validation = unsafe {
        kernels.lds_block_sum(
            LaunchConfig::for_num_elems_with_block_size(reduce_n, reduce_block_x),
            &d_partials,
            &d_reduce_input,
            reduce_n,
            partial_count,
            reduce_block_x,
        )
    };
    match missing_shared_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected missing dynamic LDS: {message}");
        }
        Err(err) => return Err(format!("unexpected missing-LDS validation error: {err}").into()),
        Ok(()) => return Err("missing dynamic LDS launch unexpectedly succeeded".into()),
    }

    unsafe {
        kernels.lds_block_sum(
            LaunchConfig::for_num_elems_with_block_size(reduce_n, reduce_block_x)
                .try_with_dynamic_shared_mem::<f32>(reduce_block_x as usize)?,
            &d_partials,
            &d_reduce_input,
            reduce_n,
            partial_count,
            reduce_block_x,
        )?;
    }
    rocm_oxide::hip::synchronize()?;
    assert_eq!(d_partials.copy_to_vec()?, reduce_expected);

    let shared_validation = unsafe {
        kernels.lds_block_sum(
            LaunchConfig::for_num_elems_with_block_size(reduce_n, reduce_block_x)
                .with_shared_mem_bytes(u32::MAX),
            &d_partials,
            &d_reduce_input,
            reduce_n,
            partial_count,
            reduce_block_x,
        )
    };
    match shared_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected excess dynamic LDS: {message}");
        }
        Err(err) => return Err(format!("unexpected LDS validation error: {err}").into()),
        Ok(()) => return Err("excess dynamic LDS launch unexpectedly succeeded".into()),
    }

    let short = DeviceBuffer::from_slice(&a[..n / 2])?;
    let validation = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &short,
            &d_b,
        )
    };
    match validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected short buffer: {message}");
        }
        Err(err) => return Err(format!("unexpected validation error: {err}").into()),
        Ok(()) => return Err("short buffer launch unexpectedly succeeded".into()),
    }

    let block_validation = unsafe {
        kernels.vector_add(
            LaunchConfig::new(Dim3::x(1), Dim3::x(0)),
            &d_out,
            &d_a,
            &d_b,
        )
    };
    match block_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected invalid launch shape: {message}");
        }
        Err(err) => return Err(format!("unexpected launch validation error: {err}").into()),
        Ok(()) => return Err("invalid launch unexpectedly succeeded".into()),
    }

    let alias_validation = unsafe {
        kernels.vector_add(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &d_out,
            &d_out,
            &d_b,
        )
    };
    match alias_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected aliased mutable buffer: {message}");
        }
        Err(err) => return Err(format!("unexpected alias validation error: {err}").into()),
        Ok(()) => return Err("aliased mutable buffer launch unexpectedly succeeded".into()),
    }

    let small_frame = DeviceBuffer::<u32>::new(512)?;
    let small_color = DeviceBuffer::<u32>::new(127)?;
    let small_depth = DeviceBuffer::<f32>::new(128)?;
    let contract_validation = unsafe {
        kernels.depth_aware_upscale(
            LaunchConfig::for_num_elems_with_block_size(512, block_x),
            &small_frame,
            &small_color,
            &small_depth,
            512,
            0,
        )
    };
    match contract_validation {
        Err(rocm_oxide::Error::InvalidLaunch(message)) => {
            println!("Validation rejected explicit buffer contract: {message}");
        }
        Err(err) => return Err(format!("unexpected contract validation error: {err}").into()),
        Ok(()) => return Err("explicit contract violation unexpectedly succeeded".into()),
    }

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
    for (index, ((got, lhs), rhs)) in out.iter().zip(&a).zip(&b).enumerate() {
        let expected = lhs + rhs;
        if (*got - expected).abs() > f32::EPSILON {
            return Err(format!("mismatch at {index}: got {got}, expected {expected}").into());
        }
    }

    let graph_out = DeviceBuffer::<f32>::new(n)?;
    let graph = rocm_oxide::hip::Graph::new()?;
    unsafe {
        kernels.vector_add_graph_node(
            &graph,
            &[],
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            &graph_out,
            &d_a,
            &d_b,
        )?;
    }
    let graph_exec = graph.instantiate()?;
    let graph_stream = rocm_oxide::Stream::new()?;
    graph_exec.launch(&graph_stream)?;
    graph_stream.synchronize()?;
    let graph_result = graph_out.copy_to_vec()?;
    assert_eq!(graph_result[4096], a[4096] + b[4096]);

    let pool = StreamPool::new(&device, 2)?;
    let lazy_a = Arc::new(DeviceBuffer::from_slice(&a)?);
    let lazy_b = Arc::new(DeviceBuffer::from_slice(&b)?);
    let lazy_out = Arc::new(DeviceBuffer::<f32>::new(n)?);
    let lazy_completion = unsafe {
        kernels.vector_add_operation(
            LaunchConfig::for_num_elems_with_block_size(n, block_x),
            Arc::clone(&lazy_out),
            Arc::clone(&lazy_a),
            Arc::clone(&lazy_b),
        )?
    }
    .async_in(&pool)
    .wait()?;
    assert_eq!(lazy_completion.retained_count(), 5);
    let lazy = lazy_out.copy_to_vec()?;
    assert_eq!(lazy[4096], a[4096] + b[4096]);

    let params = DeviceBuffer::from_slice(&[generated::AffineParams {
        scale: 2.0,
        bias: 3.0,
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
    assert_eq!(affine[17], a[17] * 2.0 + 3.0);

    println!("Generated binding vector_add passed on {}", device.arch());
    Ok(())
}

fn assert_close(
    label: &str,
    got: f32,
    expected: f32,
    tolerance: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if (got - expected).abs() > tolerance {
        Err(format!("{label}: got {got}, expected {expected} +/- {tolerance}").into())
    } else {
        Ok(())
    }
}

fn classify_control_host(value: u32) -> u32 {
    match value & 3 {
        0 => 2,
        1 => 5,
        2 => 9,
        _ => 13,
    }
}

fn control_option_host(value: u32) -> Option<u32> {
    if (value & 1) == 0 {
        Some(value / 2 + 3)
    } else {
        None
    }
}

fn control_result_host(value: u32) -> Result<u32, u32> {
    if value < 12 {
        Ok(value.wrapping_mul(3).wrapping_add(1))
    } else {
        Err(value - 12)
    }
}

fn control_pair_host(value: u32, params: generated::ControlParams) -> generated::ControlPair {
    generated::ControlPair {
        left: value.wrapping_add(params.seed),
        right: classify_control_host(value).wrapping_add(params.scale.unsigned_abs()),
    }
}

fn control_score_host(
    value: u32,
    params: generated::ControlParams,
    pair: generated::ControlPair,
) -> u32 {
    let scale = params.scale.unsigned_abs();
    let kind_score = match value & 3 {
        0 => 17u32,
        1 => 31u32,
        2 => 47u32,
        _ => 61u32,
    };
    let option_score = match control_option_host(value) {
        Some(inner) => inner.wrapping_mul(5),
        None => 23,
    };
    let result_score = match control_result_host(value) {
        Ok(ok) => ok,
        Err(err) => err.wrapping_add(101),
    };
    let fixed = [value, value.wrapping_add(1), params.seed, scale];
    let runtime_index = (value as usize) & 3;
    let mut mutable = [0u32; 4];
    mutable[0] = fixed[runtime_index];
    mutable[1] = fixed[0].wrapping_add(fixed[1]);
    mutable[2] = pair.left;
    mutable[3] = pair.right;

    let array_score = fixed
        .iter()
        .fold(0u32, |acc, item| acc.wrapping_add(item & 15))
        .wrapping_add((0..4).fold(0u32, |acc, index| {
            if index == runtime_index {
                acc
            } else {
                acc.wrapping_add(mutable[index])
            }
        }));

    let mut loop_score = 0u32;
    let mut countdown = value & 3;
    while countdown > 0 {
        loop_score = loop_score.wrapping_add(countdown);
        countdown -= 1;
    }
    let mut step = 0u32;
    loop {
        if step >= 3 {
            break;
        }
        step = step.wrapping_add(1);
        if step == 2 {
            continue;
        }
        loop_score = loop_score.wrapping_add(step);
    }

    let signed = params.scale.wrapping_add(value as i32);
    let float_score = ((signed as f32) * 0.5 + 2.0) as u32;
    let bitcast_score = ((float_score as f32).to_bits() >> 20) & 31;

    kind_score
        .wrapping_add(option_score)
        .wrapping_add(result_score)
        .wrapping_add(array_score)
        .wrapping_add(loop_score)
        .wrapping_add(float_score)
        .wrapping_add(bitcast_score)
        .wrapping_add((pair.left ^ pair.right) & 31)
}

fn flow_cast_score_host(input: &[u32], index: usize) -> u32 {
    let value = input[index];
    let mut score = 0u32;

    if (value & 1) == 0 {
        if (value & 4) == 0 {
            score = score.wrapping_add(11);
        } else {
            score = score.wrapping_add(17);
        }
    } else if value % 3 == 0 {
        score = score.wrapping_add(23);
    } else if value > 10 {
        score = score.wrapping_add(31);
    } else {
        score = score.wrapping_add(37);
    }

    for outer in 0u32..3 {
        for inner in 0u32..4 {
            let candidate = value.wrapping_add(outer * 7 + inner);
            if (candidate & 1) == 0 {
                continue;
            }
            if inner == 3 && (value & 2) != 0 {
                break;
            }
            score = score.wrapping_add(candidate & 15);
        }
    }

    for step in 0..((value & 3) + 1) {
        score = score.wrapping_add((step + 1) * 3);
    }

    let mut offset = 0usize;
    while offset < 4 && index + offset < input.len() {
        let lane = input[index + offset];
        score = score.wrapping_add((lane & 7).wrapping_mul(offset as u32 + 1));
        offset += 1;
    }

    let signed = (score as i32).wrapping_sub(value as i32);
    score.wrapping_add((signed as u32) & 31)
}
