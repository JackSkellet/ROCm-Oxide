use rocm_oxide::{DeviceBuffer, Result, RocPrim, RocThrust, gpu};

fn main() -> Result<()> {
    if !RocPrim::is_available() {
        println!("gpu_algorithms: rocPRIM/hipCUB shim unavailable; skipping algorithm demo");
        return Ok(());
    }

    let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4])?;
    let sum = gpu::reduce_sum(&input)?;
    assert_eq!(sum, 10);

    let scan = DeviceBuffer::<u32>::new(input.len())?;
    gpu::exclusive_scan(&input, &scan, 0)?;
    assert_eq!(scan.copy_to_vec()?, [0, 1, 3, 6]);

    let mapped = DeviceBuffer::<u32>::new(input.len())?;
    gpu::map_add_u32(&input, &mapped, 8)?;
    assert_eq!(mapped.copy_to_vec()?, [9, 10, 11, 12]);

    let flags = DeviceBuffer::from_slice(&[1u8, 0, 1, 0])?;
    let selected = DeviceBuffer::<u32>::new(input.len())?;
    let selected_count = DeviceBuffer::<u32>::new(1)?;
    gpu::select_flagged_u32(&input, &flags, &selected, &selected_count)?;
    assert_eq!(selected_count.copy_to_vec()?, [2]);
    assert_eq!(&selected.copy_to_vec()?[..2], [1, 3]);

    if RocThrust::is_available() {
        let mut sortable = DeviceBuffer::from_slice(&[4u32, 1, 3, 2])?;
        gpu::sort(&mut sortable)?;
        assert_eq!(sortable.copy_to_vec()?, [1, 2, 3, 4]);
    }

    println!("gpu_algorithms: reduce, scan, map, select, and sort helpers passed");
    Ok(())
}
