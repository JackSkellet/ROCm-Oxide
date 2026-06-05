use rocm_oxide::{GpuArray, Result, RocPrim, RocThrust};

fn main() -> Result<()> {
    if !RocPrim::is_available() {
        println!("gpu_algorithms: rocPRIM/hipCUB shim unavailable; skipping algorithm demo");
        return Ok(());
    }

    let input = GpuArray::from_slice(&[1u32, 2, 3, 4])?;
    let sum = input.sum()?;
    assert_eq!(sum, 10);

    let scan = input.exclusive_scan(0)?;
    assert_eq!(scan.to_vec()?, [0, 1, 3, 6]);

    let mapped = input.map_add(8)?;
    assert_eq!(mapped.to_vec()?, [9, 10, 11, 12]);

    if RocThrust::is_available() {
        let mut sortable = GpuArray::from_slice(&[4u32, 1, 3, 2])?;
        sortable.sort()?;
        assert_eq!(sortable.to_vec()?, [1, 2, 3, 4]);
    }

    println!("gpu_algorithms: GpuArray reduce, scan, map, and sort helpers passed");
    Ok(())
}
