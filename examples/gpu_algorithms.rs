use rocm_oxide::{GpuArray, Result, RocPrim, RocThrust};

fn main() -> Result<()> {
    if !RocPrim::is_available() {
        println!("gpu_algorithms: rocPRIM/hipCUB shim unavailable; skipping algorithm demo");
        return Ok(());
    }

    let input = GpuArray::from_values([1u32, 2, 3, 4])?;
    let sum = input.sum()?;
    assert_eq!(sum, 10);

    let scan = input.exclusive_scan(0)?;
    assert_eq!(scan.download()?, [0, 1, 3, 6]);

    let mapped = input.map_add(8)?;
    assert_eq!(mapped.download()?, [9, 10, 11, 12]);

    let params = GpuArray::from_value(7u32)?;
    assert_eq!(params.read()?, 7);
    params.write(11)?;
    assert_eq!(params.read()?, 11);

    let flags = GpuArray::from_slice(&[1u8, 0, 1, 0])?;
    let (selected, selected_count) = input.compact_by_flags(&flags)?;
    assert_eq!(selected_count, 2);
    assert_eq!(&selected.download()?[..selected_count], [1, 3]);

    if RocThrust::is_available() {
        let mut sortable = GpuArray::from_slice(&[4u32, 1, 3, 2, 3])?;
        assert!(sortable.contains(3)?);
        assert_eq!(sortable.count_eq(3)?, 2);
        assert_eq!(sortable.sorted_keys()?.download()?, [1, 2, 3, 3, 4]);
        sortable.sort()?;
        assert_eq!(sortable.download()?, [1, 2, 3, 3, 4]);
        assert_eq!(sortable.unique_consecutive()?, 4);

        let mut keys = GpuArray::from_slice(&[3u32, 1, 2])?;
        let mut values = GpuArray::from_slice(&[30u32, 10, 20])?;
        keys.sort_by_key(&mut values)?;
        assert_eq!(keys.download()?, [1, 2, 3]);
        assert_eq!(values.download()?, [10, 20, 30]);
    }

    println!(
        "gpu_algorithms: GpuArray constructor, copy, reduce, scan, select, map, and sort helpers passed"
    );
    Ok(())
}
