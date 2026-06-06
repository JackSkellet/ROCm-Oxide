use rocm_oxide::{GpuArray, GpuArray2D, Result, RocPrim, RocThrust, gpu};

fn main() -> Result<()> {
    if !RocPrim::is_available() {
        println!("gpu_algorithms: rocPRIM/hipCUB shim unavailable; skipping algorithm demo");
        return Ok(());
    }

    let input = gpu::array([1u32, 2, 3, 4])?;
    assert_eq!(input.shape(), [4]);
    let sum = input.sum()?;
    assert_eq!(sum, 10);
    assert_eq!(gpu::reduce_sum(&input)?, 10);

    let scan = gpu::empty::<u32>(input.len())?;
    gpu::exclusive_scan(&input, &scan, 0)?;
    assert_eq!(scan.download()?, [0, 1, 3, 6]);
    assert_eq!(input.cumsum()?.download()?, [1, 3, 6, 10]);
    assert_eq!(gpu::cumsum(&input)?.download()?, [1, 3, 6, 10]);
    assert_eq!(gpu::exclusive_cumsum(&input, 0)?.download()?, [0, 1, 3, 6]);

    let mapped = input.add_scalar(8)?;
    assert_eq!(mapped.download()?, [9, 10, 11, 12]);
    assert_eq!(gpu::add_scalar_u32(&input, 2)?.download()?, [3, 4, 5, 6]);
    let mapped_into = gpu::empty::<u32>(input.len())?;
    input.add_scalar_into(&mapped_into, 3)?;
    assert_eq!(mapped_into.download()?, [4, 5, 6, 7]);

    let copied = input.copy()?;
    assert_eq!(copied.download()?, [1, 2, 3, 4]);
    let free_copied = gpu::copy(&input)?;
    assert_eq!(free_copied.download()?, [1, 2, 3, 4]);

    let params = GpuArray::from_value(7u32)?;
    assert_eq!(params.item()?, 7);
    params.write(11)?;
    assert_eq!(params.item()?, 11);

    let filled = gpu::full(3, 42u32)?;
    assert_eq!(filled.to_list()?, [42, 42, 42]);
    let matrix = GpuArray2D::from_slice(2, 3, &[1u32, 2, 3, 4, 5, 6])?;
    assert_eq!(matrix.shape(), [2, 3]);
    assert_eq!(matrix.to_rows()?, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    let matrix_copy = matrix.copy()?;
    assert_eq!(matrix_copy.download()?, [1, 2, 3, 4, 5, 6]);
    let free_matrix = gpu::array_2d(2, 2, [9u32, 8, 7, 6])?;
    assert_eq!(free_matrix.to_rows()?, vec![vec![9, 8], vec![7, 6]]);

    let flags = GpuArray::from_slice(&[1u8, 0, 1, 0])?;
    let (selected, selected_count) = input.compact_by_flags(&flags)?;
    assert_eq!(selected_count, 2);
    assert_eq!(&selected.download()?[..selected_count], [1, 3]);
    let (selected_where, where_count) = input.where_flags(&flags)?;
    assert_eq!(where_count, 2);
    assert_eq!(&selected_where.download()?[..where_count], [1, 3]);
    let (free_where, free_where_count) = gpu::where_flags_u32(&input, &flags)?;
    assert_eq!(free_where_count, 2);
    assert_eq!(&free_where.download()?[..free_where_count], [1, 3]);

    if RocThrust::is_available() {
        let mut sortable = GpuArray::from_slice(&[4u32, 1, 3, 2, 3])?;
        assert!(sortable.contains(3)?);
        assert_eq!(sortable.count_eq(3)?, 2);
        assert_eq!(sortable.sorted_keys()?.download()?, [1, 2, 3, 3, 4]);
        let (unique_values, unique_count) = sortable.sorted_unique()?;
        assert_eq!(unique_count, 4);
        assert_eq!(&unique_values.download()?[..unique_count], [1, 2, 3, 4]);
        let (free_unique_values, free_unique_count) = gpu::sorted_unique_u32(&sortable)?;
        assert_eq!(free_unique_count, 4);
        assert_eq!(
            &free_unique_values.download()?[..free_unique_count],
            [1, 2, 3, 4]
        );
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
