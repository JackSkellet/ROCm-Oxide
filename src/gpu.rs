//! Small host-side GPU algorithms built on ROCm library interop.
//!
//! This module is the ergonomic layer above [`crate::RocPrim`] and
//! [`crate::RocThrust`]. It is intentionally small: ROCm-Oxide still exposes the
//! lower-level wrappers for callers that need explicit temporary-storage or
//! stream control.

use crate::{DeviceBuffer, DevicePod, Result, RocPrim, RocThrust};
use std::ops::{Deref, DerefMut};

/// A small method-oriented wrapper around [`DeviceBuffer`].
///
/// `GpuArray<T>` is intended for approachable host-side code and autocomplete:
/// construct an array, call methods on it, and copy values back when needed.
/// It does not own a separate runtime or scheduler; methods delegate to the
/// free functions in this module and the underlying `DeviceBuffer`.
///
/// ```rust,ignore
/// use rocm_oxide::gpu::GpuArray;
///
/// let input = GpuArray::from_slice(&[1u32, 2, 3, 4])?;
/// let sum = input.sum()?;
/// let mapped = input.map_add(8)?;
/// let scanned = input.exclusive_scan(0)?;
///
/// assert_eq!(sum, 10);
/// assert_eq!(mapped.to_vec()?, [9, 10, 11, 12]);
/// assert_eq!(scanned.to_vec()?, [0, 1, 3, 6]);
/// ```
pub struct GpuArray<T> {
    buffer: DeviceBuffer<T>,
}

impl<T> GpuArray<T> {
    /// Wrap an existing device buffer.
    pub fn from_buffer(buffer: DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    /// Return the underlying device buffer.
    pub fn into_buffer(self) -> DeviceBuffer<T> {
        self.buffer
    }

    /// Borrow the underlying device buffer.
    pub fn as_buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    /// Mutably borrow the underlying device buffer.
    pub fn as_mut_buffer(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buffer
    }

    /// Number of elements in the array.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` when the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Fill the array with zero bytes.
    pub fn fill_zero(&self) -> Result<()> {
        fill_zero(&self.buffer)
    }

    /// Fill the array with a byte pattern.
    pub fn fill_bytes(&self, value: u8) -> Result<()> {
        fill_bytes(&self.buffer, value)
    }
}

impl<T: DevicePod> GpuArray<T> {
    /// Allocate a zero-filled device array.
    pub fn zeros(len: usize) -> Result<Self> {
        let buffer = DeviceBuffer::<T>::new(len)?;
        buffer.set_zero()?;
        Ok(Self { buffer })
    }
}

impl<T: Copy> GpuArray<T> {
    /// Allocate a device array and upload `input`.
    pub fn from_slice(input: &[T]) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::from_slice(input)?,
        })
    }

    /// Copy `input` into this existing device array.
    pub fn copy_from_slice(&self, input: &[T]) -> Result<()> {
        Ok(self.buffer.copy_from_host(input)?)
    }
}

impl<T: Copy + Default> GpuArray<T> {
    /// Copy this device array back to host memory.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        Ok(self.buffer.copy_to_vec()?)
    }
}

impl<T: ReduceSum> GpuArray<T> {
    /// Sum all elements and return the scalar result on the host.
    pub fn sum(&self) -> Result<T> {
        reduce_sum(&self.buffer)
    }
}

impl<T: PrefixSum> GpuArray<T> {
    /// Return an array containing the inclusive prefix sum of this array.
    pub fn inclusive_scan(&self) -> Result<Self> {
        let output = DeviceBuffer::<T>::new(self.len())?;
        inclusive_scan(&self.buffer, &output)?;
        Ok(Self { buffer: output })
    }

    /// Return an array containing the exclusive prefix sum of this array.
    pub fn exclusive_scan(&self, initial_value: T) -> Result<Self> {
        let output = DeviceBuffer::<T>::new(self.len())?;
        exclusive_scan(&self.buffer, &output, initial_value)?;
        Ok(Self { buffer: output })
    }
}

impl GpuArray<u32> {
    /// Add `addend` to every element and return the mapped output array.
    pub fn map_add(&self, addend: u32) -> Result<Self> {
        let output = DeviceBuffer::<u32>::new(self.len())?;
        map_add_u32(&self.buffer, &output, addend)?;
        Ok(Self { buffer: output })
    }

    /// Sort this array in place.
    pub fn sort(&mut self) -> Result<()> {
        sort(&mut self.buffer)
    }

    /// Return a sorted copy of this array.
    pub fn sorted(&self) -> Result<Self> {
        let mut output = DeviceBuffer::<u32>::new(self.len())?;
        self.buffer.copy_to_device(&output)?;
        sort(&mut output)?;
        Ok(Self { buffer: output })
    }

    /// Count elements equal to `value`.
    pub fn count_eq(&self, value: u32) -> Result<usize> {
        count_eq_u32(&self.buffer, value)
    }
}

impl<T> AsRef<DeviceBuffer<T>> for GpuArray<T> {
    fn as_ref(&self) -> &DeviceBuffer<T> {
        self.as_buffer()
    }
}

impl<T> AsMut<DeviceBuffer<T>> for GpuArray<T> {
    fn as_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.as_mut_buffer()
    }
}

impl<T> Deref for GpuArray<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.as_buffer()
    }
}

impl<T> DerefMut for GpuArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_buffer()
    }
}

impl<T> From<DeviceBuffer<T>> for GpuArray<T> {
    fn from(buffer: DeviceBuffer<T>) -> Self {
        Self::from_buffer(buffer)
    }
}

impl<T> From<GpuArray<T>> for DeviceBuffer<T> {
    fn from(array: GpuArray<T>) -> Self {
        array.into_buffer()
    }
}

/// Element types supported by [`reduce_sum`].
pub trait ReduceSum: DevicePod + Default + Sized {
    fn reduce_sum(input: &DeviceBuffer<Self>) -> Result<Self>;
}

impl ReduceSum for u32 {
    fn reduce_sum(input: &DeviceBuffer<Self>) -> Result<Self> {
        let output = DeviceBuffer::<Self>::new(1)?;
        RocPrim::open()?.reduce_sum_u32(input, &output)?;
        let mut host = [Self::default(); 1];
        output.copy_to_host(&mut host)?;
        Ok(host[0])
    }
}

impl ReduceSum for i32 {
    fn reduce_sum(input: &DeviceBuffer<Self>) -> Result<Self> {
        let output = DeviceBuffer::<Self>::new(1)?;
        RocPrim::open()?.reduce_sum_i32(input, &output)?;
        let mut host = [Self::default(); 1];
        output.copy_to_host(&mut host)?;
        Ok(host[0])
    }
}

impl ReduceSum for f32 {
    fn reduce_sum(input: &DeviceBuffer<Self>) -> Result<Self> {
        let output = DeviceBuffer::<Self>::new(1)?;
        RocPrim::open()?.reduce_sum_f32(input, &output)?;
        let mut host = [Self::default(); 1];
        output.copy_to_host(&mut host)?;
        Ok(host[0])
    }
}

/// Sums all elements in `input` and returns the scalar result on the host.
///
/// Supported element types are `u32`, `i32`, and `f32`.
pub fn reduce_sum<T>(input: &DeviceBuffer<T>) -> Result<T>
where
    T: ReduceSum,
{
    T::reduce_sum(input)
}

/// Element types supported by [`inclusive_scan`] and [`exclusive_scan`].
pub trait PrefixSum: DevicePod + Sized {
    fn inclusive_scan(input: &DeviceBuffer<Self>, output: &DeviceBuffer<Self>) -> Result<()>;
    fn exclusive_scan(
        input: &DeviceBuffer<Self>,
        output: &DeviceBuffer<Self>,
        initial_value: Self,
    ) -> Result<()>;
}

impl PrefixSum for u32 {
    fn inclusive_scan(input: &DeviceBuffer<Self>, output: &DeviceBuffer<Self>) -> Result<()> {
        RocPrim::open()?.inclusive_sum_u32(input, output)
    }

    fn exclusive_scan(
        input: &DeviceBuffer<Self>,
        output: &DeviceBuffer<Self>,
        initial_value: Self,
    ) -> Result<()> {
        RocPrim::open()?.exclusive_sum_u32(input, output, initial_value)
    }
}

impl PrefixSum for i32 {
    fn inclusive_scan(input: &DeviceBuffer<Self>, output: &DeviceBuffer<Self>) -> Result<()> {
        RocPrim::open()?.inclusive_sum_i32(input, output)
    }

    fn exclusive_scan(
        input: &DeviceBuffer<Self>,
        output: &DeviceBuffer<Self>,
        initial_value: Self,
    ) -> Result<()> {
        RocPrim::open()?.exclusive_sum_i32(input, output, initial_value)
    }
}

impl PrefixSum for f32 {
    fn inclusive_scan(input: &DeviceBuffer<Self>, output: &DeviceBuffer<Self>) -> Result<()> {
        RocPrim::open()?.inclusive_sum_f32(input, output)
    }

    fn exclusive_scan(
        input: &DeviceBuffer<Self>,
        output: &DeviceBuffer<Self>,
        initial_value: Self,
    ) -> Result<()> {
        RocPrim::open()?.exclusive_sum_f32(input, output, initial_value)
    }
}

/// Writes the inclusive prefix sum of `input` into `output`.
///
/// Supported element types are `u32`, `i32`, and `f32`. `output.len()` must
/// equal `input.len()`.
pub fn inclusive_scan<T>(input: &DeviceBuffer<T>, output: &DeviceBuffer<T>) -> Result<()>
where
    T: PrefixSum,
{
    T::inclusive_scan(input, output)
}

/// Writes the exclusive prefix sum of `input` into `output`.
///
/// Supported element types are `u32`, `i32`, and `f32`. `output.len()` must
/// equal `input.len()`.
pub fn exclusive_scan<T>(
    input: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
    initial_value: T,
) -> Result<()>
where
    T: PrefixSum,
{
    T::exclusive_scan(input, output, initial_value)
}

/// Element types supported by [`sort`].
pub trait Sort: DevicePod + Sized {
    fn sort(data: &mut DeviceBuffer<Self>) -> Result<()>;
}

impl Sort for u32 {
    fn sort(data: &mut DeviceBuffer<Self>) -> Result<()> {
        RocThrust::open()?.sort_u32(data)
    }
}

/// Sorts `data` in place in ascending order.
///
/// The current high-level sort supports `u32`.
pub fn sort<T>(data: &mut DeviceBuffer<T>) -> Result<()>
where
    T: Sort,
{
    T::sort(data)
}

/// Sorts `input` into `output` in ascending order.
///
/// This out-of-place helper uses rocPRIM and currently supports `u32`.
pub fn sort_keys_u32(input: &DeviceBuffer<u32>, output: &DeviceBuffer<u32>) -> Result<()> {
    RocPrim::open()?.sort_keys_u32(input, output)
}

/// Sorts `keys` in place and reorders `values` to preserve key/value pairs.
///
/// This helper uses rocThrust and currently supports `u32` keys and values.
pub fn sort_by_key_u32(keys: &mut DeviceBuffer<u32>, values: &mut DeviceBuffer<u32>) -> Result<()> {
    RocThrust::open()?.sort_by_key_u32(keys, values)
}

/// Removes consecutive duplicate `u32` values in place.
///
/// Returns the number of unique elements. Values after that count are
/// unspecified until overwritten by the caller.
pub fn unique_u32(data: &mut DeviceBuffer<u32>) -> Result<usize> {
    RocThrust::open()?.unique_u32(data)
}

/// Counts elements equal to `value` in a `u32` buffer.
pub fn count_eq_u32(data: &DeviceBuffer<u32>, value: u32) -> Result<usize> {
    RocThrust::open()?.count_u32(data, value)
}

/// Selects `input[i]` into `output` whenever `flags[i] != 0`.
///
/// The number of selected elements is written to `selected_count[0]`.
pub fn select_flagged_u32(
    input: &DeviceBuffer<u32>,
    flags: &DeviceBuffer<u8>,
    output: &DeviceBuffer<u32>,
    selected_count: &DeviceBuffer<u32>,
) -> Result<()> {
    RocPrim::open()?.select_flagged_u32(input, flags, output, selected_count)
}

/// Adds `addend` to every `input` element and writes the result to `output`.
///
/// This is the first map-like helper over the existing rocPRIM shim. General
/// closure-based GPU maps remain future work.
pub fn map_add_u32(
    input: &DeviceBuffer<u32>,
    output: &DeviceBuffer<u32>,
    addend: u32,
) -> Result<()> {
    RocPrim::open()?.transform_add_u32(input, output, addend)
}

/// Fills a device buffer with zero bytes.
pub fn fill_zero<T>(buffer: &DeviceBuffer<T>) -> Result<()> {
    Ok(buffer.set_zero()?)
}

/// Fills a device buffer with a byte pattern.
///
/// Prefer [`fill_zero`] for typed initialization. Nonzero byte patterns are best
/// suited to byte buffers and debugging sentinels.
pub fn fill_bytes<T>(buffer: &DeviceBuffer<T>, value: u8) -> Result<()> {
    Ok(buffer.memset(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_reduce_sum_smoke_if_available() {
        if !RocPrim::is_available() {
            return;
        }

        let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4]).expect("u32 upload");
        assert_eq!(reduce_sum(&input).expect("u32 reduce"), 10);

        let signed = DeviceBuffer::from_slice(&[-2i32, 4, 9]).expect("i32 upload");
        assert_eq!(reduce_sum(&signed).expect("i32 reduce"), 11);

        let floats = DeviceBuffer::from_slice(&[1.0f32, 2.5, -0.5]).expect("f32 upload");
        let sum = reduce_sum(&floats).expect("f32 reduce");
        assert!((sum - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gpu_scan_select_map_and_fill_smoke_if_available() {
        if !RocPrim::is_available() {
            return;
        }

        let input = DeviceBuffer::from_slice(&[1u32, 2, 3, 4]).expect("u32 upload");
        let inclusive = DeviceBuffer::<u32>::new(input.len()).expect("inclusive output");
        inclusive_scan(&input, &inclusive).expect("inclusive scan");
        assert_eq!(
            inclusive.copy_to_vec().expect("inclusive download"),
            [1, 3, 6, 10]
        );

        let exclusive = DeviceBuffer::<u32>::new(input.len()).expect("exclusive output");
        exclusive_scan(&input, &exclusive, 0).expect("exclusive scan");
        assert_eq!(
            exclusive.copy_to_vec().expect("exclusive download"),
            [0, 1, 3, 6]
        );

        let flags = DeviceBuffer::from_slice(&[1u8, 0, 1, 0]).expect("flags upload");
        let selected = DeviceBuffer::<u32>::new(input.len()).expect("selected output");
        let selected_count = DeviceBuffer::<u32>::new(1).expect("selected count");
        select_flagged_u32(&input, &flags, &selected, &selected_count).expect("select flagged");
        assert_eq!(
            selected_count
                .copy_to_vec()
                .expect("selected count download"),
            [2]
        );
        assert_eq!(
            &selected.copy_to_vec().expect("selected download")[..2],
            [1, 3]
        );

        let mapped = DeviceBuffer::<u32>::new(input.len()).expect("mapped output");
        map_add_u32(&input, &mapped, 5).expect("map add");
        assert_eq!(mapped.copy_to_vec().expect("mapped download"), [6, 7, 8, 9]);

        fill_zero(&mapped).expect("fill zero");
        assert_eq!(mapped.copy_to_vec().expect("zero download"), [0, 0, 0, 0]);

        let bytes = DeviceBuffer::from_slice(&[0u8; 4]).expect("byte upload");
        fill_bytes(&bytes, 0xa5).expect("fill bytes");
        assert_eq!(bytes.copy_to_vec().expect("byte download"), [0xa5; 4]);
    }

    #[test]
    fn gpu_array_methods_smoke_if_available() {
        if !RocPrim::is_available() {
            return;
        }

        let input = GpuArray::from_slice(&[1u32, 2, 3, 4]).expect("array upload");
        assert_eq!(input.len(), 4);
        assert_eq!(input.sum().expect("array reduce"), 10);
        assert_eq!(
            input
                .exclusive_scan(0)
                .expect("array exclusive scan")
                .to_vec()
                .expect("scan download"),
            [0, 1, 3, 6]
        );
        assert_eq!(
            input
                .map_add(5)
                .expect("array map add")
                .to_vec()
                .expect("map download"),
            [6, 7, 8, 9]
        );

        let zeros = GpuArray::<u32>::zeros(4).expect("zero array");
        assert_eq!(zeros.to_vec().expect("zero download"), [0, 0, 0, 0]);
    }

    #[test]
    fn gpu_sort_smoke_if_available() {
        if !RocThrust::is_available() {
            return;
        }

        let mut data = DeviceBuffer::from_slice(&[4u32, 1, 3, 1]).expect("sort upload");
        sort(&mut data).expect("in-place sort");
        assert_eq!(data.copy_to_vec().expect("sort download"), [1, 1, 3, 4]);

        let n = unique_u32(&mut data).expect("unique");
        assert_eq!(n, 3);
        assert_eq!(
            &data.copy_to_vec().expect("unique download")[..n],
            [1, 3, 4]
        );

        assert_eq!(count_eq_u32(&data, 1).expect("count"), 1);

        let mut keys = DeviceBuffer::from_slice(&[3u32, 1, 2]).expect("keys upload");
        let mut values = DeviceBuffer::from_slice(&[30u32, 10, 20]).expect("values upload");
        sort_by_key_u32(&mut keys, &mut values).expect("sort by key");
        assert_eq!(keys.copy_to_vec().expect("keys download"), [1, 2, 3]);
        assert_eq!(values.copy_to_vec().expect("values download"), [10, 20, 30]);

        let mut array = GpuArray::from_slice(&[2u32, 4, 1, 1]).expect("array sort upload");
        array.sort().expect("array sort");
        assert_eq!(array.to_vec().expect("array sort download"), [1, 1, 2, 4]);
        assert_eq!(array.count_eq(1).expect("array count"), 2);
    }
}
