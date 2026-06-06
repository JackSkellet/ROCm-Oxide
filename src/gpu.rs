//! Small host-side GPU algorithms built on ROCm library interop.
//!
//! This module is the ergonomic layer above [`crate::RocPrim`] and
//! [`crate::RocThrust`]. It is intentionally small: ROCm-Oxide still exposes the
//! lower-level wrappers for callers that need explicit temporary-storage or
//! stream control.

use crate::{DeviceBuffer, DevicePod, Error, Result, RocPrim, RocThrust};
use std::ops::{Deref, DerefMut};

/// A small method-oriented wrapper around [`DeviceBuffer`].
///
/// `GpuArray<T>` is intended for approachable host-side code and autocomplete:
/// construct an array, call methods on it, and copy values back when needed.
/// It does not own a separate runtime or scheduler; methods delegate to the
/// free functions in this module and the underlying `DeviceBuffer`.
///
/// ```rust,ignore
/// use rocm_oxide::gpu;
///
/// let input = gpu::array([1u32, 2, 3, 4])?;
/// let sum = input.sum()?;
/// let mapped = input.add_scalar(8)?;
/// let scanned = input.exclusive_scan(0)?;
///
/// assert_eq!(sum, 10);
/// assert_eq!(mapped.to_list()?, [9, 10, 11, 12]);
/// assert_eq!(scanned.to_list()?, [0, 1, 3, 6]);
/// ```
pub struct GpuArray<T> {
    buffer: DeviceBuffer<T>,
}

impl<T> GpuArray<T> {
    /// Allocate an uninitialized device array.
    ///
    /// This mirrors [`DeviceBuffer::new`] for output arrays that a kernel or
    /// library call will fill before the host reads them.
    pub fn new(len: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::<T>::new(len)?,
        })
    }

    /// Allocate an uninitialized device array.
    ///
    /// `empty` is an autocomplete-friendly alias for [`new`](Self::new), named
    /// after the familiar NumPy/Python convention.
    pub fn empty(len: usize) -> Result<Self> {
        Self::new(len)
    }

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

    /// Number of elements in the array.
    ///
    /// Alias for [`len`](Self::len) that reads naturally beside Python and
    /// NumPy examples.
    pub fn size(&self) -> usize {
        self.len()
    }

    /// One-dimensional array shape.
    pub fn shape(&self) -> [usize; 1] {
        [self.len()]
    }

    /// Size of one element in bytes.
    pub fn element_size(&self) -> usize {
        std::mem::size_of::<T>()
    }

    /// Total logical element storage size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len().saturating_mul(self.element_size())
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

    /// Copy this device array into another same-length device array.
    pub fn copy_to(&self, output: &Self) -> Result<()> {
        Ok(self.buffer.copy_to_device(output.as_buffer())?)
    }

    /// Copy another same-length device array into this array.
    pub fn copy_from(&self, input: &Self) -> Result<()> {
        Ok(self.buffer.copy_from_device(input.as_buffer())?)
    }

    /// Return a device-to-device copy of this array.
    pub fn cloned(&self) -> Result<Self> {
        let output = Self::new(self.len())?;
        self.copy_to(&output)?;
        Ok(output)
    }

    /// Return a device-to-device copy of this array.
    ///
    /// Alias for [`cloned`](Self::cloned) that matches the familiar Python
    /// array spelling.
    pub fn copy(&self) -> Result<Self> {
        self.cloned()
    }
}

/// A row-major 2-D wrapper around [`GpuArray`].
///
/// `GpuArray2D<T>` keeps shape metadata next to the device allocation while
/// preserving the same generated-binding compatibility as [`GpuArray<T>`].
/// Passing a `GpuArray2D<T>` to generated kernels works through
/// `AsRef<DeviceBuffer<T>>`; kernels still see a flat contiguous buffer.
pub struct GpuArray2D<T> {
    array: GpuArray<T>,
    rows: usize,
    cols: usize,
}

impl<T> GpuArray2D<T> {
    /// Allocate an uninitialized row-major 2-D device array.
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        Self::empty(rows, cols)
    }

    /// Allocate an uninitialized row-major 2-D device array.
    pub fn empty(rows: usize, cols: usize) -> Result<Self> {
        let len = checked_shape_len("GpuArray2D::empty", rows, cols)?;
        Ok(Self {
            array: GpuArray::empty(len)?,
            rows,
            cols,
        })
    }

    /// Wrap an existing flat device array with 2-D shape metadata.
    pub fn from_array(array: GpuArray<T>, rows: usize, cols: usize) -> Result<Self> {
        let expected = checked_shape_len("GpuArray2D::from_array", rows, cols)?;
        expect_shape_len("GpuArray2D::from_array", array.len(), expected, rows, cols)?;
        Ok(Self { array, rows, cols })
    }

    /// Return the underlying flat array.
    pub fn into_array(self) -> GpuArray<T> {
        self.array
    }

    /// Borrow the underlying flat array.
    pub fn as_array(&self) -> &GpuArray<T> {
        &self.array
    }

    /// Mutably borrow the underlying flat array.
    pub fn as_mut_array(&mut self) -> &mut GpuArray<T> {
        &mut self.array
    }

    /// Borrow the underlying device buffer.
    pub fn as_buffer(&self) -> &DeviceBuffer<T> {
        self.array.as_buffer()
    }

    /// Mutably borrow the underlying device buffer.
    pub fn as_mut_buffer(&mut self) -> &mut DeviceBuffer<T> {
        self.array.as_mut_buffer()
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Image-style width alias for [`cols`](Self::cols).
    pub fn width(&self) -> usize {
        self.cols()
    }

    /// Image-style height alias for [`rows`](Self::rows).
    pub fn height(&self) -> usize {
        self.rows()
    }

    /// Number of elements in the flat allocation.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Number of elements in the flat allocation.
    pub fn size(&self) -> usize {
        self.len()
    }

    /// Row-major shape as `[rows, cols]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// Size of one element in bytes.
    pub fn element_size(&self) -> usize {
        self.array.element_size()
    }

    /// Total logical element storage size in bytes.
    pub fn byte_len(&self) -> usize {
        self.array.byte_len()
    }

    /// Returns `true` when the flat allocation has no elements.
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Fill the array with zero bytes.
    pub fn fill_zero(&self) -> Result<()> {
        self.array.fill_zero()
    }

    /// Fill the array with a byte pattern.
    pub fn fill_bytes(&self, value: u8) -> Result<()> {
        self.array.fill_bytes(value)
    }

    /// Copy this device array into another same-shape device array.
    pub fn copy_to(&self, output: &Self) -> Result<()> {
        self.expect_same_shape("GpuArray2D::copy_to", output)?;
        self.array.copy_to(&output.array)
    }

    /// Copy another same-shape device array into this array.
    pub fn copy_from(&self, input: &Self) -> Result<()> {
        self.expect_same_shape("GpuArray2D::copy_from", input)?;
        self.array.copy_from(&input.array)
    }

    /// Return a device-to-device copy of this array with the same shape.
    pub fn cloned(&self) -> Result<Self> {
        Ok(Self {
            array: self.array.cloned()?,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Return a device-to-device copy of this array with the same shape.
    pub fn copy(&self) -> Result<Self> {
        self.cloned()
    }

    fn expect_same_shape(&self, operation: &str, other: &Self) -> Result<()> {
        if self.shape() == other.shape() {
            Ok(())
        } else {
            Err(Error::InvalidLaunch(format!(
                "{operation} shape mismatch: left is {:?}, right is {:?}",
                self.shape(),
                other.shape()
            )))
        }
    }
}

impl<T: DevicePod> GpuArray<T> {
    /// Allocate a zero-filled device array.
    pub fn zeros(len: usize) -> Result<Self> {
        let buffer = DeviceBuffer::<T>::new(len)?;
        buffer.set_zero()?;
        Ok(Self { buffer })
    }

    /// Allocate a zero-filled device array.
    ///
    /// Alias for [`zeros`](Self::zeros) that reads naturally beside
    /// [`empty`](Self::empty).
    pub fn zeroed(len: usize) -> Result<Self> {
        Self::zeros(len)
    }
}

impl<T: Copy> GpuArray<T> {
    /// Allocate a device array and upload `input`.
    pub fn from_slice(input: &[T]) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::from_slice(input)?,
        })
    }

    /// Allocate a one-element device array and upload `value`.
    pub fn from_value(value: T) -> Result<Self> {
        Self::from_slice(&[value])
    }

    /// Allocate a device array and upload `input`.
    pub fn from_vec(input: Vec<T>) -> Result<Self> {
        Self::from_slice(input.as_slice())
    }

    /// Allocate a device array and upload values from an iterator.
    ///
    /// This is a fallible alternative to `collect::<Vec<_>>()` followed by
    /// [`from_slice`](Self::from_slice).
    pub fn from_values(values: impl IntoIterator<Item = T>) -> Result<Self> {
        let host = values.into_iter().collect::<Vec<_>>();
        Self::from_slice(host.as_slice())
    }

    /// Allocate a device array containing `len` copies of `value`.
    pub fn repeat(value: T, len: usize) -> Result<Self> {
        let host = vec![value; len];
        Self::from_slice(host.as_slice())
    }

    /// Allocate a device array containing `len` copies of `value`.
    ///
    /// Alias for [`repeat`](Self::repeat) using the familiar `full(len, value)`
    /// constructor shape.
    pub fn full(len: usize, value: T) -> Result<Self> {
        Self::repeat(value, len)
    }

    /// Copy `input` into this existing device array.
    pub fn copy_from_slice(&self, input: &[T]) -> Result<()> {
        Ok(self.buffer.copy_from_host(input)?)
    }

    /// Copy `input` into this existing device array.
    ///
    /// Alias for [`copy_from_slice`](Self::copy_from_slice) with a shorter name
    /// for script-like code.
    pub fn upload(&self, input: &[T]) -> Result<()> {
        self.copy_from_slice(input)
    }

    /// Copy `input` into this existing device array.
    ///
    /// Alias for [`upload`](Self::upload) for script-like code that treats an
    /// existing GPU allocation as an assignable array.
    pub fn assign(&self, input: &[T]) -> Result<()> {
        self.upload(input)
    }

    /// Copy this device array into an existing host slice.
    pub fn copy_to_slice(&self, output: &mut [T]) -> Result<()> {
        Ok(self.buffer.copy_to_host(output)?)
    }

    /// Copy this one-element device array back to the host.
    pub fn read(&self) -> Result<T>
    where
        T: Default,
    {
        expect_len("read", self.len(), 1)?;
        Ok(self.to_vec()?[0])
    }

    /// Copy this one-element device array back to the host.
    ///
    /// Alias for [`read`](Self::read) named after NumPy scalar extraction.
    pub fn item(&self) -> Result<T>
    where
        T: Default,
    {
        self.read()
    }

    /// Copy `value` into this one-element device array.
    pub fn write(&self, value: T) -> Result<()> {
        expect_len("write", self.len(), 1)?;
        self.copy_from_slice(&[value])
    }
}

impl<T: Copy + Default> GpuArray<T> {
    /// Copy this device array back to host memory.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        Ok(self.buffer.copy_to_vec()?)
    }

    /// Copy this device array back to host memory.
    ///
    /// Alias for [`to_vec`](Self::to_vec) with a Python-like collection name.
    pub fn to_list(&self) -> Result<Vec<T>> {
        self.to_vec()
    }

    /// Copy this device array back to host memory.
    ///
    /// Alias for [`to_vec`](Self::to_vec) with a data-science-style name.
    pub fn download(&self) -> Result<Vec<T>> {
        self.to_vec()
    }
}

impl<T: DevicePod> GpuArray2D<T> {
    /// Allocate a zero-filled row-major 2-D device array.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        let len = checked_shape_len("GpuArray2D::zeros", rows, cols)?;
        Ok(Self {
            array: GpuArray::zeros(len)?,
            rows,
            cols,
        })
    }

    /// Allocate a zero-filled row-major 2-D device array.
    pub fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        Self::zeros(rows, cols)
    }
}

impl<T: Copy> GpuArray2D<T> {
    /// Allocate a row-major 2-D device array and upload `input`.
    pub fn from_slice(rows: usize, cols: usize, input: &[T]) -> Result<Self> {
        let expected = checked_shape_len("GpuArray2D::from_slice", rows, cols)?;
        expect_shape_len("GpuArray2D::from_slice", input.len(), expected, rows, cols)?;
        Ok(Self {
            array: GpuArray::from_slice(input)?,
            rows,
            cols,
        })
    }

    /// Allocate a row-major 2-D device array and upload values from an iterator.
    pub fn from_values(
        rows: usize,
        cols: usize,
        values: impl IntoIterator<Item = T>,
    ) -> Result<Self> {
        let host = values.into_iter().collect::<Vec<_>>();
        Self::from_slice(rows, cols, host.as_slice())
    }

    /// Allocate a row-major 2-D device array containing copies of `value`.
    pub fn repeat(rows: usize, cols: usize, value: T) -> Result<Self> {
        let len = checked_shape_len("GpuArray2D::repeat", rows, cols)?;
        Ok(Self {
            array: GpuArray::repeat(value, len)?,
            rows,
            cols,
        })
    }

    /// Allocate a row-major 2-D device array containing copies of `value`.
    pub fn full(rows: usize, cols: usize, value: T) -> Result<Self> {
        Self::repeat(rows, cols, value)
    }

    /// Copy `input` into this existing row-major 2-D device array.
    pub fn copy_from_slice(&self, input: &[T]) -> Result<()> {
        expect_shape_len(
            "GpuArray2D::copy_from_slice",
            input.len(),
            self.len(),
            self.rows,
            self.cols,
        )?;
        self.array.copy_from_slice(input)
    }

    /// Copy `input` into this existing row-major 2-D device array.
    pub fn upload(&self, input: &[T]) -> Result<()> {
        self.copy_from_slice(input)
    }

    /// Copy `input` into this existing row-major 2-D device array.
    pub fn assign(&self, input: &[T]) -> Result<()> {
        self.upload(input)
    }

    /// Copy this row-major 2-D device array into an existing flat host slice.
    pub fn copy_to_slice(&self, output: &mut [T]) -> Result<()> {
        expect_shape_len(
            "GpuArray2D::copy_to_slice",
            output.len(),
            self.len(),
            self.rows,
            self.cols,
        )?;
        self.array.copy_to_slice(output)
    }
}

impl<T: Copy + Default> GpuArray2D<T> {
    /// Copy this row-major 2-D device array back to a flat host vector.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        self.array.to_vec()
    }

    /// Copy this row-major 2-D device array back to a flat host vector.
    pub fn to_list(&self) -> Result<Vec<T>> {
        self.to_vec()
    }

    /// Copy this row-major 2-D device array back to a flat host vector.
    pub fn download(&self) -> Result<Vec<T>> {
        self.to_vec()
    }

    /// Copy this device array back as nested row vectors.
    pub fn to_rows(&self) -> Result<Vec<Vec<T>>> {
        if self.cols == 0 {
            return Ok((0..self.rows).map(|_| Vec::new()).collect());
        }

        Ok(self
            .to_vec()?
            .chunks(self.cols)
            .map(|row| row.to_vec())
            .collect())
    }
}

impl<T: ReduceSum> GpuArray<T> {
    /// Sum all elements and return the scalar result on the host.
    pub fn sum(&self) -> Result<T> {
        reduce_sum(&self.buffer)
    }
}

impl<T: PrefixSum> GpuArray<T> {
    /// Write the inclusive prefix sum of this array into `output`.
    pub fn inclusive_scan_into(&self, output: &GpuArray<T>) -> Result<()> {
        inclusive_scan(&self.buffer, output)
    }

    /// Return an array containing the inclusive prefix sum of this array.
    pub fn inclusive_scan(&self) -> Result<Self> {
        let output = DeviceBuffer::<T>::new(self.len())?;
        inclusive_scan(&self.buffer, &output)?;
        Ok(Self { buffer: output })
    }

    /// Write the inclusive prefix sum of this array into `output`.
    ///
    /// Alias for [`inclusive_scan_into`](Self::inclusive_scan_into) using the
    /// NumPy/Python spelling.
    pub fn cumsum_into(&self, output: &GpuArray<T>) -> Result<()> {
        self.inclusive_scan_into(output)
    }

    /// Return an array containing the inclusive prefix sum of this array.
    ///
    /// Alias for [`inclusive_scan`](Self::inclusive_scan) using the
    /// NumPy/Python spelling.
    pub fn cumsum(&self) -> Result<Self> {
        self.inclusive_scan()
    }

    /// Write the exclusive prefix sum of this array into `output`.
    pub fn exclusive_scan_into(&self, output: &GpuArray<T>, initial_value: T) -> Result<()> {
        exclusive_scan(&self.buffer, output, initial_value)
    }

    /// Return an array containing the exclusive prefix sum of this array.
    pub fn exclusive_scan(&self, initial_value: T) -> Result<Self> {
        let output = DeviceBuffer::<T>::new(self.len())?;
        exclusive_scan(&self.buffer, &output, initial_value)?;
        Ok(Self { buffer: output })
    }
}

impl GpuArray<u32> {
    /// Add `addend` to every element and write the result into `output`.
    pub fn map_add_into(&self, output: &GpuArray<u32>, addend: u32) -> Result<()> {
        map_add_u32(&self.buffer, output, addend)
    }

    /// Add `addend` to every element and return the mapped output array.
    pub fn map_add(&self, addend: u32) -> Result<Self> {
        let output = DeviceBuffer::<u32>::new(self.len())?;
        map_add_u32(&self.buffer, &output, addend)?;
        Ok(Self { buffer: output })
    }

    /// Add `addend` to every element and write the result into `output`.
    pub fn add_scalar_into(&self, output: &GpuArray<u32>, addend: u32) -> Result<()> {
        self.map_add_into(output, addend)
    }

    /// Add `addend` to every element and return the mapped output array.
    pub fn add_scalar(&self, addend: u32) -> Result<Self> {
        self.map_add(addend)
    }

    /// Sort this array in place.
    pub fn sort(&mut self) -> Result<()> {
        sort(&mut self.buffer)
    }

    /// Sort this keys array and reorder `values` to preserve key/value pairs.
    pub fn sort_by_key(&mut self, values: &mut GpuArray<u32>) -> Result<()> {
        if values.len() != self.len() {
            return Err(Error::InvalidLaunch(format!(
                "GpuArray::sort_by_key length mismatch: keys has {} elements, values has {}",
                self.len(),
                values.len()
            )));
        }
        sort_by_key_u32(&mut self.buffer, values.as_mut_buffer())
    }

    /// Return a sorted copy of this array.
    pub fn sorted(&self) -> Result<Self> {
        let mut output = DeviceBuffer::<u32>::new(self.len())?;
        self.buffer.copy_to_device(&output)?;
        sort(&mut output)?;
        Ok(Self { buffer: output })
    }

    /// Return a sorted copy using rocPRIM's out-of-place key sort.
    pub fn sorted_keys(&self) -> Result<Self> {
        let output = DeviceBuffer::<u32>::new(self.len())?;
        sort_keys_u32(&self.buffer, &output)?;
        Ok(Self { buffer: output })
    }

    /// Return a sorted copy and the number of unique values in that copy.
    ///
    /// Only the first `unique_count` elements are unique values.
    pub fn sorted_unique(&self) -> Result<(Self, usize)> {
        let mut output = self.copy()?;
        let unique_count = output.sort_unique()?;
        Ok((output, unique_count))
    }

    /// Sort in place, remove consecutive duplicate values, and return the
    /// number of unique values.
    ///
    /// Values after the returned count are unspecified until overwritten.
    pub fn sort_unique(&mut self) -> Result<usize> {
        sort_unique_u32(&mut self.buffer)
    }

    /// Count elements equal to `value`.
    pub fn count_eq(&self, value: u32) -> Result<usize> {
        count_eq_u32(&self.buffer, value)
    }

    /// Return `true` when at least one element equals `value`.
    pub fn contains(&self, value: u32) -> Result<bool> {
        contains_eq_u32(&self.buffer, value)
    }

    /// Remove consecutive duplicate values in place and return the unique count.
    ///
    /// Values after the returned count are unspecified until overwritten.
    pub fn unique_consecutive(&mut self) -> Result<usize> {
        unique_u32(&mut self.buffer)
    }

    /// Select elements whose matching `flags[i]` is nonzero.
    ///
    /// Returns the output array and the number of valid selected elements. The
    /// returned array has the same allocation length as `self`; only
    /// `0..selected_count` contains selected values.
    pub fn select_flagged(&self, flags: &GpuArray<u8>) -> Result<(Self, usize)> {
        if flags.len() != self.len() {
            return Err(Error::InvalidLaunch(format!(
                "GpuArray::select_flagged length mismatch: input has {} elements, flags has {}",
                self.len(),
                flags.len()
            )));
        }

        let output = DeviceBuffer::<u32>::new(self.len())?;
        let selected_count = DeviceBuffer::<u32>::new(1)?;
        select_flagged_u32(&self.buffer, flags.as_buffer(), &output, &selected_count)?;
        let mut count = [0u32; 1];
        selected_count.copy_to_host(&mut count)?;
        Ok((Self { buffer: output }, count[0] as usize))
    }

    /// Alias for [`select_flagged`](Self::select_flagged) with a NumPy-style
    /// selection name.
    pub fn where_flags(&self, flags: &GpuArray<u8>) -> Result<(Self, usize)> {
        self.select_flagged(flags)
    }

    /// Alias for [`select_flagged`](Self::select_flagged) with a compacting
    /// algorithm name.
    pub fn compact_by_flags(&self, flags: &GpuArray<u8>) -> Result<(Self, usize)> {
        self.select_flagged(flags)
    }
}

fn expect_len(operation: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidLaunch(format!(
            "GpuArray::{operation} expects {expected} element(s), got {actual}"
        )))
    }
}

fn checked_shape_len(operation: &str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols).ok_or_else(|| {
        Error::InvalidLaunch(format!(
            "{operation} shape overflow: {rows} rows * {cols} columns"
        ))
    })
}

fn expect_shape_len(
    operation: &str,
    actual: usize,
    expected: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidLaunch(format!(
            "{operation} shape mismatch: [{rows}, {cols}] expects {expected} element(s), got {actual}"
        )))
    }
}

/// Allocate an uninitialized GPU array.
pub fn empty<T>(len: usize) -> Result<GpuArray<T>> {
    GpuArray::empty(len)
}

/// Allocate an uninitialized row-major 2-D GPU array.
pub fn empty_2d<T>(rows: usize, cols: usize) -> Result<GpuArray2D<T>> {
    GpuArray2D::empty(rows, cols)
}

/// Allocate a zero-filled GPU array.
pub fn zeros<T: DevicePod>(len: usize) -> Result<GpuArray<T>> {
    GpuArray::zeros(len)
}

/// Allocate a zero-filled row-major 2-D GPU array.
pub fn zeros_2d<T: DevicePod>(rows: usize, cols: usize) -> Result<GpuArray2D<T>> {
    GpuArray2D::zeros(rows, cols)
}

/// Allocate a device array and upload values from an iterator.
pub fn array<T: Copy>(values: impl IntoIterator<Item = T>) -> Result<GpuArray<T>> {
    GpuArray::from_values(values)
}

/// Allocate a row-major 2-D device array and upload values from an iterator.
pub fn array_2d<T: Copy>(
    rows: usize,
    cols: usize,
    values: impl IntoIterator<Item = T>,
) -> Result<GpuArray2D<T>> {
    GpuArray2D::from_values(rows, cols, values)
}

/// Allocate a device array containing `len` copies of `value`.
pub fn full<T: Copy>(len: usize, value: T) -> Result<GpuArray<T>> {
    GpuArray::full(len, value)
}

/// Allocate a row-major 2-D device array containing copies of `value`.
pub fn full_2d<T: Copy>(rows: usize, cols: usize, value: T) -> Result<GpuArray2D<T>> {
    GpuArray2D::full(rows, cols, value)
}

/// Copy a device buffer or GPU array into another same-length device buffer.
pub fn copy_into<T>(
    input: &impl AsRef<DeviceBuffer<T>>,
    output: &impl AsRef<DeviceBuffer<T>>,
) -> Result<()> {
    Ok(input.as_ref().copy_to_device(output.as_ref())?)
}

/// Return a device-to-device copy as a [`GpuArray`].
pub fn copy<T>(input: &impl AsRef<DeviceBuffer<T>>) -> Result<GpuArray<T>> {
    let output = GpuArray::empty(input.as_ref().len())?;
    copy_into(input, &output)?;
    Ok(output)
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

impl<T> AsRef<DeviceBuffer<T>> for GpuArray2D<T> {
    fn as_ref(&self) -> &DeviceBuffer<T> {
        self.as_buffer()
    }
}

impl<T> AsMut<DeviceBuffer<T>> for GpuArray2D<T> {
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

impl<T> From<GpuArray2D<T>> for GpuArray<T> {
    fn from(array: GpuArray2D<T>) -> Self {
        array.into_array()
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
pub fn reduce_sum<T>(input: &impl AsRef<DeviceBuffer<T>>) -> Result<T>
where
    T: ReduceSum,
{
    T::reduce_sum(input.as_ref())
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
pub fn inclusive_scan<T>(
    input: &impl AsRef<DeviceBuffer<T>>,
    output: &impl AsRef<DeviceBuffer<T>>,
) -> Result<()>
where
    T: PrefixSum,
{
    T::inclusive_scan(input.as_ref(), output.as_ref())
}

/// Writes the exclusive prefix sum of `input` into `output`.
///
/// Supported element types are `u32`, `i32`, and `f32`. `output.len()` must
/// equal `input.len()`.
pub fn exclusive_scan<T>(
    input: &impl AsRef<DeviceBuffer<T>>,
    output: &impl AsRef<DeviceBuffer<T>>,
    initial_value: T,
) -> Result<()>
where
    T: PrefixSum,
{
    T::exclusive_scan(input.as_ref(), output.as_ref(), initial_value)
}

/// Returns the inclusive prefix sum of `input` as a new [`GpuArray`].
///
/// Alias for [`inclusive_scan`] using the NumPy/Python spelling.
pub fn cumsum<T>(input: &impl AsRef<DeviceBuffer<T>>) -> Result<GpuArray<T>>
where
    T: PrefixSum,
{
    let output = GpuArray::<T>::empty(input.as_ref().len())?;
    inclusive_scan(input, &output)?;
    Ok(output)
}

/// Returns the exclusive prefix sum of `input` as a new [`GpuArray`].
pub fn exclusive_cumsum<T>(
    input: &impl AsRef<DeviceBuffer<T>>,
    initial_value: T,
) -> Result<GpuArray<T>>
where
    T: PrefixSum,
{
    let output = GpuArray::<T>::empty(input.as_ref().len())?;
    exclusive_scan(input, &output, initial_value)?;
    Ok(output)
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
pub fn sort<T>(data: &mut impl AsMut<DeviceBuffer<T>>) -> Result<()>
where
    T: Sort,
{
    T::sort(data.as_mut())
}

/// Sorts `input` into `output` in ascending order.
///
/// This out-of-place helper uses rocPRIM and currently supports `u32`.
pub fn sort_keys_u32(
    input: &impl AsRef<DeviceBuffer<u32>>,
    output: &impl AsRef<DeviceBuffer<u32>>,
) -> Result<()> {
    RocPrim::open()?.sort_keys_u32(input.as_ref(), output.as_ref())
}

/// Sorts `keys` in place and reorders `values` to preserve key/value pairs.
///
/// This helper uses rocThrust and currently supports `u32` keys and values.
pub fn sort_by_key_u32(
    keys: &mut impl AsMut<DeviceBuffer<u32>>,
    values: &mut impl AsMut<DeviceBuffer<u32>>,
) -> Result<()> {
    RocThrust::open()?.sort_by_key_u32(keys.as_mut(), values.as_mut())
}

/// Removes consecutive duplicate `u32` values in place.
///
/// Returns the number of unique elements. Values after that count are
/// unspecified until overwritten by the caller.
pub fn unique_u32(data: &mut impl AsMut<DeviceBuffer<u32>>) -> Result<usize> {
    RocThrust::open()?.unique_u32(data.as_mut())
}

/// Counts elements equal to `value` in a `u32` buffer.
pub fn count_eq_u32(data: &impl AsRef<DeviceBuffer<u32>>, value: u32) -> Result<usize> {
    RocThrust::open()?.count_u32(data.as_ref(), value)
}

/// Returns `true` when at least one `u32` element equals `value`.
pub fn contains_eq_u32(data: &impl AsRef<DeviceBuffer<u32>>, value: u32) -> Result<bool> {
    Ok(count_eq_u32(data, value)? != 0)
}

/// Sorts `data` in place, removes consecutive duplicates, and returns the
/// number of unique values.
///
/// Values after the returned count are unspecified until overwritten.
pub fn sort_unique_u32(data: &mut impl AsMut<DeviceBuffer<u32>>) -> Result<usize> {
    sort(data)?;
    unique_u32(data)
}

/// Returns a sorted copy and the number of unique values in that copy.
///
/// Only the first returned `usize` elements are unique values.
pub fn sorted_unique_u32(input: &impl AsRef<DeviceBuffer<u32>>) -> Result<(GpuArray<u32>, usize)> {
    let mut output = copy(input)?;
    let unique_count = sort_unique_u32(&mut output)?;
    Ok((output, unique_count))
}

/// Selects `input[i]` into `output` whenever `flags[i] != 0`.
///
/// The number of selected elements is written to `selected_count[0]`.
pub fn select_flagged_u32(
    input: &impl AsRef<DeviceBuffer<u32>>,
    flags: &impl AsRef<DeviceBuffer<u8>>,
    output: &impl AsRef<DeviceBuffer<u32>>,
    selected_count: &impl AsRef<DeviceBuffer<u32>>,
) -> Result<()> {
    RocPrim::open()?.select_flagged_u32(
        input.as_ref(),
        flags.as_ref(),
        output.as_ref(),
        selected_count.as_ref(),
    )
}

/// Returns selected `input` values where `flags[i] != 0`.
///
/// The returned array has the same allocation length as `input`; only
/// `0..selected_count` contains selected values.
pub fn where_flags_u32(
    input: &impl AsRef<DeviceBuffer<u32>>,
    flags: &impl AsRef<DeviceBuffer<u8>>,
) -> Result<(GpuArray<u32>, usize)> {
    if flags.as_ref().len() != input.as_ref().len() {
        return Err(Error::InvalidLaunch(format!(
            "where_flags_u32 length mismatch: input has {} elements, flags has {}",
            input.as_ref().len(),
            flags.as_ref().len()
        )));
    }

    let output = GpuArray::<u32>::empty(input.as_ref().len())?;
    let selected_count = DeviceBuffer::<u32>::new(1)?;
    select_flagged_u32(input, flags, &output, &selected_count)?;
    let mut count = [0u32; 1];
    selected_count.copy_to_host(&mut count)?;
    Ok((output, count[0] as usize))
}

/// Adds `addend` to every `input` element and writes the result to `output`.
///
/// This is the first map-like helper over the existing rocPRIM shim. General
/// closure-based GPU maps remain future work.
pub fn map_add_u32(
    input: &impl AsRef<DeviceBuffer<u32>>,
    output: &impl AsRef<DeviceBuffer<u32>>,
    addend: u32,
) -> Result<()> {
    RocPrim::open()?.transform_add_u32(input.as_ref(), output.as_ref(), addend)
}

/// Adds `addend` to every `input` element and returns the mapped output.
pub fn add_scalar_u32(input: &impl AsRef<DeviceBuffer<u32>>, addend: u32) -> Result<GpuArray<u32>> {
    let output = GpuArray::<u32>::empty(input.as_ref().len())?;
    map_add_u32(input, &output, addend)?;
    Ok(output)
}

/// Fills a device buffer with zero bytes.
pub fn fill_zero<T>(buffer: &impl AsRef<DeviceBuffer<T>>) -> Result<()> {
    Ok(buffer.as_ref().set_zero()?)
}

/// Fills a device buffer with a byte pattern.
///
/// Prefer [`fill_zero`] for typed initialization. Nonzero byte patterns are best
/// suited to byte buffers and debugging sentinels.
pub fn fill_bytes<T>(buffer: &impl AsRef<DeviceBuffer<T>>, value: u8) -> Result<()> {
    Ok(buffer.as_ref().memset(value)?)
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
        assert_eq!(input.size(), 4);
        assert_eq!(input.shape(), [4]);
        assert_eq!(input.element_size(), std::mem::size_of::<u32>());
        assert_eq!(input.byte_len(), 4 * std::mem::size_of::<u32>());
        assert_eq!(input.sum().expect("array reduce"), 10);
        assert_eq!(reduce_sum(&input).expect("free reduce over array"), 10);

        let inclusive = GpuArray::<u32>::empty(input.len()).expect("array inclusive output");
        input
            .inclusive_scan_into(&inclusive)
            .expect("array inclusive scan into");
        assert_eq!(
            inclusive.download().expect("inclusive into download"),
            [1, 3, 6, 10]
        );
        assert_eq!(
            input
                .cumsum()
                .expect("array cumsum")
                .download()
                .expect("cumsum download"),
            [1, 3, 6, 10]
        );
        let cumsum_into = GpuArray::<u32>::empty(input.len()).expect("cumsum into output");
        input.cumsum_into(&cumsum_into).expect("cumsum into");
        assert_eq!(
            cumsum_into.download().expect("cumsum into download"),
            [1, 3, 6, 10]
        );
        assert_eq!(
            cumsum(&input)
                .expect("free cumsum")
                .download()
                .expect("free cumsum download"),
            [1, 3, 6, 10]
        );

        let free_exclusive = GpuArray::<u32>::empty(input.len()).expect("free exclusive output");
        exclusive_scan(&input, &free_exclusive, 0).expect("free exclusive over array");
        assert_eq!(
            free_exclusive
                .download()
                .expect("free exclusive array download"),
            [0, 1, 3, 6]
        );

        let exclusive_into = GpuArray::<u32>::empty(input.len()).expect("exclusive into output");
        input
            .exclusive_scan_into(&exclusive_into, 0)
            .expect("exclusive scan into");
        assert_eq!(
            exclusive_into.download().expect("exclusive into download"),
            [0, 1, 3, 6]
        );
        assert_eq!(
            input
                .exclusive_scan(0)
                .expect("array exclusive scan")
                .to_vec()
                .expect("scan download"),
            [0, 1, 3, 6]
        );
        assert_eq!(
            exclusive_cumsum(&input, 0)
                .expect("free exclusive cumsum")
                .download()
                .expect("exclusive cumsum download"),
            [0, 1, 3, 6]
        );
        assert_eq!(
            input
                .add_scalar(5)
                .expect("array add scalar")
                .to_vec()
                .expect("map download"),
            [6, 7, 8, 9]
        );

        let mapped_into = GpuArray::<u32>::empty(input.len()).expect("mapped into output");
        input
            .add_scalar_into(&mapped_into, 7)
            .expect("array add scalar into");
        assert_eq!(
            mapped_into.download().expect("mapped into download"),
            [8, 9, 10, 11]
        );

        let free_mapped = GpuArray::<u32>::empty(input.len()).expect("free mapped output");
        map_add_u32(&input, &free_mapped, 3).expect("free map add over arrays");
        assert_eq!(
            free_mapped.download().expect("free mapped download"),
            [4, 5, 6, 7]
        );
        assert_eq!(
            add_scalar_u32(&input, 2)
                .expect("free add scalar")
                .download()
                .expect("free add scalar download"),
            [3, 4, 5, 6]
        );

        let zeros = GpuArray::<u32>::zeros(4).expect("zero array");
        assert_eq!(zeros.to_vec().expect("zero download"), [0, 0, 0, 0]);
        fill_zero(&free_mapped).expect("free fill zero over array");
        assert_eq!(
            free_mapped.download().expect("free fill zero download"),
            [0, 0, 0, 0]
        );

        let flags = GpuArray::from_slice(&[1u8, 0, 1, 0]).expect("array flags upload");
        let (selected, selected_count) = input.select_flagged(&flags).expect("array select");
        assert_eq!(selected_count, 2);
        assert_eq!(
            &selected.download().expect("selected download")[..selected_count],
            [1, 3]
        );
        let (where_selected, where_count) = input.where_flags(&flags).expect("array where flags");
        assert_eq!(where_count, 2);
        assert_eq!(
            &where_selected.download().expect("where download")[..where_count],
            [1, 3]
        );
        let (free_where, free_where_count) =
            where_flags_u32(&input, &flags).expect("free where flags");
        assert_eq!(free_where_count, 2);
        assert_eq!(
            &free_where.download().expect("free where download")[..free_where_count],
            [1, 3]
        );
    }

    #[test]
    fn gpu_array_copy_helpers_smoke_if_device_available() {
        if crate::Device::first().is_err() {
            return;
        }

        let empty = GpuArray::<u32>::empty(3).expect("empty array");
        assert_eq!(empty.len(), 3);
        assert_eq!(empty.shape(), [3]);

        let zeroed = GpuArray::<u32>::zeroed(2).expect("zeroed array");
        assert_eq!(zeroed.download().expect("zeroed download"), [0, 0]);

        let free_zeroed = zeros::<u32>(2).expect("free zeros constructor");
        assert_eq!(free_zeroed.to_list().expect("free zeros download"), [0, 0]);

        let values = GpuArray::from_values([1u32, 2, 3]).expect("values upload");
        assert_eq!(values.download().expect("values download"), [1, 2, 3]);

        let free_values = array([1u32, 2, 3]).expect("free array constructor");
        assert_eq!(
            free_values.to_list().expect("free array download"),
            [1, 2, 3]
        );

        let from_vec = GpuArray::from_vec(vec![4u32, 5, 6]).expect("vec upload");
        let mut host = [0u32; 3];
        from_vec.copy_to_slice(&mut host).expect("copy to slice");
        assert_eq!(host, [4, 5, 6]);

        from_vec.upload(&[7, 8, 9]).expect("upload alias");
        assert_eq!(from_vec.download().expect("upload download"), [7, 8, 9]);
        from_vec.assign(&[10, 11, 12]).expect("assign alias");
        assert_eq!(from_vec.to_list().expect("assign download"), [10, 11, 12]);

        let cloned = from_vec.cloned().expect("device clone");
        assert_eq!(cloned.download().expect("clone download"), [10, 11, 12]);
        let copied = from_vec.copy().expect("device copy alias");
        assert_eq!(copied.download().expect("copy download"), [10, 11, 12]);
        let free_copied = copy(&from_vec).expect("free copy");
        assert_eq!(
            free_copied.download().expect("free copy download"),
            [10, 11, 12]
        );

        let destination = GpuArray::<u32>::zeroed(3).expect("copy destination");
        destination.copy_from(&values).expect("device copy from");
        assert_eq!(
            destination.download().expect("copy destination download"),
            [1, 2, 3]
        );

        let repeated = GpuArray::repeat(42u32, 3).expect("repeat upload");
        assert_eq!(repeated.download().expect("repeat download"), [42, 42, 42]);

        let filled = GpuArray::full(3, 5u32).expect("full method");
        assert_eq!(filled.to_list().expect("full method download"), [5, 5, 5]);
        let free_filled = full(3, 6u32).expect("free full constructor");
        assert_eq!(
            free_filled.to_list().expect("free full download"),
            [6, 6, 6]
        );

        let scalar = GpuArray::from_value(11u32).expect("scalar upload");
        assert_eq!(scalar.read().expect("scalar read"), 11);
        assert_eq!(scalar.item().expect("scalar item"), 11);
        scalar.write(13).expect("scalar write");
        assert_eq!(scalar.read().expect("scalar reread"), 13);

        let err = values.read().expect_err("multi-element read should fail");
        assert!(err.to_string().contains("GpuArray::read expects 1"));

        let matrix = GpuArray2D::from_slice(2, 3, &[1u32, 2, 3, 4, 5, 6]).expect("matrix upload");
        assert_eq!(matrix.shape(), [2, 3]);
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.height(), 2);
        assert_eq!(matrix.width(), 3);
        assert_eq!(matrix.len(), 6);
        assert_eq!(
            matrix.to_rows().expect("matrix rows"),
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        );

        let matrix_copy = matrix.copy().expect("matrix copy");
        assert_eq!(
            matrix_copy.download().expect("matrix copy download"),
            [1, 2, 3, 4, 5, 6]
        );

        let free_matrix = array_2d(2, 2, [9u32, 8, 7, 6]).expect("free matrix");
        assert_eq!(
            free_matrix.to_rows().expect("free matrix rows"),
            vec![vec![9, 8], vec![7, 6]]
        );

        let zero_matrix = zeros_2d::<u32>(2, 2).expect("zero matrix");
        assert_eq!(
            zero_matrix.download().expect("zero matrix download"),
            [0, 0, 0, 0]
        );

        let full_matrix = full_2d(1, 3, 5u32).expect("full matrix");
        assert_eq!(
            full_matrix.to_rows().expect("full matrix rows"),
            vec![vec![5, 5, 5]]
        );

        let matrix_err = match GpuArray2D::from_slice(2, 3, &[1u32, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert!(matrix_err.to_string().contains("GpuArray2D::from_slice"));
    }

    #[test]
    fn gpu_sort_smoke_if_available() {
        if !RocThrust::is_available() {
            return;
        }

        let mut data = DeviceBuffer::from_slice(&[4u32, 1, 3, 1]).expect("sort upload");
        assert!(contains_eq_u32(&data, 3).expect("buffer contains"));
        assert!(!contains_eq_u32(&data, 99).expect("buffer not contains"));
        sort(&mut data).expect("in-place sort");
        assert_eq!(data.copy_to_vec().expect("sort download"), [1, 1, 3, 4]);

        let n = unique_u32(&mut data).expect("unique");
        assert_eq!(n, 3);
        assert_eq!(
            &data.copy_to_vec().expect("unique download")[..n],
            [1, 3, 4]
        );

        assert_eq!(count_eq_u32(&data, 1).expect("count"), 1);

        let mut free_sort_unique =
            DeviceBuffer::from_slice(&[3u32, 1, 3, 2, 1]).expect("free sort unique upload");
        assert_eq!(
            sort_unique_u32(&mut free_sort_unique).expect("free sort unique"),
            3
        );
        assert_eq!(
            &free_sort_unique
                .copy_to_vec()
                .expect("free sort unique download")[..3],
            [1, 2, 3]
        );

        let mut keys = DeviceBuffer::from_slice(&[3u32, 1, 2]).expect("keys upload");
        let mut values = DeviceBuffer::from_slice(&[30u32, 10, 20]).expect("values upload");
        sort_by_key_u32(&mut keys, &mut values).expect("sort by key");
        assert_eq!(keys.copy_to_vec().expect("keys download"), [1, 2, 3]);
        assert_eq!(values.copy_to_vec().expect("values download"), [10, 20, 30]);

        let mut array = GpuArray::from_slice(&[2u32, 4, 1, 1]).expect("array sort upload");
        assert!(array.contains(4).expect("array contains"));
        assert!(!array.contains(99).expect("array not contains"));
        assert_eq!(
            array
                .sorted_keys()
                .expect("array sorted keys")
                .download()
                .expect("sorted keys download"),
            [1, 1, 2, 4]
        );
        array.sort().expect("array sort");
        assert_eq!(array.to_vec().expect("array sort download"), [1, 1, 2, 4]);
        assert_eq!(array.count_eq(1).expect("array count"), 2);
        assert_eq!(array.unique_consecutive().expect("array unique"), 3);
        assert_eq!(
            &array.download().expect("array unique download")[..3],
            [1, 2, 4]
        );

        let mut sort_unique =
            GpuArray::from_slice(&[3u32, 1, 3, 2, 1]).expect("sort unique upload");
        assert_eq!(sort_unique.sort_unique().expect("sort unique"), 3);
        assert_eq!(
            &sort_unique.download().expect("sort unique download")[..3],
            [1, 2, 3]
        );
        let sorted_unique_input =
            GpuArray::from_slice(&[2u32, 4, 1, 1]).expect("sorted unique input");
        let (sorted_unique, sorted_unique_count) = sorted_unique_input
            .sorted_unique()
            .expect("array sorted unique copy");
        assert_eq!(sorted_unique_count, 3);
        assert_eq!(
            &sorted_unique
                .download()
                .expect("array sorted unique download")[..sorted_unique_count],
            [1, 2, 4]
        );
        let (free_sorted_unique, free_sorted_unique_count) =
            sorted_unique_u32(&sorted_unique_input).expect("free sorted unique copy");
        assert_eq!(free_sorted_unique_count, 3);
        assert_eq!(
            &free_sorted_unique
                .download()
                .expect("free sorted unique download")[..free_sorted_unique_count],
            [1, 2, 4]
        );

        let mut array_keys = GpuArray::from_slice(&[3u32, 1, 2]).expect("array keys upload");
        let mut array_values = GpuArray::from_slice(&[30u32, 10, 20]).expect("array values upload");
        array_keys
            .sort_by_key(&mut array_values)
            .expect("array sort by key");
        assert_eq!(
            array_keys.download().expect("array keys download"),
            [1, 2, 3]
        );
        assert_eq!(
            array_values.download().expect("array values download"),
            [10, 20, 30]
        );
    }
}
