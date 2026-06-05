//! Test helpers for GPU-backed SDK examples and downstream crates.

use crate::{Device, DeviceLimits, Error, Result};

/// Context passed to GPU tests created by [`gpu_test!`](crate::gpu_test).
///
/// `GpuTestContext::new()` opens the first visible HIP device and records the
/// detected architecture and launch limits through the normal [`Device`] API.
#[derive(Debug, Clone)]
pub struct GpuTestContext {
    device: Device,
}

impl GpuTestContext {
    /// Opens the first visible HIP device for a GPU-backed test.
    pub fn new() -> Result<Self> {
        Ok(Self {
            device: Device::first()?,
        })
    }

    /// Returns the opened device handle.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the HIP device ordinal used by this test.
    pub fn ordinal(&self) -> i32 {
        self.device.ordinal()
    }

    /// Returns the detected GPU architecture, such as `gfx1100`.
    pub fn arch(&self) -> &str {
        self.device.arch()
    }

    /// Returns cached launch and memory limits for the test device.
    pub fn limits(&self) -> DeviceLimits {
        self.device.limits()
    }
}

/// Returns true when a GPU test should be reported as skipped, not failed.
pub fn is_missing_gpu_error(error: &Error) -> bool {
    matches!(error, Error::NoDevice)
}

/// Defines a GPU-backed Rust test with a [`testing::GpuTestContext`](crate::testing::GpuTestContext).
///
/// The generated test opens the first visible HIP device. If no HIP device is
/// visible, the test prints a skip message and returns `Ok(())`; other setup
/// failures are returned as normal test errors.
///
/// ```rust,ignore
/// rocm_oxide::gpu_test!(device_buffer_round_trip, |gpu| {
///     eprintln!("running on {}", gpu.arch());
///     let buffer = rocm_oxide::DeviceBuffer::from_slice(&[1u32, 2, 3])?;
///     assert_eq!(buffer.copy_to_vec()?, [1, 2, 3]);
///     Ok(())
/// });
/// ```
#[macro_export]
macro_rules! gpu_test {
    ($name:ident, |$context:ident| $body:block) => {
        #[test]
        fn $name() -> $crate::Result<()> {
            let $context = match $crate::testing::GpuTestContext::new() {
                Ok(context) => context,
                Err(error) if $crate::testing::is_missing_gpu_error(&error) => {
                    eprintln!("skipping GPU test `{}`: {error}", stringify!($name));
                    return Ok(());
                }
                Err(error) => return Err(error),
            };

            (|| -> $crate::Result<()> { $body })()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::DeviceBuffer;

    #[test]
    fn missing_gpu_predicate_matches_no_device() {
        assert!(super::is_missing_gpu_error(&crate::Error::NoDevice));
        assert!(!super::is_missing_gpu_error(&crate::Error::InvalidLaunch(
            "bad launch".to_string()
        )));
    }

    crate::gpu_test!(gpu_test_macro_round_trips_device_buffer, |gpu| {
        assert!(gpu.ordinal() >= 0);
        assert!(gpu.arch().starts_with("gfx"));

        let buffer = DeviceBuffer::from_slice(&[1u32, 2, 3])?;
        assert_eq!(buffer.copy_to_vec()?, [1, 2, 3]);
        Ok(())
    });
}
