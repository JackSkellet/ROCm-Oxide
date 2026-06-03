use crate::{Device, DeviceBuffer, DevicePod, Error, Result, RocmTileTransferPlan, Stream, hip};
use std::ffi::c_void;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Waker};
use std::thread;

const MAX_STREAM_POOL_SIZE: usize = 64;

#[must_use = "keeps launch resources alive until the execution context is synchronized"]
pub struct KernelLaunchCompletion {
    keep_alive: Vec<Box<dyn Send>>,
}

impl KernelLaunchCompletion {
    pub fn new() -> Self {
        Self {
            keep_alive: Vec::new(),
        }
    }

    pub fn keep_alive<T>(&mut self, value: T)
    where
        T: Send + 'static,
    {
        self.keep_alive.push(Box::new(value));
    }

    pub fn retained_count(&self) -> usize {
        self.keep_alive.len()
    }
}

impl Default for KernelLaunchCompletion {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct ExecutionContext {
    device_ordinal: i32,
    stream: Arc<Stream>,
}

impl ExecutionContext {
    pub fn new(device: &Device) -> Result<Self> {
        hip::set_device(device.ordinal())?;
        Ok(Self {
            device_ordinal: device.ordinal(),
            stream: Arc::new(Stream::new()?),
        })
    }

    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn bind_thread(&self) -> Result<()> {
        hip::set_device(self.device_ordinal)?;
        Ok(())
    }
}

pub struct StreamPool {
    contexts: Vec<ExecutionContext>,
    next: AtomicUsize,
}

impl StreamPool {
    pub fn new(device: &Device, streams: usize) -> Result<Self> {
        let streams = streams.max(1);
        if streams > MAX_STREAM_POOL_SIZE {
            return Err(crate::Error::Async(format!(
                "stream pool size {streams} exceeds ROCm-Oxide safety cap {MAX_STREAM_POOL_SIZE}"
            )));
        }
        let mut contexts = Vec::with_capacity(streams);
        for _ in 0..streams {
            contexts.push(ExecutionContext::new(device)?);
        }
        Ok(Self {
            contexts,
            next: AtomicUsize::new(0),
        })
    }

    pub fn next_context(&self) -> ExecutionContext {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.contexts.len();
        self.contexts[index].clone()
    }

    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }
}

pub trait DeviceOperation: Send + Sized + 'static {
    type Output: Send + 'static;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output>;

    fn sync_on(self, context: &ExecutionContext) -> Result<Self::Output> {
        context.bind_thread()?;
        let output = self.execute(context)?;
        context.synchronize()?;
        Ok(output)
    }

    fn sync(self, device: &Device) -> Result<Self::Output> {
        let context = ExecutionContext::new(device)?;
        self.sync_on(&context)
    }

    fn capture_graph_on(self, context: &ExecutionContext) -> Result<CapturedGraph<Self::Output>> {
        context.bind_thread()?;
        context
            .stream()
            .begin_capture(hip::StreamCaptureMode::ThreadLocal)?;
        let output = match self.execute(context) {
            Ok(output) => output,
            Err(err) => {
                let _ = context.stream().end_capture();
                return Err(err);
            }
        };
        let graph = context.stream().end_capture()?;
        let exec = graph.instantiate()?;
        Ok(CapturedGraph {
            exec,
            capture_output: output,
        })
    }

    fn capture_graph(self, device: &Device) -> Result<CapturedGraph<Self::Output>> {
        let context = ExecutionContext::new(device)?;
        self.capture_graph_on(&context)
    }

    fn async_on(self, context: ExecutionContext) -> DeviceFuture<Self::Output> {
        let (sender, receiver) = mpsc::channel();
        let waker_slot = Arc::new(Mutex::new(None::<Waker>));
        let worker_waker_slot = Arc::clone(&waker_slot);

        thread::spawn(move || {
            let result = (|| {
                context.bind_thread()?;
                let output = self.execute(&context)?;
                context.synchronize()?;
                Ok(output)
            })();
            let _ = sender.send(result);
            if let Some(waker) = worker_waker_slot
                .lock()
                .ok()
                .and_then(|mut slot| slot.take())
            {
                waker.wake();
            }
        });

        DeviceFuture {
            receiver: Some(receiver),
            waker_slot,
        }
    }

    fn async_in(self, pool: &StreamPool) -> DeviceFuture<Self::Output> {
        self.async_on(pool.next_context())
    }

    fn map<F, U>(self, map: F) -> Map<Self, F>
    where
        F: FnOnce(Self::Output) -> U + Send + 'static,
        U: Send + 'static,
    {
        Map {
            operation: self,
            map,
        }
    }

    fn and_then<F, Next>(self, next: F) -> AndThen<Self, F>
    where
        F: FnOnce(Self::Output) -> Next + Send + 'static,
        Next: DeviceOperation,
    {
        AndThen {
            operation: self,
            next,
        }
    }

    fn zip<Other>(self, other: Other) -> Zip<Self, Other>
    where
        Other: DeviceOperation,
    {
        Zip { left: self, other }
    }
}

pub struct CapturedGraph<T> {
    exec: hip::GraphExec,
    capture_output: T,
}

impl<T> CapturedGraph<T> {
    pub fn capture_output(&self) -> &T {
        &self.capture_output
    }

    pub fn launch_on(&self, context: &ExecutionContext) -> Result<()> {
        context.bind_thread()?;
        Ok(self.exec.launch(context.stream())?)
    }

    pub fn launch_and_sync_on(&self, context: &ExecutionContext) -> Result<()> {
        self.launch_on(context)?;
        context.synchronize()
    }
}

impl<F, T> DeviceOperation for F
where
    F: FnOnce(&ExecutionContext) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    type Output = T;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        self(context)
    }
}

pub struct Value<T> {
    value: T,
}

impl<T> Value<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> DeviceOperation for Value<T>
where
    T: Send + 'static,
{
    type Output = T;

    fn execute(self, _context: &ExecutionContext) -> Result<Self::Output> {
        Ok(self.value)
    }
}

#[must_use = "keeps stream-enqueued copy resources alive until the execution context is synchronized"]
pub struct DeviceCopyCompletion<T> {
    destination: Arc<DeviceBuffer<T>>,
    completion: KernelLaunchCompletion,
}

impl<T> DeviceCopyCompletion<T> {
    pub fn destination(&self) -> &Arc<DeviceBuffer<T>> {
        &self.destination
    }

    pub fn retained_count(&self) -> usize {
        self.completion.retained_count()
    }
}

#[must_use = "keeps stream-enqueued memset resources alive until the execution context is synchronized"]
pub struct DeviceMemsetCompletion<T> {
    buffer: Arc<DeviceBuffer<T>>,
    completion: KernelLaunchCompletion,
}

impl<T> DeviceMemsetCompletion<T> {
    pub fn buffer(&self) -> &Arc<DeviceBuffer<T>> {
        &self.buffer
    }

    pub fn retained_count(&self) -> usize {
        self.completion.retained_count()
    }
}

#[must_use = "device copy jobs do not enqueue until executed as a DeviceOperation"]
pub struct HostToDeviceCopy<T> {
    destination: Arc<DeviceBuffer<T>>,
    input: Vec<T>,
}

impl<T: DevicePod> HostToDeviceCopy<T> {
    pub fn new(destination: Arc<DeviceBuffer<T>>, input: Vec<T>) -> Result<Self> {
        validate_operation_len("host-to-device source", input.len(), destination.len())?;
        Ok(Self { destination, input })
    }

    pub fn len(&self) -> usize {
        self.input.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input.is_empty()
    }
}

impl<T: DevicePod> DeviceOperation for HostToDeviceCopy<T> {
    type Output = DeviceCopyCompletion<T>;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        unsafe {
            self.destination
                .copy_from_host_async(context.stream(), self.input.as_slice())?;
        }
        let mut completion = KernelLaunchCompletion::new();
        completion.keep_alive(Arc::clone(&self.destination));
        completion.keep_alive(self.input);
        Ok(DeviceCopyCompletion {
            destination: self.destination,
            completion,
        })
    }
}

#[must_use = "device copy jobs do not enqueue until executed as a DeviceOperation"]
pub struct DeviceToDeviceCopy<T> {
    destination: Arc<DeviceBuffer<T>>,
    source: Arc<DeviceBuffer<T>>,
}

impl<T: DevicePod> DeviceToDeviceCopy<T> {
    pub fn new(destination: Arc<DeviceBuffer<T>>, source: Arc<DeviceBuffer<T>>) -> Self {
        Self {
            destination,
            source,
        }
    }

    pub fn try_new(
        destination: Arc<DeviceBuffer<T>>,
        source: Arc<DeviceBuffer<T>>,
    ) -> Result<Self> {
        validate_operation_len("device-to-device source", source.len(), destination.len())?;
        validate_distinct_buffers(
            "device-to-device copy",
            destination.as_ref(),
            source.as_ref(),
        )?;
        Ok(Self::new(destination, source))
    }

    pub fn len(&self) -> usize {
        self.destination.len()
    }

    pub fn is_empty(&self) -> bool {
        self.destination.is_empty()
    }
}

impl<T: DevicePod> DeviceOperation for DeviceToDeviceCopy<T> {
    type Output = DeviceCopyCompletion<T>;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        validate_operation_len(
            "device-to-device source",
            self.source.len(),
            self.destination.len(),
        )?;
        validate_distinct_buffers(
            "device-to-device copy",
            self.destination.as_ref(),
            self.source.as_ref(),
        )?;
        unsafe {
            self.destination
                .copy_from_device_async(context.stream(), self.source.as_ref())?;
        }
        let mut completion = KernelLaunchCompletion::new();
        completion.keep_alive(Arc::clone(&self.destination));
        completion.keep_alive(Arc::clone(&self.source));
        Ok(DeviceCopyCompletion {
            destination: self.destination,
            completion,
        })
    }
}

#[must_use = "device memset jobs do not enqueue until executed as a DeviceOperation"]
pub struct DeviceMemset<T> {
    buffer: Arc<DeviceBuffer<T>>,
    value: u8,
}

impl<T: DevicePod> DeviceMemset<T> {
    pub fn new(buffer: Arc<DeviceBuffer<T>>, value: u8) -> Self {
        Self { buffer, value }
    }

    pub fn zero(buffer: Arc<DeviceBuffer<T>>) -> Self {
        Self::new(buffer, 0)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn value(&self) -> u8 {
        self.value
    }
}

impl<T: DevicePod> DeviceOperation for DeviceMemset<T> {
    type Output = DeviceMemsetCompletion<T>;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        unsafe {
            self.buffer.memset_async(context.stream(), self.value)?;
        }
        let mut completion = KernelLaunchCompletion::new();
        completion.keep_alive(Arc::clone(&self.buffer));
        Ok(DeviceMemsetCompletion {
            buffer: self.buffer,
            completion,
        })
    }
}

#[must_use = "tile transfer jobs do not enqueue until executed as a DeviceOperation"]
pub struct DeviceTileTransfer<T> {
    plan: RocmTileTransferPlan,
    destination: Arc<DeviceBuffer<T>>,
    source: Arc<DeviceBuffer<T>>,
}

impl<T: DevicePod> DeviceTileTransfer<T> {
    pub fn new(
        plan: RocmTileTransferPlan,
        destination: Arc<DeviceBuffer<T>>,
        source: Arc<DeviceBuffer<T>>,
    ) -> Result<Self> {
        validate_tile_transfer_plan::<T>(plan, destination.len())?;
        validate_operation_len(
            "tile device-to-device source",
            source.len(),
            destination.len(),
        )?;
        validate_distinct_buffers(
            "tile device-to-device copy",
            destination.as_ref(),
            source.as_ref(),
        )?;
        Ok(Self {
            plan,
            destination,
            source,
        })
    }

    pub fn plan(&self) -> RocmTileTransferPlan {
        self.plan
    }

    pub fn len(&self) -> usize {
        self.destination.len()
    }

    pub fn is_empty(&self) -> bool {
        self.destination.is_empty()
    }
}

impl<T: DevicePod> DeviceOperation for DeviceTileTransfer<T> {
    type Output = DeviceCopyCompletion<T>;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        validate_tile_transfer_plan::<T>(self.plan, self.destination.len())?;
        unsafe {
            self.destination
                .copy_from_device_async(context.stream(), self.source.as_ref())?;
        }
        let mut completion = KernelLaunchCompletion::new();
        completion.keep_alive(Arc::clone(&self.destination));
        completion.keep_alive(Arc::clone(&self.source));
        Ok(DeviceCopyCompletion {
            destination: self.destination,
            completion,
        })
    }
}

pub fn copy_host_to_device<T: DevicePod>(
    destination: Arc<DeviceBuffer<T>>,
    input: Vec<T>,
) -> Result<HostToDeviceCopy<T>> {
    HostToDeviceCopy::new(destination, input)
}

pub fn copy_device_to_device<T: DevicePod>(
    destination: Arc<DeviceBuffer<T>>,
    source: Arc<DeviceBuffer<T>>,
) -> Result<DeviceToDeviceCopy<T>> {
    DeviceToDeviceCopy::try_new(destination, source)
}

pub fn memset_device<T: DevicePod>(buffer: Arc<DeviceBuffer<T>>, value: u8) -> DeviceMemset<T> {
    DeviceMemset::new(buffer, value)
}

pub fn zero_device<T: DevicePod>(buffer: Arc<DeviceBuffer<T>>) -> DeviceMemset<T> {
    DeviceMemset::zero(buffer)
}

pub fn tile_transfer_device_to_device<T: DevicePod>(
    plan: RocmTileTransferPlan,
    destination: Arc<DeviceBuffer<T>>,
    source: Arc<DeviceBuffer<T>>,
) -> Result<DeviceTileTransfer<T>> {
    DeviceTileTransfer::new(plan, destination, source)
}

impl hip::Graph {
    /// Adds a typed host-to-device memcpy node.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid until every executable graph launch
    /// that can reach this node has completed. `src` must not be mutated while
    /// a graph launch may read it.
    pub unsafe fn add_typed_memcpy_host_to_device_node<T: DevicePod>(
        &self,
        dependencies: &[hip::GraphNode],
        dst: &DeviceBuffer<T>,
        src: &[T],
    ) -> Result<hip::GraphNode> {
        validate_operation_len("graph host-to-device source", src.len(), dst.len())?;
        let bytes = checked_operation_bytes::<T>(dst.len(), "graph host-to-device copy")?;
        if bytes == 0 {
            return Ok(self.add_empty_node(dependencies)?);
        }
        unsafe {
            Ok(self.add_memcpy_node_1d(
                dependencies,
                dst.as_mut_ptr().cast::<c_void>(),
                src.as_ptr().cast::<c_void>(),
                bytes,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?)
        }
    }

    /// Adds a typed device-to-host memcpy node.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must remain valid until every executable graph launch
    /// that can reach this node has completed. `dst` must not be read, written,
    /// dropped, or aliased while a graph launch may write it.
    pub unsafe fn add_typed_memcpy_device_to_host_node<T: DevicePod>(
        &self,
        dependencies: &[hip::GraphNode],
        dst: &mut [T],
        src: &DeviceBuffer<T>,
    ) -> Result<hip::GraphNode> {
        validate_operation_len("graph device-to-host destination", dst.len(), src.len())?;
        let bytes = checked_operation_bytes::<T>(src.len(), "graph device-to-host copy")?;
        if bytes == 0 {
            return Ok(self.add_empty_node(dependencies)?);
        }
        unsafe {
            Ok(self.add_memcpy_node_1d(
                dependencies,
                dst.as_mut_ptr().cast::<c_void>(),
                src.as_ptr().cast::<c_void>(),
                bytes,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?)
        }
    }
}

impl hip::GraphNode {
    /// Retargets a typed host-to-device memcpy node.
    ///
    /// # Safety
    ///
    /// `self` must be a memcpy node. `dst` and `src` must remain valid until
    /// every executable graph launch that can reach this node has completed.
    /// `src` must not be mutated while a graph launch may read it.
    pub unsafe fn set_typed_memcpy_host_to_device<T: DevicePod>(
        self,
        dst: &DeviceBuffer<T>,
        src: &[T],
    ) -> Result<()> {
        validate_operation_len("graph host-to-device source", src.len(), dst.len())?;
        let bytes = checked_operation_bytes::<T>(dst.len(), "graph host-to-device copy")?;
        if bytes == 0 {
            return Err(Error::InvalidLaunch(
                "cannot retarget a HIP graph memcpy node to a zero-byte host-to-device copy"
                    .to_string(),
            ));
        }
        unsafe {
            Ok(self.set_memcpy_1d(
                dst.as_mut_ptr().cast::<c_void>(),
                src.as_ptr().cast::<c_void>(),
                bytes,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?)
        }
    }

    /// Retargets a typed device-to-host memcpy node.
    ///
    /// # Safety
    ///
    /// `self` must be a memcpy node. `src` and `dst` must remain valid until
    /// every executable graph launch that can reach this node has completed.
    /// `dst` must not be read, written, dropped, or aliased while a graph launch
    /// may write it.
    pub unsafe fn set_typed_memcpy_device_to_host<T: DevicePod>(
        self,
        dst: &mut [T],
        src: &DeviceBuffer<T>,
    ) -> Result<()> {
        validate_operation_len("graph device-to-host destination", dst.len(), src.len())?;
        let bytes = checked_operation_bytes::<T>(src.len(), "graph device-to-host copy")?;
        if bytes == 0 {
            return Err(Error::InvalidLaunch(
                "cannot retarget a HIP graph memcpy node to a zero-byte device-to-host copy"
                    .to_string(),
            ));
        }
        unsafe {
            Ok(self.set_memcpy_1d(
                dst.as_mut_ptr().cast::<c_void>(),
                src.as_ptr().cast::<c_void>(),
                bytes,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?)
        }
    }
}

fn checked_operation_bytes<T>(len: usize, label: &str) -> Result<usize> {
    len.checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        Error::InvalidLaunch(format!(
            "{label} byte size overflows usize for {len} elements"
        ))
    })
}

fn validate_operation_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidLaunch(format!(
            "{label} length mismatch: got {actual}, expected {expected}"
        )))
    }
}

fn validate_distinct_buffers<T>(
    label: &str,
    dst: &DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
) -> Result<()> {
    if dst.is_empty() || src.is_empty() || !std::ptr::eq(dst.as_ptr(), src.as_ptr()) {
        Ok(())
    } else {
        Err(Error::InvalidLaunch(format!(
            "{label} source and destination buffers alias"
        )))
    }
}

fn validate_tile_transfer_plan<T>(plan: RocmTileTransferPlan, len: usize) -> Result<()> {
    let element_size = mem::size_of::<T>();
    if element_size == 0 {
        return Err(Error::InvalidLaunch(
            "tile transfer element type must have nonzero size".to_string(),
        ));
    }
    if plan.tile_bytes % element_size != 0 {
        return Err(Error::InvalidLaunch(format!(
            "tile transfer byte size {} is not divisible by element size {element_size}",
            plan.tile_bytes
        )));
    }
    let expected_len = plan.tile_bytes / element_size;
    validate_operation_len("tile transfer buffer", len, expected_len)
}

pub struct Map<Op, F> {
    operation: Op,
    map: F,
}

impl<Op, F, U> DeviceOperation for Map<Op, F>
where
    Op: DeviceOperation,
    F: FnOnce(Op::Output) -> U + Send + 'static,
    U: Send + 'static,
{
    type Output = U;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        Ok((self.map)(self.operation.execute(context)?))
    }
}

pub struct AndThen<Op, F> {
    operation: Op,
    next: F,
}

impl<Op, F, Next> DeviceOperation for AndThen<Op, F>
where
    Op: DeviceOperation,
    F: FnOnce(Op::Output) -> Next + Send + 'static,
    Next: DeviceOperation,
{
    type Output = Next::Output;

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        let output = self.operation.execute(context)?;
        (self.next)(output).execute(context)
    }
}

pub struct Zip<Left, Right> {
    left: Left,
    other: Right,
}

impl<Left, Right> DeviceOperation for Zip<Left, Right>
where
    Left: DeviceOperation,
    Right: DeviceOperation,
{
    type Output = (Left::Output, Right::Output);

    fn execute(self, context: &ExecutionContext) -> Result<Self::Output> {
        Ok((self.left.execute(context)?, self.other.execute(context)?))
    }
}

pub struct DeviceFuture<T> {
    receiver: Option<mpsc::Receiver<Result<T>>>,
    waker_slot: Arc<Mutex<Option<Waker>>>,
}

impl<T> DeviceFuture<T> {
    pub fn wait(mut self) -> Result<T> {
        let receiver = self
            .receiver
            .take()
            .expect("DeviceFuture receiver missing before wait");
        receiver
            .recv()
            .unwrap_or_else(|_| Err(crate::runtime::Error::Async("device worker stopped".into())))
    }
}

impl<T> Future for DeviceFuture<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(receiver) = self.receiver.as_ref() else {
            return Poll::Ready(Err(crate::runtime::Error::Async(
                "device future was already consumed".into(),
            )));
        };

        match receiver.try_recv() {
            Ok(result) => {
                self.receiver.take();
                Poll::Ready(result)
            }
            Err(mpsc::TryRecvError::Empty) => {
                if let Ok(mut slot) = self.waker_slot.lock() {
                    *slot = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.receiver.take();
                Poll::Ready(Err(crate::runtime::Error::Async(
                    "device worker stopped".into(),
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceMemset, DeviceOperation, DeviceToDeviceCopy, Value, copy_device_to_device,
        copy_host_to_device, memset_device, tile_transfer_device_to_device,
    };
    use crate::{DeviceBuffer, ExecutionContext, PinnedHostBuffer, Result, RocmTileTransferPlan};
    use std::mem::size_of;
    use std::sync::Arc;

    fn fake_context() -> ExecutionContext {
        ExecutionContext {
            device_ordinal: 0,
            stream: std::sync::Arc::new(crate::Stream::null()),
        }
    }

    #[test]
    fn value_map_and_then_compose_lazily() -> Result<()> {
        let context = fake_context();
        let result = Value::new(4)
            .map(|value| value + 2)
            .and_then(|value| Value::new(value * 3))
            .execute(&context)?;
        assert_eq!(result, 18);
        Ok(())
    }

    #[test]
    fn zip_returns_both_outputs() -> Result<()> {
        let context = fake_context();
        let result = Value::new(2).zip(Value::new(5)).execute(&context)?;
        assert_eq!(result, (2, 5));
        Ok(())
    }

    #[test]
    fn stream_pool_size_is_bounded() -> Result<()> {
        let device = crate::Device::first()?;
        let pool = super::StreamPool::new(&device, 0)?;
        assert_eq!(pool.len(), 1);

        let err = match super::StreamPool::new(&device, super::MAX_STREAM_POOL_SIZE + 1) {
            Ok(_) => panic!("oversized stream pool should fail before creating streams"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("safety cap"));
        Ok(())
    }

    #[test]
    fn owned_copy_memset_and_tile_operations_round_trip() -> Result<()> {
        let device = crate::Device::first()?;
        let context = device.execution_context()?;

        let upload = Arc::new(DeviceBuffer::<u32>::new(4)?);
        let completion =
            copy_host_to_device(Arc::clone(&upload), vec![1, 2, 3, 4])?.sync_on(&context)?;
        assert_eq!(completion.retained_count(), 2);
        assert_eq!(upload.copy_to_vec()?, [1, 2, 3, 4]);

        let completion = memset_device(Arc::clone(&upload), 0).sync_on(&context)?;
        assert_eq!(completion.retained_count(), 1);
        assert_eq!(upload.copy_to_vec()?, [0, 0, 0, 0]);

        let _completion =
            copy_host_to_device(Arc::clone(&upload), vec![9, 10, 11, 12])?.sync_on(&context)?;
        let copy = Arc::new(DeviceBuffer::<u32>::new(4)?);
        let completion =
            copy_device_to_device(Arc::clone(&copy), Arc::clone(&upload))?.sync_on(&context)?;
        assert_eq!(completion.retained_count(), 2);
        assert_eq!(copy.copy_to_vec()?, [9, 10, 11, 12]);

        let plan =
            RocmTileTransferPlan::for_2d_tile(device.properties()?, size_of::<u32>(), 2, 2, 1)?;
        let tile = Arc::new(DeviceBuffer::<u32>::new(4)?);
        let completion =
            tile_transfer_device_to_device(plan, Arc::clone(&tile), Arc::clone(&copy))?
                .sync_on(&context)?;
        assert_eq!(completion.retained_count(), 2);
        assert_eq!(tile.copy_to_vec()?, [9, 10, 11, 12]);
        Ok(())
    }

    #[test]
    fn typed_graph_host_copy_helpers_round_trip() -> Result<()> {
        let buffer = DeviceBuffer::<u32>::new(4)?;
        let mut input = PinnedHostBuffer::<u32>::new_zeroed(4)?;
        input.as_mut_slice().copy_from_slice(&[13, 21, 34, 55]);
        let mut output = PinnedHostBuffer::<u32>::new_zeroed(4)?;
        let graph = crate::hip::Graph::new()?;

        let upload =
            unsafe { graph.add_typed_memcpy_host_to_device_node(&[], &buffer, input.as_slice())? };
        unsafe {
            graph.add_typed_memcpy_device_to_host_node(
                &[upload],
                output.as_mut_slice(),
                &buffer,
            )?;
        }

        let exec = graph.instantiate()?;
        let stream = crate::Stream::new()?;
        exec.launch(&stream)?;
        stream.synchronize()?;
        assert_eq!(output.as_slice(), [13, 21, 34, 55]);
        Ok(())
    }

    #[test]
    fn owned_copy_operation_rejects_length_mismatch_before_ffi() -> Result<()> {
        let dst = Arc::new(DeviceBuffer::<u32>::new(0)?);
        let Err(err) = copy_host_to_device(Arc::clone(&dst), vec![1]) else {
            panic!("length mismatch should fail before enqueue");
        };
        assert!(err.to_string().contains("length mismatch"));
        Ok(())
    }

    #[test]
    fn device_operation_error_after_async_enqueue_cleans_up_buffers() -> Result<()> {
        let device = crate::Device::first()?;
        let context = ExecutionContext::new(&device)?;
        let result = (|context: &ExecutionContext| -> Result<()> {
            let buffer = unsafe { crate::DeviceBuffer::<u32>::new_async(context.stream(), 4)? };
            unsafe {
                buffer.set_zero_async(context.stream())?;
            }
            Err(crate::Error::Async("injected failure".into()))
        })
        .sync_on(&context);

        assert!(matches!(result, Err(crate::Error::Async(_))));
        let buffer = crate::DeviceBuffer::from_slice(&[1u32, 2, 3, 4])?;
        assert_eq!(buffer.copy_to_vec()?, [1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn device_copy_operation_round_trips() -> Result<()> {
        let device = crate::Device::first()?;
        let source = Arc::new(crate::DeviceBuffer::from_slice(&[8u32, 6, 7, 5])?);
        let destination = Arc::new(crate::DeviceBuffer::<u32>::new(4)?);
        let completion =
            DeviceToDeviceCopy::new(Arc::clone(&destination), Arc::clone(&source)).sync(&device)?;

        assert_eq!(completion.retained_count(), 2);
        assert!(Arc::ptr_eq(completion.destination(), &destination));
        assert_eq!(destination.copy_to_vec()?, [8, 6, 7, 5]);
        Ok(())
    }

    #[test]
    fn device_copy_operation_rejects_length_mismatch() -> Result<()> {
        let device = crate::Device::first()?;
        let source = Arc::new(crate::DeviceBuffer::from_slice(&[1u32, 2])?);
        let destination = Arc::new(crate::DeviceBuffer::<u32>::new(4)?);
        let err = match DeviceToDeviceCopy::new(destination, source).sync(&device) {
            Ok(_) => panic!("short lazy copy should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("length mismatch"));
        Ok(())
    }

    #[test]
    fn device_memset_operation_round_trips() -> Result<()> {
        let device = crate::Device::first()?;
        let buffer = Arc::new(crate::DeviceBuffer::<u8>::new(8)?);
        let completion = DeviceMemset::new(Arc::clone(&buffer), 0x3c).sync(&device)?;

        assert_eq!(completion.retained_count(), 1);
        assert!(Arc::ptr_eq(completion.buffer(), &buffer));
        assert_eq!(buffer.copy_to_vec()?, [0x3c; 8]);
        Ok(())
    }
}
