use crate::{Device, Result, Stream, hip};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Waker};
use std::thread;

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
    use super::{DeviceOperation, Value};
    use crate::{ExecutionContext, Result};

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
}
