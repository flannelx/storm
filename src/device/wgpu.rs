use crate::ops::{Dtype, DEVICE};
use std::sync::Arc;

use super::{Buffer, Device};

#[derive(Debug)]
pub struct WGPUBuffer {
    buffer: *mut wgpu::Buffer,
    dtype: Dtype,
}

impl Buffer for WGPUBuffer {
    fn device(&self) -> String {
        "WGPU".into()
    }

    fn ptr(&self) -> *mut core::ffi::c_void {
        self.buffer as _
    }

    fn dtype(&self) -> crate::ops::Dtype {
        self.dtype.clone()
    }

    fn bytesize(&self) -> usize {
        unsafe { (*self.buffer).size() as _ }
    }

    fn to_cpu(&self) -> Vec<u8> {
        unsafe {
            let mut buffer = &*(self.buffer);
            buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
                result.unwrap();
            });
            DEVICE.synchronize();
            let r = buffer.slice(..).get_mapped_range();
            let ret = r.get(..).unwrap().to_vec();
            drop(r);
            buffer.unmap();
            ret
        }
    }
}

#[derive(Debug)]
pub struct WGPUDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn new() -> anyhow::Result<WGPUDevice> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();
    Ok(WGPUDevice { device, queue })
}

impl WGPUDevice {
    pub fn new() -> anyhow::Result<Arc<dyn Device>> {
        Ok(Arc::new(pollster::block_on(new())?))
    }
}

impl Device for WGPUDevice {
    fn name(&self) -> String {
        "WGPU".into()
    }

    fn _alloc(
        &self,
        size: usize,
        dtype: crate::ops::Dtype,
    ) -> anyhow::Result<std::sync::Arc<dyn super::Buffer>> {
        Ok(Arc::new(WGPUBuffer {
            buffer: Box::into_raw(Box::new(self.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: None,
                    size: (size * dtype.size) as _,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            ))),
            dtype,
        }))
    }

    fn buf_from_mem_ptr(
        &self,
        size: usize,
        dtype: crate::ops::Dtype,
        mem: *mut std::ffi::c_void,
    ) -> std::sync::Arc<dyn super::Buffer> {
        unsafe {
            Arc::new(WGPUBuffer {
                buffer: mem as *mut _,
                dtype,
            })
        }
    }

    fn build(&self, name: &str, program: &str) -> std::sync::Arc<dyn super::Program> {
        todo!()
    }

    fn copyout(&self, src: &dyn super::Buffer, dst: *mut u8) {
        todo!()
    }

    fn copyin(&self, src: Vec<u8>, dst: &dyn super::Buffer) {
        unsafe { self.queue.write_buffer(&(*(dst.ptr() as *mut _)), 0, &src) }
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn renderer(&self) -> std::sync::Arc<dyn crate::renderer::cstyle::Renderer> {
        todo!()
    }

    fn free(&self, ptr: *mut std::ffi::c_void) {
        todo!()
    }
}
