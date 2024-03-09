#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::sync::Arc;

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

use crate::codegen::linearizer::LinearizerOptions;
use crate::prelude::*;
use crate::renderer::cstyle::{LanguageOpts, Renderer};
use crate::shape::symbolic::CStyle;

use super::{Buffer, Device, Program};

#[derive(Debug, Clone)]
pub struct CLDevice {
    pub device_id: usize,
    pub device: opencl3::device::Device,
    pub context: Arc<opencl3::context::Context>,
    pub queue: Arc<opencl3::command_queue::CommandQueue>,
    pub renderer: Arc<dyn Renderer>,
}

unsafe impl Send for CLDevice {}
unsafe impl Sync for CLDevice {}

impl CLDevice {
    pub fn new() -> anyhow::Result<Arc<dyn Device>> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
            .first()
            .expect("no device found in platform");
        let device = opencl3::device::Device::new(device_id);
        let context = Context::from_device(&device).unwrap();
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("CommandQueue::create_default failed");
        Ok(Arc::new(Self {
            device_id: device_id as usize,
            device,
            context: Arc::new(context),
            queue: Arc::new(queue),
            renderer: Arc::new(CLRenderer::default()),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct CLBuffer {
    ptr: opencl3::memory::cl_mem,
    bytesize: usize,
    dtype: Dtype,
}

impl Buffer for CLBuffer {
    fn device(&self) -> String {
        "OPENCL".into()
    }

    fn ptr(&self) -> *mut core::ffi::c_void {
        self.ptr
    }

    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }

    fn bytesize(&self) -> usize {
        self.bytesize
    }

    fn to_cpu(&self) -> Vec<u8> {
        let mut dst = vec![0u8; self.bytesize()];
        let ptr = dst.as_mut_ptr() as *mut u8;
        DEVICE.copyout(self, ptr);
        DEVICE.synchronize();
        dst
    }

    fn from_cpu(&mut self, data: Vec<u8>) {
        DEVICE.copyin(data, self);
    }
}

impl Drop for CLBuffer {
    fn drop(&mut self) {
        ALLOCTOR.0.free(&*self)
    }
}

#[derive(Debug)]
pub struct CLProgram {
    program: opencl3::program::Program,
    kernel: opencl3::kernel::Kernel,
    device: CLDevice,
}

impl Program for CLProgram {
    fn run(
        &self,
        bufs: &[Arc<dyn Buffer>],
        global_size: &[usize],
        local_size: Option<&[usize]>,
        args: &[isize],
        extra: &[String],
    ) {
        unsafe {
            let mut global_size = global_size.to_vec();
            if let Some(l) = local_size {
                global_size = v![(g*l), for (g, l) in global_size.iter().zip(l)];
            }
            for (i, b) in bufs.into_iter().enumerate() {
                self.kernel.set_arg(i as _, &b.ptr());
            }
            if opencl3::command_queue::enqueue_nd_range_kernel(
                self.device.queue.get(),
                self.kernel.get(),
                global_size.len() as _,
                std::ptr::null(),
                global_size.as_ptr() as _,
                if local_size.is_some() {
                    local_size.as_ref().unwrap().as_ptr() as _
                } else {
                    std::ptr::null()
                },
                0,
                std::ptr::null(),
            )
            .is_err_and(|e| e == -4)
            {
                ALLOCTOR.0.free_cached();
                opencl3::command_queue::enqueue_nd_range_kernel(
                    self.device.queue.get(),
                    self.kernel.get(),
                    global_size.len() as _,
                    std::ptr::null(),
                    global_size.as_ptr() as _,
                    if local_size.is_some() {
                        local_size.as_ref().unwrap().as_ptr() as _
                    } else {
                        std::ptr::null()
                    },
                    0,
                    std::ptr::null(),
                );
            }
        }
    }
}

impl Device for CLDevice {
    fn name(&self) -> String {
        "OPENCL".into()
    }

    fn _alloc(&self, size: usize, dtype: Dtype) -> anyhow::Result<Arc<dyn Buffer>> {
        unsafe {
            let ptr = opencl3::memory::create_buffer(
                self.context.get(),
                CL_MEM_READ_WRITE,
                size * dtype.size,
                core::ptr::null_mut(),
            );
            if ptr.is_err() {
                return Err(anyhow::anyhow!("{:?}", ptr));
            }
            Ok(Arc::new(CLBuffer {
                ptr: ptr.unwrap(),
                bytesize: size * dtype.size,
                dtype,
            }))
        }
    }

    fn buf_from_mem_ptr(
        &self,
        size: usize,
        dtype: Dtype,
        mem: *mut std::ffi::c_void,
    ) -> Arc<dyn Buffer> {
        unsafe {
            Arc::new(CLBuffer {
                ptr: mem,
                bytesize: size * dtype.size,
                dtype,
            })
        }
    }

    fn build(&self, name: &str, program: &str) -> Arc<dyn Program> {
        let program =
            opencl3::program::Program::create_and_build_from_source(&self.context, program, "");
        // Need to `Display` print, panic will print message in Debug which will escape characters
        // like new lines `\n`
        if program.is_err() {
            println!("{}", program.err().unwrap());
            panic!("Program::create_and_build_from_source failed");
        };
        let program = program.unwrap();
        let kernel = Kernel::create(&program, name).expect("Kernel::create failed");
        Arc::new(CLProgram {
            device: self.clone(),
            program,
            kernel,
        })
    }

    fn copyout(&self, src: &dyn Buffer, dst: *mut u8) {
        unsafe {
            opencl3::command_queue::enqueue_read_buffer(
                self.queue.get(),
                src.ptr(),
                CL_BLOCKING,
                0,
                src.bytesize(),
                dst as opencl3::memory::cl_mem,
                0,
                core::ptr::null(),
            )
            .expect("Copyout failed");
        }
    }

    fn copyin(&self, mut src: Vec<u8>, dst: &dyn Buffer) {
        unsafe {
            opencl3::command_queue::enqueue_write_buffer(
                self.queue.get(),
                dst.ptr(),
                CL_BLOCKING,
                0,
                dst.bytesize(),
                src.as_mut_ptr() as opencl3::memory::cl_mem,
                0,
                core::ptr::null(),
            )
            .expect("copyin failed");
        }
    }

    fn synchronize(&self) {
        opencl3::command_queue::finish(self.queue.get()).expect("Queue finish failed");
    }

    fn renderer(&self) -> Arc<dyn Renderer> {
        self.renderer.clone()
    }

    fn free(&self, ptr: *mut std::ffi::c_void) {
        unsafe { opencl3::memory::release_mem_object(ptr) };
    }
}

#[derive(Debug)]
pub struct CLRenderer {
    opts: Arc<LanguageOpts>,
}

impl Default for CLRenderer {
    fn default() -> Self {
        Self {
            opts: Arc::new(LanguageOpts {
                kernel_prefix: "__kernel ".into(),
                buffer_prefix: "__global ".into(),
                smem_align: "__attribute__ ((aligned (16))) ".into(),
                smem_prefix: "__local ".into(),
                arg_int_prefix: "const int".into(),
                half_prekernel: Some("#pragma OPENCL EXTENSION cl_khr_fp16 : enable".into()),
                barrier: "barrier(CLK_LOCAL_MEM_FENCE);".into(),
                float4: Some("(float4)".into()),
                code_for_workitem: HashMap::from([
                    (
                        "g".into(),
                        (0..3).map(|i| format!("get_group_id({i})")).collect(),
                    ),
                    (
                        "l".into(),
                        (0..3).map(|i| format!("get_local_id({i})")).collect(),
                    ),
                    (
                        "i".into(),
                        (0..3).map(|i| format!("get_global_id({i})")).collect(),
                    ),
                ]),
                uses_vload: true,
                ..Default::default()
            }),
        }
    }
}

impl crate::ops::Op for CLRenderer {
    fn mulacc(&self, a: &str, b: &str, c: &str) -> String {
        format!("mad({a}, {b}, {c})")
    }

    fn _where(&self, a: &str, b: &str, c: &str) -> String {
        format!("((bool)({a})?{b}:{c})")
    }
}

impl Renderer for CLRenderer {
    fn lang_opts(&self) -> Arc<LanguageOpts> {
        self.opts.clone()
    }
}
