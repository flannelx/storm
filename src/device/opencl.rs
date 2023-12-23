use std::sync::Arc;

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

use crate::prelude::*;
use crate::renderer::cstyle::CstyleLanguage;
use crate::shape::symbolic::CStyle;

use super::{Buffer, Device, Program};

#[derive(Debug, Clone)]
pub struct CLDevice {
    pub device_id: usize,
    pub device: opencl3::device::Device,
    pub context: Arc<opencl3::context::Context>,
    pub queue: Arc<opencl3::command_queue::CommandQueue>,
}

unsafe impl Send for CLDevice {}
unsafe impl Sync for CLDevice {}

impl CLDevice {
    pub fn new() -> Self {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .unwrap()
            .first()
            .expect("no device found in platform");
        let device = opencl3::device::Device::new(device_id);
        let context = Context::from_device(&device).unwrap();
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("CommandQueue::create_default failed");
        Self {
            device_id: device_id as usize,
            device,
            context: Arc::new(context),
            queue: Arc::new(queue),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CLBuffer {
    ptr: opencl3::memory::cl_mem,
    size: usize,
    dtype: Dtype,
}

#[allow(unused)]
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
            //opencl3::command_queue::enqueue_nd_range_kernel(command_queue, kernel, work_dim, global_work_offset, global_work_dims, local_work_dims, num_events_in_wait_list, event_wait_list)
            let mut ek = ExecuteKernel::new(&self.kernel);
            for b in bufs {
                ek.set_arg(&b.ptr());
            }
            ek.set_global_work_size(global_size[0]);
            ek.enqueue_nd_range(&self.device.queue)
                .expect("enqueue failed");
        }
    }
}

impl Buffer for CLBuffer {
    fn ptr(&self) -> *mut core::ffi::c_void {
        self.ptr
    }

    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }

    fn to_cpu(&self) -> Vec<u8> {
        let mut dst = vec![0u8; self.size()];
        let ptr = dst.as_mut_ptr() as *mut u8;
        DEVICE.copyout(self, ptr);
        dst
    }

    fn from_cpu(&mut self, data: Vec<u8>) {
        DEVICE.copyin(data, self);
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl Device for CLDevice {
    fn alloc(&self, size: usize, dtype: Dtype) -> Arc<dyn Buffer> {
        unsafe {
            Arc::new(CLBuffer {
                ptr: opencl3::memory::create_buffer(
                    self.context.get(),
                    CL_MEM_READ_WRITE,
                    size * dtype.size,
                    core::ptr::null_mut(),
                )
                .unwrap(),
                size: size * dtype.size,
                dtype,
            })
        }
    }

    fn build(&self, name: &str, program: &str) -> Arc<dyn Program> {
        let program =
            opencl3::program::Program::create_and_build_from_source(&self.context, program, "")
                .expect("Program::create_and_build_from_source failed");
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
                CL_NON_BLOCKING,
                0,
                src.size(),
                dst as opencl3::memory::cl_mem,
                0,
                core::ptr::null(),
            )
            .expect("Copyout failed");
        }
    }

    fn copyin(&self, mut src: Vec<u8>, dst: &mut dyn Buffer) {
        unsafe {
            opencl3::command_queue::enqueue_write_buffer(
                self.queue.get(),
                dst.ptr(),
                CL_NON_BLOCKING,
                0,
                dst.size(),
                src.as_mut_ptr() as opencl3::memory::cl_mem,
                0,
                core::ptr::null(),
            )
            .expect("copyin failed");
            PENDING_COPY.lock().unwrap().0.push(src);
        }
    }

    fn synchronize(&self) {
        opencl3::command_queue::finish(self.queue.get()).expect("Queue finish failed");
        PENDING_COPY.lock().unwrap().0.clear();
    }

    fn renderer(&self) -> CstyleLanguage {
        CstyleLanguage {
            kernel_prefix: "__kernel ".into(),
            buffer_prefix: "__global ".into(),
            smem_align: "__attribute__ ((aligned (16))) ".into(),
            smem_prefix: "__local ".into(),
            arg_int_prefix: "const int".into(),
            half_prekernel: Some("#pragma OPENCL EXTENSION cl_khr_fp16 : enable".into()),
            barrier: "barrier(CLK_LOCAL_MEM_FENCE);".into(),
            float4: Some("(float4)".into()),
            gid: (0..3).map(|i| format!("get_group_id({i})")).collect(),
            lid: (0..3).map(|i| format!("get_local_id({i})")).collect(),
            uses_vload: true,
            ..Default::default()
        }
    }
}
