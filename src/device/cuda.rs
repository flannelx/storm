#![cfg(not(macos))]

use cudarc::driver::result::malloc_sync;
use cudarc::driver::sys::{
    cuCtxCreate_v2, cuCtxSetCurrent, cuMemAllocManaged, cudaError_enum, CUcontext,
};
use cudarc::driver::sys::{cuMemFree_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, CUdevice, CUdeviceptr};
use cudarc::driver::{CudaFunction, DevicePtrMut};
use cudarc::nvrtc::{compile_ptx, compile_ptx_with_opts};

use super::{Buffer, Device, Program};
use crate::prelude::*;
use crate::renderer::cstyle::{LanguageOpts, Renderer};
use crate::shape::symbolic::CStyle;
use std::ffi::CString;
use std::ptr::null_mut;
use std::sync::Arc;

#[derive(Debug)]
pub struct CudaRenderer {
    opts: Arc<LanguageOpts>,
}

impl Default for CudaRenderer {
    fn default() -> Self {
        Self {
            opts: Arc::new(LanguageOpts {
                kernel_prefix: "#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\nextern \"C\" __global__ ".into(),
                smem_prefix: "__shared__ ".into(),
                arg_int_prefix: "const int".into(),
                half_prekernel: Some("#include <cuda_fp16.h>".into()),
                barrier: "__syncthreads();".into(),
                float4: Some("make_float4".into()),
                gid: (0..3)
                    .map(|i| {
                        format!(
                            "blockDim.{}*blockIdx.{}+threadIdx.{}",
                            (120u8 + i) as char,
                            (120u8 + i) as char,
                            (120u8 + i) as char
                        )
                    })
                    .collect(),
                lid: (0..3)
                    .map(|i| format!("threadIdx.{}", (128u8 + i) as char))
                    .collect(),
                uses_vload: true,
                ..Default::default()
            }),
        }
    }
}

impl crate::ops::Op for CudaRenderer {}

impl Renderer for CudaRenderer {
    fn lang_opts(&self) -> Arc<LanguageOpts> {
        self.opts.clone()
    }
}

#[derive(Debug)]
pub struct CudaBuffer {
    ptr: CUdeviceptr,
    bytesize: usize,
    dtype: Dtype,
}

impl Buffer for CudaBuffer {
    fn ptr(&self) -> *mut core::ffi::c_void {
        self.ptr as _
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
        dst
    }

    fn from_cpu(&mut self, data: Vec<u8>) {
        DEVICE.copyin(data, self);
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cuMemFree_v2(self.ptr);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    device: Arc<cudarc::driver::CudaDevice>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new() -> anyhow::Result<Arc<dyn Device>> {
        let device = cudarc::driver::CudaDevice::new(0)?;
        Ok(Arc::new(Self { device }))
    }
}

impl Device for CudaDevice {
    fn _alloc(&self, size: usize, dtype: Dtype) -> anyhow::Result<Arc<dyn Buffer>> {
        unsafe {
            Ok(Arc::new(CudaBuffer {
                ptr: malloc_sync(size * dtype.size)?,
                bytesize: size * dtype.size,
                dtype,
            }))
        }
    }

    fn buf_from_mem_ptr(&self, size: usize, dtype: Dtype, mem: *mut std::ffi::c_void) -> Arc<dyn Buffer> {
        unsafe {
            Arc::new(CudaBuffer {
                ptr: mem as _,
                bytesize: size * dtype.size,
                dtype,
            })
        }
    }

    fn build(&self, name: &str, program: &str) -> Arc<dyn Program> {
        unsafe {
            let ptx = cudarc::nvrtc::compile_ptx(program).unwrap();
            let mut module: cudarc::driver::sys::CUmodule = std::ptr::null_mut();
            let cstring = CString::new(ptx.to_src()).unwrap();
            let r =
                cudarc::driver::sys::cuModuleLoadData((&mut module) as _, cstring.as_ptr() as _);
            assert!(r == cudaError_enum::CUDA_SUCCESS, "{:?}", r);
            let mut func: cudarc::driver::sys::CUfunction = std::ptr::null_mut();
            let cstring = CString::new(name).unwrap();
            let r = cudarc::driver::sys::cuModuleGetFunction(
                (&mut func) as _,
                module,
                cstring.as_ptr() as _,
            );
            assert!(r == cudaError_enum::CUDA_SUCCESS, "{:?}", r);
            Arc::new(CudaProgram {
                module,
                func,
                device: self.clone(),
            })
        }
    }

    fn copyout(&self, src: &dyn Buffer, dst: *mut u8) {
        unsafe {
            let r = cuMemcpyDtoH_v2(dst as _, src.ptr() as _, src.bytesize());
            assert!(r == cudaError_enum::CUDA_SUCCESS, "{:?}", r);
        }
    }

    fn copyin(&self, src: Vec<u8>, dst: &dyn Buffer) {
        unsafe {
            let r = cuMemcpyHtoD_v2(dst.ptr() as _, src.as_ptr() as _, src.len());
            assert!(r == cudaError_enum::CUDA_SUCCESS, "{:?}", r);
        }
    }

    fn synchronize(&self) {
        self.device.synchronize().expect("Device fail to sync");
    }

    fn renderer(&self) -> Arc<dyn Renderer> {
        Arc::new(CudaRenderer::default())
    }

    fn free(&self, ptr: *mut std::ffi::c_void) {
        unsafe {
            cuMemFree_v2(ptr as _);
        }
    }
}

#[derive(Debug)]
pub struct CudaProgram {
    func: cudarc::driver::sys::CUfunction,
    module: cudarc::driver::sys::CUmodule,
    device: CudaDevice,
}

impl Program for CudaProgram {
    fn run(
        &self,
        bufs: &[Arc<dyn Buffer>],
        global_size: &[usize],
        local_size: Option<&[usize]>,
        args: &[isize],
        extra: &[String],
    ) {
        let mut args = v![b.ptr() as CUdeviceptr, for b in bufs.iter()];
        let local_size = local_size.unwrap();
        unsafe {
            let r = cudarc::driver::sys::cuLaunchKernel(
                self.func,
                global_size[0] as _,
                global_size[1] as _,
                global_size[2] as _,
                local_size[0] as _,
                local_size[1] as _,
                local_size[2] as _,
                0,
                null_mut(),
                args.iter().collect::<Vec<&u64>>().as_mut_ptr() as _,
                null_mut(),
            );
            assert!(r == cudaError_enum::CUDA_SUCCESS, "{:?}", r);
        }
    }
}
