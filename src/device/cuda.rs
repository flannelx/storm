#![cfg(not(target_os = "macos"))]
use cudarc::driver::result::malloc_sync;
use cudarc::driver::sys::{
    cuCtxCreate_v2, cuCtxSetCurrent, cuDeviceComputeCapability, cuGetErrorString,
    cuMemAllocManaged, cudaError_enum, CUcontext,
};
use cudarc::driver::sys::{cuMemFree_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, CUdevice, CUdeviceptr};
use cudarc::driver::{CudaFunction, DevicePtrMut};
use cudarc::nvrtc::{compile_ptx, compile_ptx_with_opts, CompileOptions};

use super::{Buffer, Device, Program};
use crate::codegen::linearizer::{LinearizerOptions, UOp};
use crate::prelude::*;
use crate::renderer::cstyle::{LanguageOpts, Renderer};
use crate::shape::symbolic::CStyle;
use std::collections::HashMap;
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
                kernel_prefix: "extern \"C\" __global__ ".into(),
                smem_prefix: "__shared__ ".into(),
                arg_int_prefix: "const int".into(),
                half_prekernel: Some("#include <cuda_fp16.h>".into()),
                barrier: "__syncthreads();".into(),
                float4: Some("make_float4".into()),
                code_for_workitem: HashMap::from([
                    (
                        "g".into(),
                        (0..3)
                            .map(|i| format!("blockIdx.{}", (120u8 + i) as char,))
                            .collect(),
                    ),
                    (
                        "l".into(),
                        (0..3)
                            .map(|i| format!("threadIdx.{}", (120u8 + i) as char,))
                            .collect(),
                    ),
                    (
                        "i".into(),
                        (0..3)
                            .map(|i| {
                                format!(
                                    "(blockIdx.{}*blockDim.{}+threadIdx.{})",
                                    (120u8 + i) as char,
                                    (120u8 + i) as char,
                                    (120u8 + i) as char,
                                )
                            })
                            .collect(),
                    ),
                ]),
                uses_vload: true,
                global_max: vec![65535, 65535, 2147483647],
                local_max: vec![64, 1024, 1024],
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

    fn render_kernel(
        &self,
        function_name: &str,
        kernel: &[String],
        bufs: &[(String, dtype::Dtype)],
        local_size: &[usize],
        uops: &[UOp],
        prekernel: &[String],
    ) -> String {
        // Cuda stuff
        let mut prekernel = vec![
            "#define INFINITY (__int_as_float(0x7f800000))",
            "#define NAN (__int_as_float(0x7fffffff))",
        ];
        if any(&v![u.dtype.as_ref().is_some_and(|d| *d == half), for u in uops]) {
            prekernel.extend(["#include <cuda_fp16.h>", "struct half4 { half x, y, z, w; };",
      "__device__ half4 make_half4(half x, half y, half z, half w) { half4 ret; ret.x = x; ret.y = y; ret.z = z; ret.w = w; return ret; }"]);
        }
        if any(&v![u.dtype.as_ref().is_some_and(|d| *d == bfloat16), for u in uops]) {
            prekernel.push("#include <cuda_fp16.h>")
        }

        // Default impl below
        let tmp = if bufs.iter().any(|(_, dt)| dt.shape.is_some()) {
            "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
        } else {
            ""
        };
        let mut buftypes = vec![];
        for (i, (name, dtype)) in bufs.iter().enumerate() {
            let s = if dtype.type_name.starts_with("image") {
                format!(
                    "{} image2d_t",
                    if 1 > 0 { "read_only" } else { "write_only" }
                )
            } else {
                if dtype == &dtype::_arg_int32 {
                    self.lang_opts().arg_int_prefix.to_string()
                } else {
                    (if i > 0 {
                        "const ".to_string()
                    } else {
                        "".to_string()
                    }) + &self.lang_opts().buffer_prefix
                        + dtype.c_name
                        + "*"
                        + &self.lang_opts().buffer_suffix
                }
            };
            buftypes.push((name, s));
        }

        let prod_local_size = local_size.iter().product::<usize>();
        let mut prg = {
            format!(
                "{}void {}{function_name}(",
                self.lang_opts().kernel_prefix,
                if self.lang_opts().launch_bounds {
                    format!("__launch_bounds__ ({prod_local_size}, 1)")
                } else {
                    "".to_string()
                }
            )
        };

        let mut args = buftypes
            .iter()
            .map(|(name, t)| format!("{t} {name}"))
            .collect::<Vec<String>>();
        args.extend(self.lang_opts().extra_args.clone());
        prg += &args.join(", ");

        prg += &format!("{}{}{}{}", ") {\n", tmp, kernel.join("\n"), "\n}");

        if self.lang_opts().half_prekernel.is_some()
            && bufs.iter().any(|(_, dtype)| *dtype == dtype::float16)
        {
            prg = self.lang_opts().half_prekernel.as_ref().unwrap().clone() + "\n" + &prg;
        }

        if prekernel.len() > 0 {
            format!("{}\n{}", prekernel.join("\n"), prg)
        } else {
            prg
        }
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
    arch: &'static str,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new() -> anyhow::Result<Arc<dyn Device>> {
        let device = cudarc::driver::CudaDevice::new(0)?;
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            cuDeviceComputeCapability(&mut major, &mut minor, *device.cu_device());
        }
        let arch = format!("sm_{major}{minor}");
        Ok(Arc::new(Self {
            device,
            arch: arch.leak(),
        }))
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

    fn buf_from_mem_ptr(
        &self,
        size: usize,
        dtype: Dtype,
        mem: *mut std::ffi::c_void,
    ) -> Arc<dyn Buffer> {
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
            //let ptx = cudarc::nvrtc::compile_ptx(program).unwrap();
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                program,
                CompileOptions {
                    include_paths: vec![
                        "/usr/local/cuda/include".into(),
                        "/usr/include".into(),
                        "/opt/cuda/include/".into(),
                    ],
                    arch: Some(self.arch),
                    ..Default::default()
                },
            )
            .unwrap();
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

    fn linearizer_opts(&self) -> crate::codegen::linearizer::LinearizerOptions {
        let mut ret = LinearizerOptions::default();
        ret.global_max = Some(vec![65535, 65535, 2147483647]);
        ret.local_max = Some(vec![64, 1024, 1024]);
        ret
    }

    fn renderer(&self) -> Arc<dyn Renderer> {
        Arc::new(CudaRenderer::default())
    }

    fn free(&self, ptr: *mut std::ffi::c_void) {
        unsafe {
            cuMemFree_v2(ptr as _);
        }
    }

    fn name(&self) -> String {
        "CUDA".into()
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
            //println!("{:?}\nbuffers: {:?}\nglobal:{:?}\nlocal:{:?}", r, args, global_size, local_size);
            assert!(
                r == cudaError_enum::CUDA_SUCCESS,
                "{:?}\nbuffers: {:?}\nglobal:{:?}\nlocal:{:?}",
                r,
                args,
                global_size,
                local_size
            );
        }
    }
}
