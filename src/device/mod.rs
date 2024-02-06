use anyhow::anyhow;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    codegen::linearizer::{Linearizer, LinearizerOptions},
    dtype::Dtype,
    ops::{getenv, LazyOp},
    renderer::cstyle::{uops_to_cstyle, LanguageOpts, Renderer},
    shape::symbolic::NodeOp,
};

lazy_static::lazy_static! {
    pub static ref DEVICES: Vec<fn() -> anyhow::Result<Arc<dyn Device>>> = vec![
        #[cfg(target_arch = "x86_64")]
        cuda::CudaDevice::new,
        opencl::CLDevice::new,
    ];
}

lazy_static::lazy_static! {
    pub static ref DEVICE: Arc<dyn Device> = {
        match getenv::<String>("DEVICE", "".into()).to_uppercase().as_str() {
            "CUDA" => cuda::CudaDevice::new().unwrap(),
            "OPENCL" => opencl::CLDevice::new().unwrap(),
            _ =>  {
                let mut d = vec![];
                for func in DEVICES.iter() {
                    if let Ok(device) = func() {
                        d.push(device);
                        break;
                    }
                }
                d[0].to_owned()
            }
        }
    };
}

// #[derive(Default, Debug)]
// pub struct PendingCopy(Vec<Vec<u8>>);
//
// unsafe impl Send for PendingCopy {}
// unsafe impl Sync for PendingCopy {}

pub mod cuda;
pub mod opencl;

pub mod prelude {
    pub use super::opencl::{CLBuffer, CLDevice, CLProgram};
    pub use super::{ALLOCTOR, DEVICE};
}

pub trait Device: Send + Sync + core::fmt::Debug {
    fn name(&self) -> String;
    fn _alloc(&self, size: usize, dtype: Dtype) -> anyhow::Result<Arc<dyn Buffer>>;
    fn alloc(&self, size: usize, dtype: Dtype) -> Arc<dyn Buffer> {
        ALLOCTOR.0.alloc(size, dtype)
    }
    fn buf_from_mem_ptr(
        &self,
        size: usize,
        dtype: Dtype,
        mem: *mut std::ffi::c_void,
    ) -> Arc<dyn Buffer>;
    fn build(&self, name: &str, program: &str) -> Arc<dyn Program>;
    fn copyout(&self, src: &dyn Buffer, dst: *mut u8);
    fn copyin(&self, src: Vec<u8>, dst: &dyn Buffer);
    fn synchronize(&self);
    fn linearizer_opts(&self) -> LinearizerOptions {
        LinearizerOptions::default()
    }
    fn renderer(&self) -> Arc<dyn Renderer>;
    fn get_lin(&self, ast: LazyOp) -> Linearizer {
        let mut ret = Linearizer::new(ast, Some(self.linearizer_opts()));
        if getenv("OP", 0) == 1 {
            ret.kernel.hand_coded_optim();
        }
        ret
    }
    fn render(&self, mut lin: Linearizer) -> (String, String) {
        lin.linearize();
        let prg = uops_to_cstyle(self.renderer(), &lin.name, &lin.uops);
        (lin.name, prg)
    }
    fn free(&self, ptr: *mut std::ffi::c_void);
}

pub trait Program: core::fmt::Debug {
    fn run(
        &self,
        bufs: &[Arc<dyn Buffer>],
        global_size: &[usize],
        local_size: Option<&[usize]>,
        args: &[isize],
        extra: &[String],
    );
}

// Also need to implement Drop
pub trait Buffer: core::fmt::Debug {
    fn ptr(&self) -> *mut core::ffi::c_void;
    fn dtype(&self) -> Dtype;
    fn bytesize(&self) -> usize;
    fn to_cpu(&self) -> Vec<u8>;
    fn from_cpu(&mut self, data: Vec<u8>);
}

#[derive(Default)]
pub struct Allocator {
    pub cached: Arc<HashMap<usize, Vec<*mut std::ffi::c_void>>>, // <Bytesize, Vec<_>>
}

unsafe impl Send for Allocator {}
unsafe impl Sync for Allocator {}

impl Allocator {
    pub fn alloc(&self, size: usize, dtype: Dtype) -> Arc<dyn Buffer> {
        unsafe {
            let mut cc = self.cached.clone();
            let cached = Arc::get_mut_unchecked(&mut cc);
            if let Some(mems) = cached.get_mut(&(size * dtype.size)) && !mems.is_empty() {
                DEVICE.buf_from_mem_ptr(size, dtype, mems.pop().unwrap())
            } else {
                if let Ok(b) = DEVICE._alloc(size, dtype.clone()) {
                    b
                } else {
                    self.free_cached();
                    DEVICE._alloc(size, dtype).unwrap()
                }
            }
        }
    }

    pub fn free(&self, buf: &dyn Buffer) {
        unsafe {
            if std::env::var("LRU").unwrap_or("1".into()) == "1" {
                let mut cc = self.cached.clone();
                let cached = Arc::get_mut_unchecked(&mut cc);
                cached.entry(buf.bytesize()).or_default().push(buf.ptr());
            } else {
                DEVICE.free(buf.ptr());
            }
        }
    }

    pub fn free_cached(&self) {
        for cached in self.cached.values() {
            for opaque in cached.iter() {
                DEVICE.free(*opaque)
            }
        }
        unsafe {
            let mut cc = self.cached.clone();
            let cached = Arc::get_mut_unchecked(&mut cc);
            cached.clear();
        }
    }
}

#[derive(Default)]
pub struct AllocatorWrapper(pub Allocator);

lazy_static::lazy_static! {
    pub static ref ALLOCTOR: AllocatorWrapper = AllocatorWrapper::default();
}
