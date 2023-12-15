use std::sync::Arc;

use crate::dtype::Dtype;

lazy_static::lazy_static! {
    pub static ref DEVICE: Arc<dyn Device> = Arc::new(opencl::CLDevice::new());
}

pub mod opencl;

pub mod prelude {
    pub use super::opencl::{CLDevice, CLBuffer, CLProgram};
    pub use super::DEVICE;
}

pub trait Device: Send + Sync + core::fmt::Debug {
    fn alloc(&self, size: usize, dtype: Dtype) -> Arc<dyn Buffer>;
    fn build(&self, name: &str, program: &str) -> Arc<dyn Program>;
    fn copyout(&self, src: &dyn Buffer, dst: *mut u8);
    fn copyin(&self, src: *const u8, dst: &mut dyn Buffer);
    fn synchronize(&self);
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

pub trait Buffer: core::fmt::Debug {
    fn ptr(&self) -> *mut core::ffi::c_void;
    fn dtype(&self) -> Dtype;
    fn size(&self) -> usize;
    fn to_cpu(&self) -> Vec<u8>;
    fn from_cpu(&mut self, data: &[u8]);
}
