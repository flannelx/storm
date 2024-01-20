#![feature(get_mut_unchecked, exclusive_range_pattern, let_chains)]
#![allow(unused)]

pub mod arg;
pub mod codegen;
pub mod device;
pub mod dtype;
pub mod lazy;
pub mod macros;
pub mod nn;
pub mod ops;
pub mod renderer;
pub mod shape;
pub mod tensor;

#[derive(Debug, Clone)]
pub struct DebugStruct(isize);

lazy_static::lazy_static! {
    pub static ref DEBUG: DebugStruct = DebugStruct(std::env::var("DEBUG").unwrap_or("-1".into()).parse::<isize>().unwrap_or(-1));
}

pub mod prelude {
    pub use crate::device::{prelude::*, Buffer, Device, Program};
    pub use crate::dtype::{self, Dtype};
    pub use crate::izip;
    pub use crate::lazy::LazyBuffer;
    pub use crate::macros::*;
    pub use crate::nn::optim::*;
    pub use crate::tensor::{Tensor, TensorDefaultType};
    pub use crate::DEBUG;
}
