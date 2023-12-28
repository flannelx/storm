#![feature(get_mut_unchecked)]
#![allow(unused)]

pub mod arg;
pub mod codegen;
/// Op(Ast: LazyOp) -> Device.get_linearizer(Ast) -> Linearizer(Ast, ...) -> Device.to_program(Linearizer)
pub mod device;
pub mod dtype;
pub mod lazy;
pub mod macros;
pub mod ops;
pub mod renderer;
pub mod shape;
pub mod tensor;
pub mod nn;

pub mod prelude {
    pub use crate::device::{prelude::*, Buffer, Device, Program};
    pub use crate::dtype::{self, Dtype};
    pub use crate::izip;
    pub use crate::lazy::LazyBuffer;
    pub use crate::macros::*;
    pub use crate::tensor::{Tensor, TensorDefaultType};
    pub use crate::nn::optim::*;
}
