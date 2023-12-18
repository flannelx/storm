#![feature(get_mut_unchecked)]
#![allow(unused)]


/// Op(Ast: LazyOp) -> Device.get_linearizer(Ast) -> Linearizer(Ast, ...) -> Device.to_program(Linearizer)

pub mod device;
pub mod dtype;
pub mod ops;
pub mod shape;
pub mod tensor;
pub mod lazy;
pub mod codegen;
pub mod arg;
pub mod renderer;
pub mod macros;

pub mod prelude {
    pub use crate::device::{prelude::*, Buffer, Device, Program};
    pub use crate::dtype::{self, Dtype};
    pub use crate::lazy::LazyBuffer;
    pub use crate::tensor::Tensor;
    pub use crate::izip;
    pub use crate::macros::*;
    pub use half::f16;
}
