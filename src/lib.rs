#![feature(get_mut_unchecked, exclusive_range_pattern, let_chains)]
#![allow(unused, non_snake_case, non_upper_case_globals)]

use std::collections::HashSet;

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
pub mod utils;

#[derive(Debug, Clone)]
pub struct DebugStruct(HashSet<String>);

lazy_static::lazy_static! {
    pub static ref DEBUG: DebugStruct = DebugStruct(HashSet::from_iter(std::env::var("DEBUG").unwrap_or("NONE".into()).split(",").map(|s| s.to_string().to_uppercase()).collect::<Vec<String>>()));
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
    pub use crate::utils::*;
    pub use dtype::*;
    pub use num_traits::{AsPrimitive, Bounded, Float, FromPrimitive, Num, ToPrimitive, NumOps};
}
