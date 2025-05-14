// pub enum OpT {
//     Add,
//     Sub,
//     Mul,
//     Div
// }
//
// pub struct Op {
// }

use std::sync::Arc;

pub trait Device {
}

pub trait Buffer {}

#[allow(non_camel_case_types)]
pub enum Dtype {
    r#f32
}

impl Dtype {
    fn bytesize(&self) -> usize {
        match self {
            Dtype::f32 => 4,
        }
    }
}

pub enum Op {
    Load {
        buffer: Arc<dyn Buffer>,
        dtype: Dtype,
        shape: Vec<isize>,
    },
    Reshape {
        shape: Vec<isize>,
    },
    Add {
        buffer: Arc<dyn Buffer>,
        dtype: Dtype,
        lhs: Arc<Op>,
        rhs: Arc<Op>
    }
}

fn main() {
    println!("Hello, world!");
}
