
use std::sync::{Weak, Arc};

use storm::prelude::*;

fn main() {
    for _ in 0..20000 {
        let a = Tensor::_const(0).reshape([4,4]);
        let b = Tensor::_const(0).reshape([4,4]);
        a.matmul(&b).realize();
    }
}
