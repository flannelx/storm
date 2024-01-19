
use std::sync::{Weak, Arc};

use storm::prelude::*;

fn main() {
    let a = Tensor::_arange(-10.0, 10., 1.).pow(9.9, false);
    println!("{:?}", a.to_vec());
}
