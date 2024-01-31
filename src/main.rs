use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}};

fn main() {
    let a = Tensor::_arange(0.0046601, 1., 0.0046601);
    println!("{}", a.nd());
}
