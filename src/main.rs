use half::{bf16, f16};
use ndarray::ArrayD;
use storm::{
    nn::{Conv2d, GroupNorm},
    prelude::*,
    shape::ShapeTracker,
};

fn main() {
    let a = Tensor::arange(10.).pow(2., false);
    println!("{}", a.nd())
}
