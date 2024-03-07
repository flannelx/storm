use half::{bf16, f16};
use ndarray::ArrayD;
use storm::{
    nn::{Conv2d, GroupNorm},
    prelude::*,
    shape::ShapeTracker,
};

fn main() {
    let a = Tensor::rand([3,3]);
    let b = Tensor::rand([3,3]).cast(float16).realize();
    let x = Tensor::rand([3,3]);
    let c = (a + b + x).realize();
}
