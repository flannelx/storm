use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}, shape::ShapeTracker};

fn main() {
    let a = Tensor::rand([3,3]);
    let b = Tensor::rand([3,3]);
    println!("{:?}", a.matmul(&b).nd());
}
