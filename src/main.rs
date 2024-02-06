use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}, shape::ShapeTracker};

fn main() {
    let mut x = Tensor::rand([256, 256, 3]).reshape([3,256,256]).permute([1,2,0]);
    println!("{:?}", x.realize());
}
