use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}, shape::ShapeTracker};

fn main() {
    let x = Tensor::arange(27.).reshape([3,3,3]).permute([2,1,0]);
    println!("{:?}", x.contiguous().nd());
}
