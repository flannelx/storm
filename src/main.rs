use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}, shape::ShapeTracker};

fn main() {
    let a = Tensor::randn([3,3,3]);
    println!("{:?}", ShapeTracker::from_shape(&[1,3,3]))
}
