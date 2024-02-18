use ndarray::ArrayD;
use storm::{
    nn::{Conv2d, GroupNorm},
    prelude::*,
    shape::ShapeTracker,
};

fn main() {
    let mut a = Tensor::arange(27.).reshape([3,3,3]);
    println!("{}", a[0..2].shape());
    println!("{}", a[0..2].nd());
}
