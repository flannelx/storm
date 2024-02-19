use half::{bf16, f16};
use ndarray::ArrayD;
use storm::{
    nn::{Conv2d, GroupNorm},
    prelude::*,
    shape::ShapeTracker,
};

fn main() {
    let mut a = Tensor::arange(10.).cast(float16);
    println!("{}", a.nd_t::<f16>());
}
