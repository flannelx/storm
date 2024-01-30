use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}};

fn main() {
    let a = Tensor::ones([1,1,8,8]);
    let c = Conv2d::default(1, 3, 3);
    println!("{:?}", c.call(&a).nd())
}
