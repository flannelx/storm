use ndarray::ArrayD;
use storm::{prelude::*, nn::{GroupNorm, Conv2d}};

fn main() {
    let a = Tensor::rand([768, 49408]);
    let b = Tensor::rand([49408, 768]);
    a.matmul(&b).realize();
}
