use ndarray::ArrayD;
use storm::prelude::*;

fn main() {
    let a = Tensor::randn([2, 8, 4096, 40]).transpose(-1, -2);
    println!("{}", a.realize());
}
