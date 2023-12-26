use storm::prelude::*;

fn main() {
    let a = Tensor::rand([128, 36864]);
    let b = Tensor::rand([9216, 128]);
    println!("{:?}", b.matmul(&a).mean().to_vec());
    // let b = Tensor::scaled_uniform([10,10]);
    // let c = b.matmul(&a) * 2;
}
