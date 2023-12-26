use storm::prelude::*;

fn main() {
    let a = Tensor::ones([10, 2]).sum(0).exp();
    // let b = Tensor::scaled_uniform([10,10]);
    // let c = b.matmul(&a) * 2;
    println!("{:?}", (a).to_vec());
}
