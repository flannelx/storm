use storm::prelude::*;

fn main() {
    let a = Tensor::scaled_uniform([10,10]);
    // let b = Tensor::scaled_uniform([10,10]);
    // let c = (a+b);
    println!("{:?}", a.to_vec());
}
