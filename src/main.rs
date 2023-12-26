use storm::prelude::*;

fn main() {
    let a = Tensor::rand([10, 10]) * 2;
    for i in 0..10 {
        let b = (&a * &a * 2);
        println!("{:?}", (&b * &b).to_vec());
    }
    // let b = Tensor::scaled_uniform([10,10]);
    // let c = b.matmul(&a) * 2;
}
