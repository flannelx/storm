use storm::prelude::*;

fn main() {
    let a = Tensor::rand([16, 4, 3, 3]);
    let b = Tensor::rand([1, 4, 28, 28]);
    let c = b.conv2d(&a) * 2;
    println!("{:?}", (c.sum_all()).to_vec());
}
