use storm::prelude::*;

fn main() {
    let c = Tensor::ones([100, 100]);
    println!("{:?}", c.realize().to_vec());
}
