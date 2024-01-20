use storm::prelude::*;

fn main() {
    let a = Tensor::from([-0.314]).pow(0.314, false);
    println!("{:?}", a.to_vec());
}
