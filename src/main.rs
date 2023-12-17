use storm::prelude::*;

fn main() {
    let a = Tensor::rand([1_000]);
    let out = a.to_vec::<f32>();
    println!("{:?}", out);
}
