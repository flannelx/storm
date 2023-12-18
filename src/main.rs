use storm::prelude::*;

fn main() {
    let a = Tensor::<f32>::rand([1_000]) - 1.;
    let out = a.to_vec();
    println!("{out:?}");
}
