use storm::prelude::*;
use num_traits::ToPrimitive;

fn main() {
    let a = Tensor::<f16>::rand([1_000_000]);
    let out = a.realize().to_vec();
    println!("{out:?}");
}
