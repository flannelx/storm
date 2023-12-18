use storm::prelude::*;
use num_traits::ToPrimitive;

fn main() {
    let a = Tensor::rand([1_000]);
    let out = a.to_vec::<f32>();
}
