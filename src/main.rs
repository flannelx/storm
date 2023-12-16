use half::f16;
use storm::prelude::*;

fn main() {
    let a = Tensor::rand([1_000_000_00]);
    let out = a.realize().to_vec();
    let mut v = Vec::new();
    for n in out.windows(a.dtype.size).step_by(a.dtype.size) {
        v.push(f32::from_le_bytes(n.try_into().unwrap()))
    }
    println!("{v:?}");
}
