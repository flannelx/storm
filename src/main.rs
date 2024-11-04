use std::time::SystemTime;

use storm::prelude::*;

fn main() {
    let m = 512;
    let n = 512;
    let a = Tensor::rand([m, n]);
    let b = Tensor::rand([n, m]);
    let s = SystemTime::now();
    println!("{:?}", (a * b).realize());
    let e = s.elapsed().unwrap().as_nanos() as f64;
    let flops = (m * n) as f64 / e * 10e8 / 10e9;
    println!("{flops} gflop/s");
}
