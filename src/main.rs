use ndarray::ArrayD;
use storm::{
    nn::{Conv2d, GroupNorm},
    prelude::*,
    shape::ShapeTracker,
};

fn main() {
    // let timesteps = Tensor::_const(801);
    // let emb = timestep_embedding(&timesteps, 320, None);
    let n = 256;
    let a = Tensor::rand([n, n]).pad([(1,1), (1,1)], 0);
    println!("{}", a.nd());
}

pub fn timestep_embedding(timesteps: &Tensor, dim: usize, max_period: Option<usize>) -> Tensor {
    let half = dim as f32 / 2.;
    let max_period = max_period.unwrap_or(10000);
    let freqs = (-(max_period as f32).ln() * Tensor::arange(half) / half).exp();
    let args = timesteps * &freqs;
    Tensor::cat(&args.cos(), &[args.sin()], None)
}
