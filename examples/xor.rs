use storm::prelude::*;
use rand::{Rng, thread_rng};
fn main() {
    struct Xornet {
        l1: Tensor,
        l2: Tensor,
    }
    impl Xornet {
        pub fn new() -> Self {
            Self {
                l1: Tensor::randn([2, 10]),
                l2: Tensor::randn([10, 1]),
            }
        }
        pub fn forward(&mut self, x: &Tensor) -> Tensor {
            let mut x = x.matmul(&self.l1).sigmoid();
            x = x.matmul(&self.l2);
            x
        }
    }

    // loss = (y - out).abs().sum() / y.numel()
    let mut model = Xornet::new();
    let mut optim = adam(&[&mut model.l1, &mut model.l2], 0.1);
    let x = Tensor::from([0., 0., 0., 1., 1., 0., 1., 1.]).reshape([4, 2]);
    let y = Tensor::from([0., 1., 1., 0.]).reshape([4, 1]);
    for i in 0..1000 {
        let out = model.forward(&x);
        //println!("{:?}", out.to_vec());
        let mut loss = &y - &out;
        loss = (&loss * &loss).mean();
        optim.zero_grad();
        loss.backward();
        optim.step();
        println!("epoch:{i} loss: {:?}", loss.to_vec());
    }

    let t = Tensor::from([0., 0.]).reshape([1, 2]);
    println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([1., 0.]).reshape([1, 2]);
    println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([0., 1.]).reshape([1, 2]);
    println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([1., 1.]).reshape([1, 2]);
    println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);
}
