use storm::prelude::*;
fn main() {
    struct Xornet {
        l1: Tensor,
        l2: Tensor,
    }
    impl Xornet {
        pub fn new() -> Self {
            Self {
                l1: Tensor::scaled_uniform([2, 10]),
                l2: Tensor::scaled_uniform([10, 1]),
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
    let x = Tensor::from([0f32, 0., 0., 1., 1., 0., 1., 1.]).reshape([4, 2]);
    let y = Tensor::from([0f32, 1., 1., 0.]).reshape([1, 4]);
    for _ in 0..1 {
        let out = model.forward(&x);
        //let mut loss = (&out - &y).abs().sum_all() / y.numel();
        let mut loss = (&out - &y).abs().mean();
        optim.zero_grad();
        println!("loss {:?}", loss.to_vec());
        loss.backward();
        optim.step();
    }

    let t = Tensor::from([0., 0.]).reshape([1, 2]);
    let y = Tensor::from([0.]).reshape([1]);
    println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([1., 0.]).reshape([1,2]);
    let y = Tensor::from([1.]).reshape([1]);
    //println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([0., 1.]).reshape([1,2]);
    let y = Tensor::from([1.]).reshape([1]);
    //println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([1., 1.]).reshape([1,2]);
    let y = Tensor::from([0.]).reshape([1]);
    println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);
}
