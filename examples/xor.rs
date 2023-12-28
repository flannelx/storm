use storm::prelude::*;
fn main() {
    struct Xornet {
        l1: Tensor,
        l2: Tensor,
    }
    impl Xornet {
        pub fn new() -> Self {
            Self {
                // l1: Tensor::scaled_uniform([2, 10]),
                // l2: Tensor::scaled_uniform([10, 1]),
                l1: Tensor::scaled_uniform([10, 10]),
                l2: Tensor::scaled_uniform([10, 10]),
            }
        }
        pub fn forward(&mut self, x: &Tensor) -> Tensor {
            let mut x = &self.l1 * x;
            x = &self.l2 * &x;
            x
        }
    }

    // loss = (y - out).abs().sum() / y.numel()
    let mut model = Xornet::new();
    let mut optim = adam(&[&mut model.l1, &mut model.l2], 0.1);
    let x = Tensor::from([0f32, 0., 0., 1., 1., 0., 1., 1.]).reshape([4, 2]);
    let y = Tensor::from([0f32, 1., 1., 0.]).reshape([1, 4]);
    for i in 0..1 {
        let x = Tensor::rand([10,10]);
        let out = model.forward(&x);
        //println!("{:?}", out.to_vec());
        // let mut loss = (&out - &y).mean();
        // optim.zero_grad();
        // loss.backward();
        // optim.step();
        //println!("model grad realized? {}",model.l1.grad.lock().unwrap().as_ref().unwrap().buffer.is_realized());
        println!("model l1 lbid: {:?} buf_ptr: {:?} weights: {:?}", model.l1.buffer.id, (*model.l1.buffer.device_buffer).as_ref().unwrap().ptr(), model.l1.to_vec());
        println!("model l2 lbid: {:?} buf_ptr: {:?} weights: {:?}", model.l2.buffer.id, (*model.l2.buffer.device_buffer).as_ref().unwrap().ptr(),model.l2.to_vec());
        //println!("{i}: {:?}", loss.to_vec());
    }

    let t = Tensor::from([0., 0.]).reshape([1, 2]);
    let y = Tensor::from([0.]).reshape([1]);
    println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([1., 0.]).reshape([1,2]);
    let y = Tensor::from([1.]).reshape([1]);
    println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([0., 1.]).reshape([1,2]);
    let y = Tensor::from([1.]).reshape([1]);
    println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);

    let t = Tensor::from([1., 1.]).reshape([1,2]);
    let y = Tensor::from([0.]).reshape([1]);
    println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);
}
