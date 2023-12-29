use storm::prelude::*;

fn main() {
    // let mut a = Tensor::rand([10, 10]);
    // a.require_grad = true;
    // let y = Tensor::ones([10, 10]);
    // let mut b = (&a * &y).mean();
    // b.realize();
    // b.backward();
    // println!("grad: {:?}", a.grad.lock().unwrap().as_ref().unwrap().to_vec());

    // let mut a = Tensor::ones([1]);
    // let mut b = Tensor::rand([10,10]);
    // println!("{:?}", (a*b).to_vec());


    let a = Tensor::arange(10.);
    println!("{:?}", a.to_vec());
}
