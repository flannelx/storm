use storm::prelude::*;

fn main() {
    let a = Tensor::rand([10, 10]);
    let b = Tensor::rand([10, 10]);
    println!("---------------------------");
    println!("{:?}", (&a * &b).to_vec());
    //println!("{:?}", (&a * &b).to_vec());
    println!("{:?}", (&a.realize() * &b.realize()).to_vec());
    // println!(
    //     "a lbid: {:?} buf_ptr: {:?} weights: {:?}",
    //     a.buffer.id,
    //     (*a.buffer.device_buffer).as_ref().unwrap().ptr(),
    //     a.to_vec()
    // );
    // println!(
    //     "b lbid: {:?} buf_ptr: {:?} weights: {:?}",
    //     b.buffer.id,
    //     (*b.buffer.device_buffer).as_ref().unwrap().ptr(),
    //     b.to_vec()
    // );
}
