use storm::prelude::*;

fn main() {
    let _a = Tensor::rand([10, 10]);
    let _b = Tensor::rand([10, 10]);
    let a = (&_a * 0.1);
    let b = (&_b * 0.1);
    println!("---------------------------");
    println!("{:?}", (&a * &b).to_vec());
    // println!(
    //     "_a lbid: {:?}, device id {:?}",
    //     _a.buffer.id, _a.buffer.device_buffer
    // );
    // println!(
    //     "_b lbid: {:?}, device id {:?}",
    //     _b.buffer.id, _b.buffer.device_buffer
    // );
    println!("{:?}", (&a.realize() * &b.realize()).to_vec());
    // println!(
    //     "_a lbid: {:?}, device id {:?}",
    //     _a.buffer.id, _a.buffer.device_buffer
    // );
    // println!(
    //     "_b lbid: {:?}, device id {:?}",
    //     _b.buffer.id, _b.buffer.device_buffer
    // );
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
