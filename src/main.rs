use storm::prelude::*;

fn main() {
    let mut a = Tensor::rand([16,1]);
    a = a.expand([16,20]).permute([1,0]).mean();
    a.backward();
    //println!("{:?}", c.to_vec());
    // let b = Tensor::scaled_uniform([10,10]);
    // let c = b.matmul(&a) * 2;
}
