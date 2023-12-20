use storm::prelude::*;

fn main() {
    // let a = Tensor::<f32>::rand([100, 100]);
    // let b = Tensor::<f32>::rand([100, 100]);
    // a.matmul(&b).to_vec();
    //let out = a.to_vec();
    //println!("{out:?}");
    let a = Tensor::<f32>::ones([10,10]);
    let b = Tensor::<f32>::ones([10,10]);
    (a+b).realize();
}
