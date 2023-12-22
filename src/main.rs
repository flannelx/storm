use storm::prelude::*;

fn main() {
    let a = Tensor::rand([16,4,3,3]);
    let b = Tensor::rand([1,4,3,3]);
    b.conv2d(&a).realize();
}
