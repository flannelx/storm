use storm::device::wgpu::WGPUDevice;
use storm::prelude::*;

fn main() {
    let t = Tensor::rand([3,3]);
    println!("{:?}", t.nd());
}
