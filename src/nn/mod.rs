use crate::prelude::*;

pub mod optim;

pub struct Conv2d {
    pub weights: Tensor,
    pub bias: Option<Tensor>,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv2d {
    pub fn default(in_channel: usize, out_channel: usize, kernel_size: usize) -> Self {
        Self::new(in_channel, out_channel, kernel_size, 1, 1, 1, 1, false)
    }

    pub fn new(
        in_channel: usize,
        out_channel: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        let weights = Tensor::uniform([out_channel, in_channel / groups, kernel_size, kernel_size]);
        let bound = 1.0 / f32::sqrt(weights.shape().dims[1..].iter().product::<isize>() as f32);
        let bias = if bias {
            Some(Tensor::uniform_range([out_channel], -bound, bound))
        } else {
            None
        };
        Self {
            weights,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        x._conv2d(&self.weights, self.bias.as_ref(), self.groups, self.stride, self.dilation, vec![self.padding])
    }
}
