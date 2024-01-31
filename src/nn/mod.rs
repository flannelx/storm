use std::sync::Arc;

use crate::prelude::*;

pub mod optim;

pub struct Conv2d {
    pub weights: Tensor,
    pub bias: Option<Tensor>,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: Vec<usize>,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv2d {
    pub fn default(in_channel: usize, out_channel: usize, kernel_size: usize) -> Self {
        Self::new(
            in_channel,
            out_channel,
            kernel_size,
            None,
            [],
            None,
            None,
            None,
        )
    }

    pub fn new<V: Into<Vec<usize>>>(
        in_channel: usize,
        out_channel: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: V,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or(1);
        let mut padding = padding.into();
        if padding.len() == 0 {
            padding.push(0);
        }
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let bias = bias.unwrap_or(false);
        let weights = Tensor::kaiming_uniform(
            [out_channel, in_channel / groups, kernel_size, kernel_size],
            Some(5.0.sqrt()),
        );
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
        x._conv2d(
            &self.weights,
            self.bias.as_ref(),
            self.groups,
            self.stride,
            self.dilation,
            self.padding.clone(),
        )
    }
}

impl Sequential for Conv2d {
    fn forward(&self, x: &Tensor, other: Option<SeqArgs>) -> Tensor {
        x._conv2d(
            &self.weights,
            self.bias.as_ref(),
            self.groups,
            self.stride,
            self.dilation,
            self.padding.clone(),
        )
    }
}

pub struct Linear {
    pub weights: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_featuers: usize, bias: Option<bool>) -> Self {
        let bias = bias.unwrap_or(true);
        let bound = 1. / (in_features as f32).sqrt();
        Self {
            weights: Tensor::kaiming_uniform([out_featuers, in_features], Some(5f32.sqrt())),
            bias: if bias {
                Some(Tensor::uniform_range([out_featuers], -bound, bound))
            } else {
                None
            },
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        x.linear(&self.weights.t(), self.bias.as_ref())
    }
}

impl Sequential for Linear {
    fn forward(&self, x: &Tensor, other: Option<SeqArgs>) -> Tensor {
        x.linear(&self.weights.t(), self.bias.as_ref())
    }
}

pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
    pub weights: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: Option<f32>,
        affine: Option<bool>,
    ) -> Self {
        let eps = eps.unwrap_or(1e-5);
        let affine = affine.unwrap_or(true);
        Self {
            num_groups,
            num_channels,
            eps,
            weights: if affine {
                Some(Tensor::ones([num_channels]))
            } else {
                None
            },
            bias: if affine {
                Some(Tensor::zeros([num_channels]))
            } else {
                None
            },
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        let x = x
            .reshape([x.shape()[0], self.num_groups as isize, -1])
            .layernorm(None, Some(self.eps))
            .reshape(x.shape());
        if self.weights.is_none() || self.bias.is_none() {
            x
        } else {
            let shape = vec![vec![1, -1], vec![1; x.shape().len() - 2]].concat();
            &x * &self.weights.as_ref().unwrap().reshape(shape.clone())
                + &self.bias.as_ref().unwrap().reshape(shape)
        }
    }
}

impl Sequential for GroupNorm {
    fn forward(&self, x: &Tensor, other: Option<SeqArgs>) -> Tensor {
        let x = x
            .reshape([x.shape()[0], self.num_groups as isize, -1])
            .layernorm(None, Some(self.eps))
            .reshape(x.shape());
        if self.weights.is_none() || self.bias.is_none() {
            x
        } else {
            let shape = vec![vec![1, -1], vec![1; x.shape().len() - 2]].concat();
            &x * &self.weights.as_ref().unwrap().reshape(shape.clone())
                + &self.bias.as_ref().unwrap().reshape(shape)
        }
    }
}

pub struct Embedding {
    pub vocab_size: usize,
    pub embed_size: usize,
    pub vocab_counter: Arc<Option<Tensor>>,
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_size: usize) -> Self {
        Self {
            vocab_counter: Default::default(),
            vocab_size,
            embed_size,
            weight: Tensor::glorot_uniform([vocab_size, embed_size]),
        }
    }

    pub fn call(&self, idx: &Tensor) -> Tensor {
        if self.vocab_counter.is_none() {
            unsafe {
                *Arc::get_mut_unchecked(&mut self.vocab_counter.clone()) =
                    Some(Tensor::arange(self.vocab_size as f32).reshape([1, 1, self.vocab_size]))
            };
        }
        let [batch_size, seqlen] = idx.shape().dims[..] else {
            panic!()
        };
        if seqlen == 0 {
            Tensor::empty([batch_size, 0, self.embed_size as isize])
        } else {
            (*self.vocab_counter)
                .as_ref()
                .unwrap()
                ._eq(&idx.unsqueeze(2))
                .expand(vec![idx.shape().dims.clone(), vec![self.vocab_size as isize]].concat())
                .matmul(&self.weight)
        }
    }
}

impl Sequential for Embedding {
    fn forward(&self, idx: &Tensor, other: Option<SeqArgs>) -> Tensor {
        if self.vocab_counter.is_none() {
            unsafe {
                *Arc::get_mut_unchecked(&mut self.vocab_counter.clone()) =
                    Some(Tensor::arange(self.vocab_size as f32).reshape([1, 1, self.vocab_size]))
            };
        }
        let [batch_size, seqlen] = idx.shape().dims[..] else {
            panic!()
        };
        if seqlen == 0 {
            Tensor::empty([batch_size, 0, self.embed_size as isize])
        } else {
            (*self.vocab_counter)
                .as_ref()
                .unwrap()
                ._eq(&idx.unsqueeze(2))
                .expand(vec![idx.shape().dims.clone(), vec![self.vocab_size as isize]].concat())
                .matmul(&self.weight)
        }
    }
}

pub struct BatchNorm2d {
    pub eps: f32,
    pub momentum: f32,
    pub track_running_stats: bool,
    pub weights: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub num_batches_tracked: Tensor,
}

impl BatchNorm2d {
    pub fn new(
        size: usize,
        eps: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
        momentum: Option<f32>,
    ) -> Self {
        let eps = eps.unwrap_or(1e-5);
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);
        let momentum = momentum.unwrap_or(0.1);
        let (weights, bias) = if affine {
            (Some(Tensor::ones([size])), Some(Tensor::zeros([size])))
        } else {
            (None, None)
        };
        let running_mean = Tensor::zeros([size]);
        let running_var = Tensor::ones([size]);
        let num_batches_tracked = Tensor::zeros([1]);
        Self {
            eps,
            momentum,
            track_running_stats,
            weights,
            bias,
            running_mean,
            running_var,
            num_batches_tracked,
        }
    }
}

impl Sequential for BatchNorm2d {
    fn forward(&self, x: &Tensor, other: Option<SeqArgs>) -> Tensor {
        todo!()
    }
}

pub struct LayerNorm {
    pub normalized_shape: Vec<isize>,
    pub axis: Vec<isize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub weights: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    pub fn new<S: Into<crate::tensor::shape::Shape>>(
        normalized_shape: S,
        eps: Option<f32>,
        elementwise_affine: Option<bool>,
    ) -> Self {
        let normalized_shape = normalized_shape.into().dims;
        let eps = eps.unwrap_or(1e-5);
        let elementwise_affine = elementwise_affine.unwrap_or(true);
        let axis = v![-1-i, for i in 0..normalized_shape.len() as isize];
        let weights = if elementwise_affine {
            Some(Tensor::ones(normalized_shape.clone()))
        } else {
            None
        };
        let bias = if elementwise_affine {
            Some(Tensor::zeros(normalized_shape.clone()))
        } else {
            None
        };
        Self {
            normalized_shape,
            axis,
            eps,
            elementwise_affine,
            weights,
            bias,
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        let x = x.layernorm(Some(self.axis.clone()), Some(self.eps));
        if self.elementwise_affine {
            &x * self.weights.as_ref().unwrap() + self.bias.as_ref().unwrap()
        } else {
            x
        }
    }
}

impl Sequential for LayerNorm {
    fn forward(&self, x: &Tensor, other: Option<SeqArgs>) -> Tensor {
        let x = x.layernorm(Some(self.axis.clone()), Some(self.eps));
        if self.elementwise_affine {
            &x * self.weights.as_ref().unwrap() + self.bias.as_ref().unwrap()
        } else {
            x
        }
    }
}

pub enum SeqArgs {
    Tensor(Tensor),
    Context(Option<Tensor>),
}

pub trait Sequential {
    fn forward(&self, x: &Tensor, other: Option<SeqArgs>) -> Tensor;
}
