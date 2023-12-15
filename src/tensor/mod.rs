pub mod core_ops;
pub mod mlops;
pub mod shape;
use crate::arg::Arg;
use crate::dtype::type_to_dtype;
use crate::ops::Load;
use crate::ops::OpType;
use crate::prelude::*;
use crate::prelude::*;
use crate::tensor::mlops::*;
use crate::tensor::shape::Shape;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

pub type TensorDefaultType = f32;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct TensorId(pub(crate) usize);

pub(crate) fn tensor_id() -> TensorId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    TensorId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[derive(Clone)]
pub struct Tensor {
    pub buffer: LazyBuffer,
    pub require_grad: bool,
    pub grad: Arc<Mutex<Option<Tensor>>>,
    pub _ctx: Option<Box<dyn Function>>,
    pub id: TensorId,
    pub dtype: Dtype,
    pub device: String,
}

impl Tensor {
    pub fn device(&self) -> String {
        self.device.clone()
    }

    pub fn dtype(&self) -> String {
        self.dtype.type_name.to_string()
    }

    pub fn shape(&self) -> Shape {
        self.buffer.shape.clone().into()
    }

    pub fn strides(&self) -> Vec<isize> {
        self.buffer.st.strides()
    }

    pub fn from_buf(buf: LazyBuffer) -> Self {
        Self {
            require_grad: false,
            grad: Arc::default(),
            _ctx: None,
            id: tensor_id(),
            dtype: buf.dtype.clone(),
            device: buf.device.clone(),
            buffer: buf,
        }
    }

    pub fn from<T: num_traits::Num, V: Into<Vec<T>>>(data: V) -> Self {
        let data = data.into();
        let buffer = LazyBuffer::from_cpu(data);
        Self {
            require_grad: false,
            _ctx: None,
            id: tensor_id(),
            grad: Arc::new(Mutex::new(None)),
            dtype: buffer.dtype.clone(),
            device: buffer.device.clone(),
            buffer,
        }
    }

    pub fn contiguous(&self) -> Self {
        todo!()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut ret: Vec<f32> = Vec::new();
        let bytes = self
            .buffer
            .realize()
            .base()
            .device_buffer
            .as_deref()
            .unwrap()
            .to_cpu();
        for n in bytes.windows(4).step_by(4) {
            ret.push(f32::from_le_bytes([n[0], n[1], n[2], n[3]]));
        }
        ret
    }

    // ------------ Load
    pub fn _load(
        op: OpType,
        size: usize,
        dtype: Dtype,
        device: Option<&str>,
        args: Option<Vec<Arg>>,
    ) -> Self {
        Self::from_buf(LazyBuffer::loadop(
            op,
            &[size as isize],
            dtype,
            {
                if device.is_none() {
                    "GPU"
                } else {
                    device.unwrap()
                }
            },
            {
                if args.is_some() {
                    args
                } else {
                    None
                }
            },
            None,
        ))
    }

    pub fn empty<S: Into<Shape>>(shape: S, dtype: Dtype) -> Self {
        let shape = shape.into();
        assert!(
            shape.dims.iter().any(|n| *n >= 0),
            "load op can not infer dim"
        );
        Self::_load(OpType::Load(Load::Empty), shape.numel(), dtype, None, None)
    }

    pub fn _const<T: num_traits::ToBytes>(value: T) -> Self {
        let dtype = crate::dtype::name_to_dtype(std::any::type_name::<T>());
        Self::_load(
            OpType::Load(Load::Const),
            1,
            dtype,
            None,
            Some(vec![Arg::Num(value.to_le_bytes().as_ref().to_vec())]),
        )
    }

    pub fn const_like<T: num_traits::ToBytes>(&self, const_value: T) -> Self {
        Self::_const(const_value)
            .reshape(vec![1; self.shape().len()])
            .expand(self.shape().dims)
    }

    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::_const(0 as TensorDefaultType)
            .reshape(vec![1; shape.len()])
            .expand(shape.dims)
    }

    pub fn ones<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::_const(1 as TensorDefaultType)
            .reshape(vec![1; shape.len()])
            .expand(shape.dims)
    }

    pub fn rand<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::_load(
            OpType::Load(Load::Rand),
            shape.numel(),
            type_to_dtype::<TensorDefaultType>(),
            None,
            None,
        )
    }

    pub fn randn<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let mut ret = Self::rand(shape.clone());
        let sec_ = Self::rand(shape.clone());
        ret = ret * (core::f32::consts::PI * 2.0);
        ret = ret.cos() * ((1.0f32 - sec_).log() * -2.0f32).sqrt();
        ret
    }

    pub fn normal<S: Into<Shape>>(shape: S, mean: f32, std: f32) -> Self {
        Self::randn(shape) * std + mean
    }

    pub fn uniform<S: Into<Shape>>(shape: S) -> Self {
        let low = -1.0;
        let high = 1.0;
        Self::rand(shape) * (high - low) + low
    }

    pub fn uniform_range<S: Into<Shape>>(shape: S, low: f32, high: f32) -> Self {
        Self::rand(shape) * (high - low) + low
    }

    //def scaled_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, **kwargs).mul(math.prod(shape)**-0.5)
    pub fn scaled_uniform<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::uniform(shape.clone()) * (shape.numel() as f32).powf(-0.5)
    }

    pub fn glorot_uniform<S: Into<Shape>>(shape: S) -> Self {
        // Tensor.uniform(*shape, **kwargs).mul((6/(shape[0]+math.prod(shape[1:])))**0.5)
        let shape = shape.into();
        Self::uniform(shape.clone())
            * (6.0 / (shape[0] + shape.dims[1..].iter().product::<isize>()) as f32).powf(0.5)
    }

    pub fn kaiming_uniform<S: Into<Shape>>(_shape: S) -> Self {
        todo!()
    }

    pub fn kaiming_normal<S: Into<Shape>>(_shape: S) -> Self {
        todo!()
    }

    pub fn dropout(self, p: Option<f32>) -> Self {
        // mask = (Tensor.rand(*self.shape, requires_grad=False, device=self.device) >= p).cast(dtypes.bool)
        // return self * mask * (1/(1.0 - p))
        let p = if p.is_some() { p.unwrap() } else { 0.2 };
        let mask = Self::rand(self.shape())._ge(&Tensor::from([p]));
        self * mask * (1.0 / (1.0 - p))
    }

    // ------------ Movement
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Self {
        Reshape::default().apply(self, None, None, Some(shape.into().dims), None)
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Self {
        Expand::default().apply(self, None, None, Some(shape.into().dims), None)
    }

    pub fn permute<S: Into<Shape>>(&self, shape: S) -> Self {
        Permute::default().apply(self, None, None, Some(shape.into().dims), None)
    }

    pub fn shrink<A: Into<Vec<(usize, usize)>>>(&self, arg: A) -> Self {
        let arg = arg.into();
        if !arg
            .iter()
            .zip(self.shape().dims)
            .any(|(a, sh)| *a != (0, sh as usize))
        {
            return self.clone();
        }
        let flatten_p: Vec<isize> = arg
            .iter()
            .map(|(p1, p2)| vec![*p1 as isize, *p2 as isize])
            .flatten()
            .collect();
        Shrink::default().apply(self, None, None, Some(flatten_p), None)
    }

    pub fn pad<A: Into<Vec<(usize, usize)>>>(
        &self,
        arg: A,
        const_value: impl num_traits::ToBytes,
    ) -> Self {
        let flatten_p: Vec<isize> = arg
            .into()
            .iter()
            .map(|(p1, p2)| vec![*p1 as isize, *p2 as isize])
            .flatten()
            .collect();
        Pad::default().apply(
            self,
            None,
            None,
            Some(flatten_p.into()),
            Some(const_value.to_le_bytes().as_ref().to_vec()),
        )
        // Tensor {
        //     inner: self.inner.pad(arg, const_value),
        //     require_grad: false,
        //     _ctx: None,
        //     id: tensor_id(),
        //     grad: None,
        // }
    }

    pub fn pad2d<P: Into<Vec<usize>>>(
        &self,
        padding: P,
        const_value: impl num_traits::ToBytes,
    ) -> Self {
        let padding = padding.into();
        let slc: Vec<(isize, isize)> = padding
            .iter()
            .step_by(2)
            .zip(
                padding[1..]
                    .iter()
                    .step_by(2)
                    .zip(self.shape().dims.iter().rev()),
            )
            .map(|(p0, (p1, s))| (-(*p0 as isize), *s as isize + *p1 as isize))
            .rev()
            .collect();
        //slc.iter().step_by(step)
        // println!("slc {:?}", slc);
        let mut slice_shape: Vec<(isize, isize)> = self.shape().dims
            [..self.shape().len() - padding.len() / 2]
            .iter()
            .map(|sh| (0, *sh as isize))
            .collect();
        slice_shape.extend(slc.iter());
        // println!("slice shape {:?}", slc);
        self.slice(slice_shape, const_value)
    }

    // -------- unary

    pub fn log(&self) -> Self {
        Log::default().apply(&self, None, None, None, None)
    }

    pub fn log2(&self) -> Self {
        Log::default().apply(&self, None, None, None, None) / 2.0f32.log(core::f32::consts::E)
    }

    pub fn exp(&self) -> Self {
        Exp::default().apply(&self, None, None, None, None)
    }

    pub fn relu(&self) -> Self {
        Relu::default().apply(self, None, None, None, None)
    }

    pub fn leakyrelu(&self, neg_slope: Option<f32>) -> Self {
        let neg_slope = if neg_slope.is_none() {
            0.01
        } else {
            neg_slope.unwrap()
        };
        self.relu() - (-neg_slope * self).relu()
    }

    pub fn sigmoid(&self) -> Self {
        Sigmoid::default().apply(self, None, None, None, None)
    }

    pub fn tanh(&self) -> Self {
        // 2.0 * ((2.0 * self).sigmoid()) - 1.0
        2.0 * ((2.0f32 * self).sigmoid()) - 1.0
    }

    pub fn _softmax(&self, axis: isize) -> (Self, Self, Self) {
        let m = self - &self.max_keepdim(axis);
        let e = m.exp();
        let ss = e.sum_keepdim(axis);
        (m, e, ss)
    }

    pub fn softmax(&self) -> Self {
        let (_, e, ss) = self._softmax(-1);
        e / ss
    }

    pub fn log_softmax(&self) -> Self {
        let (m, _, ss) = self._softmax(-1);
        m - ss.log()
    }

    pub fn sin(&self) -> Self {
        Sin::default().apply(self, None, None, None, None)
    }

    pub fn cos(&self) -> Self {
        (-self + core::f32::consts::PI / 2.0).sin()
    }

    pub fn sqrt(&self) -> Self {
        Sqrt::default().apply(self, None, None, None, None)
    }

    pub fn rsqrt(&self) -> Self {
        (&Self::ones([1]) / self).sqrt()
    }

    pub fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    pub fn sum_keepdim(&self, axis: isize) -> Self {
        let axis = if axis < 0 {
            self.shape().len() as isize + axis
        } else {
            axis
        };
        let mut shape = self.shape().clone();
        shape[axis] = 1;
        Sum::default().apply(&self, None, None, Some(shape.dims), None)
    }

    pub fn sum(&self, axis: isize) -> Self {
        let mut shape = self.shape().clone();
        let ret = self.sum_keepdim(axis);
        let axis = if axis < 0 {
            (shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };
        shape.dims.remove(axis);
        ret.reshape(shape)
    }

    pub fn sum_all(&self) -> Self {
        self.reshape([self.shape().dims.iter().product::<isize>()])
            .sum_keepdim(0)
    }

    pub fn _reduce(&self, fxn: impl Function, axis: Option<Vec<isize>>, keepdim: bool) -> Self {
        todo!()
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn max(&self, axis: isize) -> Self {
        let mut shape = self.shape();
        let _axis = if axis < 0 {
            (self.shape().len() as isize + axis) as usize
        } else {
            axis as usize
        };
        // FIXME: Teenygrad just output a clone of self when axis >= self.ndim
        if _axis >= self.ndim() {
            return self.clone();
        }
        let ret = self.max_keepdim(axis);
        shape.dims.remove(_axis);
        ret.reshape(shape)
    }

    pub fn max_keepdim(&self, axis: isize) -> Self {
        let axis = if axis < 0 {
            self.shape().len() as isize + axis
        } else {
            axis
        };
        let mut shape = self.shape().clone();
        shape.dims.push(axis);
        Max::default().apply(&self, None, None, Some(shape.dims), None)
    }

    pub fn max_all(&self) -> Self {
        self.reshape([self.shape().dims.iter().product::<isize>()])
            .max_keepdim(0)
    }

    pub fn _argmax(&self, axis: Option<isize>, keepdim: bool) -> Self {
        if axis.is_none() {
            let idx = (self._eq(&self.max_all()))
                * Self::_arange(self.numel() as f32 - 1., -1., -1.).reshape(self.shape());
            return self.numel() - idx.max_all() - 1;
        }
        let mut axis = axis.unwrap();
        axis = axis + if axis < 0 { self.ndim() as isize } else { axis };
        let m = self._eq(&self.max_keepdim(axis));
        let idx = m * Self::_arange(self.shape()[axis as usize] as f32 - 1., -1., -1.).reshape(
            [
                vec![self.shape()[axis as usize]],
                vec![1; self.ndim() - axis as usize - 1],
            ]
            .concat(),
        );
        if keepdim {
            return self.shape()[axis as usize] - idx.max_keepdim(axis) - 1;
        }
        self.shape()[axis as usize] - idx.max(axis) - 1
    }

    pub fn argmax(&self, axis: isize) -> Self {
        self._argmax(Some(axis), false)
    }

    pub fn argmax_keepdim(&self, axis: isize, _keepdim: bool) -> Self {
        self._argmax(Some(axis), true)
    }

    pub fn argmax_all(&self) -> Self {
        self._argmax(None, false)
    }

    pub fn matmul(&self, w: &Self) -> Self {
        let n1 = self.shape().len();
        let n2 = w.shape().len();
        {
            let tmp = -(n2.min(2) as isize);
            assert!(n1 != 0 && n2 != 0);
            assert!(
                self.shape()[-1] == w.shape()[tmp],
                "{}[-1] != {}[{tmp}]",
                self.shape(),
                w.shape()
            );
        }
        let mut x_reshape = Vec::new();
        //x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
        x_reshape.extend_from_slice(&self.shape().dims[..n1 - 1]);
        x_reshape.extend_from_slice(&vec![1; (n1 - 1).min(n2 - 1).min(1)]);
        x_reshape.push(self.shape()[-1]);
        let x = self.reshape(x_reshape);
        // w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
        let mut w_reshape = Vec::new();
        w_reshape.extend_from_slice(&w.shape().dims[0..n2 - 2]);
        w_reshape.extend_from_slice(&vec![1; (n1 - 1).min(n2 - 1).min(1)]);
        w_reshape.extend_from_slice(&w.shape().dims[n2 - (n2.min(2))..]);
        let w = w.reshape(w_reshape).transpose(-1, -(n2.min(2) as isize));
        (x * w).sum(-1)
    }

    pub fn _broadcast_r(x: &Self, y: &Self) -> (Self, Self) {
        Self::_broadcast(y, x)
    }

    pub fn _broadcast(x: &Self, y: &Self) -> (Self, Self) {
        let mut xshape = x.shape();
        let mut yshape = y.shape();
        let mut x = x.clone();
        let mut y = y.clone();
        if xshape == yshape {
            return (x, y);
        }
        let shape_delta = xshape.len() as isize - yshape.len() as isize;
        if shape_delta > 0 {
            let mut ysh = vec![1; shape_delta as usize];
            ysh.extend_from_slice(&yshape.dims);
            y = y.reshape(ysh);
        } else if shape_delta < 0 {
            let mut xsh = vec![1; (shape_delta * -1) as usize];
            xsh.extend_from_slice(&xshape.dims);
            x = x.reshape(xsh);
        }
        xshape = x.shape();
        yshape = y.shape();
        if xshape == yshape {
            return (x, y);
        }

        let shape_ret = Vec::<isize>::from(
            xshape
                .dims
                .iter()
                .zip(yshape.dims.iter())
                .map(|(x, y)| *x.max(y))
                .collect::<Vec<isize>>(),
        );
        if xshape.dims != shape_ret {
            x = x.expand(shape_ret.clone());
        }
        if yshape.dims != shape_ret {
            y = y.expand(shape_ret.clone());
        }
        (x, y)
    }

    pub fn transpose(&self, d1: isize, d2: isize) -> Self {
        let d1 = if d1 < 0 {
            (self.shape().len() as isize + d1) as usize
        } else {
            d1 as usize
        };
        let d2 = if d2 < 0 {
            (self.shape().len() as isize + d2) as usize
        } else {
            d2 as usize
        };

        let mut p = (0..self.shape().len()).collect::<Vec<usize>>();
        p.swap(d1, d2);
        self.permute(p)
    }

    // self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
    pub fn max_pool2d(&self) -> Self {
        let k_ = [2, 2];
        let stride = 2;
        let dilation = 1;
        let mut ret = self._pool(k_, stride, dilation);
        for i in -2..0 {
            ret = ret.max(i);
        }
        ret
    }


    #[rustfmt::skip]
    pub fn _pool<S: Into<Shape>>(&self, k_: S, stride: usize, dilation: usize) -> Self {
        let self_shape = self.shape();
        let k_ = k_.into().dims.iter().map(|n| *n as usize).collect::<Vec<usize>>();
        let d_ = vec![dilation;k_.len()];
        let s_ = vec![stride;k_.len()];
        assert!(self_shape.len() >= k_.len(), "can't pool {self_shape:?} with {k_:?}");
        assert!(k_.len() == s_.len() && s_.len() == d_.len(), "stride/dilation mismatch kernel:{k_:?} stride:{s_:?} dilation:{d_:?}");
        let slc_prefix: Vec<(isize, isize)> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| (0, *sh as isize))
            .collect();
        let prefix: Vec<usize> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| *sh as usize)
            .collect();
        let i_: Vec<usize> = self_shape.dims[self_shape.len() - k_.len()..]
            .iter()
            .map(|sh| *sh as usize)
            .collect();

        let xup = if k_.iter().zip(s_.iter()).any(|(k, s)| k > s) || d_.iter().any(|d| *d != 1) {
            let o_: Vec<usize> = i_
                .iter()
                .zip(d_.iter().zip(k_.iter().zip(s_.iter())))
                .map(|(i, (d, (k, s)))| (i - d * (k - 1) - 1) / s + 1)
                .collect();
            let e_: Vec<usize> = k_
                .iter()
                .zip(i_.iter().zip(d_.iter()))
                .map(|(k, (i, d))| f32::ceil((k * (i + d)) as f32 / *i as f32) as usize)
                .collect();
            self.reshape([prefix.clone(), i_.iter().flat_map(|i| [1, *i]).collect()].concat())
                .expand([prefix.clone(), e_.iter().zip(i_.iter()).flat_map(|(e, i)| [*e, *i]).collect()].concat())
                .reshape([prefix.clone(), e_.iter().zip(i_.iter()).map(|(e, i)| *e * *i).collect()].concat())
                .slice(vec![slc_prefix.clone(), c![(0, (k*(i+d)) as isize), for (k, i, d) in izip!(k_.iter(), i_.iter(), d_.iter())]].concat(), 0)
                .reshape(vec![prefix.clone(), c![[*k, i + d], for (k, i, d) in izip!(k_.iter(), i_.iter(), d_.iter())].concat()].concat())
                .slice(vec![slc_prefix.clone(), c![[(0, *k as isize),(0, (o * s) as isize)], for (k, o, s) in izip!(k_.iter(), o_.iter(), s_.iter())].concat()].concat(), 0)
                .reshape(vec![prefix.clone(), c![[k, o, s], for (&k, &o, &s) in izip!(k_.iter(), o_.iter(), s_.iter())].concat()].concat())
                .slice(vec![slc_prefix.clone(), c![[(0, *k as isize), (0, *o as isize), (0, 1)], for (k, o) in izip!(k_.iter(), o_.iter())].concat()].concat(), 0)
                .reshape(vec![prefix.clone(), c![[k, o], for (&k, &o) in izip!(k_.iter(), o_.iter())].concat()].concat())
                .permute(vec![c![i, for i in 0..prefix.len()], c![prefix.len() + i * 2 + 1, for i in 0..k_.len()], c![prefix.len() + i * 2, for i in 0..k_.len()]].concat())
        } else {
            let o_:Vec<usize> = i_.iter().zip(s_.iter().zip(k_.iter())).map(|(i, (s, k))| (*i+(*s-*k))/s).collect();

            let mut xup = self.slice(vec![slc_prefix.clone(), c![(0, (o * s)as isize), for (&o, &s) in izip!(o_.iter(), s_.iter())]].concat(), 0);
            xup = xup.reshape(vec![prefix.clone(), c![[o, s], for (&o, &s) in izip!(o_.iter(), s_.iter())].concat()].concat());
            let mut xup = self.slice(vec![slc_prefix.clone(), c![[(0, o as isize), (0, k as isize)], for (&o, &k) in izip!(o_.iter(), k_.iter())].concat()].concat(), 0);

            let mut tmp: Vec<usize> = (0..prefix.len()).into_iter().collect();
            tmp.extend((0..k_.len()).map(|i| prefix.len() + i * 2));
            tmp.extend((0..k_.len()).map(|i| prefix.len() + i * 2 + 1));
            xup.permute(tmp)
        };
        xup
    }

    pub fn slice<A: Into<Vec<(isize, isize)>>>(
        &self,
        arg: A,
        const_value: impl num_traits::ToBytes,
    ) -> Self {
        let arg = arg.into();
        let self_shape = self.shape();
        let padding: Vec<(usize, usize)> = arg
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    0.max(-p.0) as usize,
                    0.max(p.1 - self_shape[i] as isize) as usize,
                )
            })
            .collect();
        //println!("padding in slice {:?}", padding);
        let shrink: Vec<(usize, usize)> = arg
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    (p.0 + padding[i].0 as isize) as usize,
                    (p.1 + padding[i].0 as isize) as usize,
                )
            })
            .collect();
        //println!("shrink in slice {:?}", shrink);
        self.pad(padding, const_value).shrink(shrink)
    }

    pub fn conv2d(&self, weigth: &Self) -> Self {
        self._conv2d(weigth, None, 1, 1, 1, [0])
    }

    pub fn _conv2d<V: Into<Vec<usize>>>(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        groups: usize,
        stride: usize,
        dilation: usize,
        padding: V,
    ) -> Self {
        // assert!(self.shape().len() == 4, "conv2d weight should be 4d");
        // let shape = self.shape();
        // let bs = shape[0] as usize;
        // let cin_ = shape[1] as usize;
        // let cout = weight.shape()[0] as usize;
        // let cin = weight.shape()[1] as usize;
        // let hw = [shape[2] as usize, shape[3] as usize];
        let [bs, cin_] = self.shape().dims[..2] else {
            panic!()
        };
        let bs = bs as usize;
        let cin_ = cin_ as usize;
        let [cout, cin] = weight.shape().dims[..2] else {
            panic!()
        };
        let cout = cout as usize;
        let cin = cin as usize;
        let hw = weight.shape().dims[2..]
            .iter()
            .map(|i| *i as usize)
            .collect::<Vec<usize>>();
        assert!(
            groups * cin == cin_ && self.shape().len() == weight.shape().len(),
            "Input Tensor shape {} does not match the shape of the weights {}. ({} vs. {})",
            self.shape(),
            weight.shape(),
            groups * cin,
            cin_
        );
        let padding = padding.into();
        let padding_ = if padding.len() == 1 {
            vec![padding[0]; 2 * hw.len()]
        } else if padding.len() == 2 * hw.len() {
            padding.clone()
        } else {
            let mut ret = vec![padding.clone(), padding.clone()].concat();
            ret.reverse();
            ret
        };
        assert!(
            padding_.len() != self.shape().len() * 2,
            "padding should be dim x2 padding: {:?} shape:{}",
            padding_,
            self.shape()
        );
        let mut x = self
            .pad2d(padding_, 0)
            ._pool(Shape::from(hw.clone()), stride, dilation);
        let rcout = cout / groups;
        let oyx = x.shape().dims[2..x.shape().len() - hw.len()]
            .iter()
            .map(|i| *i as usize)
            .collect::<Vec<usize>>();
        //reshape(bs, groups, cin, 1, *oyx, *HW)
        let mut rsh_tmp = vec![bs, groups, cin, 1];
        rsh_tmp.extend(oyx.iter());
        rsh_tmp.extend(hw.iter());
        //expand(bs, groups, cin, rcout, *oyx, *HW)
        let mut exp_tmp = vec![bs, groups, cin, rcout];
        exp_tmp.extend(oyx.iter());
        exp_tmp.extend(hw.iter());
        //permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])
        let mut permute_tmp = vec![0, 1, 3];
        permute_tmp.extend((0..oyx.len()).into_iter().map(|i| 4 + i));
        permute_tmp.push(2);
        permute_tmp.extend((0..hw.len()).into_iter().map(|i| 4 + oyx.len() + i));
        x = x.reshape(rsh_tmp).expand(exp_tmp).permute(permute_tmp);
        // ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx)
        let mut w_rsh_tmp = vec![1, groups, rcout];
        w_rsh_tmp.extend(vec![1; oyx.len()]);
        w_rsh_tmp.push(cin);
        w_rsh_tmp.extend(hw.iter());
        let mut ret = x * weight.reshape(w_rsh_tmp);
        for i in 0..oyx.len() + 1 {
            let reduce_i = -1 - (i as isize);
            ret = ret.sum_keepdim(reduce_i);
        }
        let mut ret_rsh_tmp = vec![bs, cout];
        ret_rsh_tmp.extend(oyx.iter());
        ret = ret.reshape(ret_rsh_tmp);
        if bias.is_none() {
            return ret;
        }
        // bias.reshape(1, -1, *[1] * len(HW))
        let bias = bias.unwrap();
        let mut b_rsh_tmp = vec![1, bias.shape().len()];
        b_rsh_tmp.extend(vec![1; hw.len()]);
        ret + bias.reshape(b_rsh_tmp)
    }

    pub fn t(&self) -> Self {
        self.transpose(1, 0)
    }

    pub fn deepwalk(&self) -> Vec<Self> {
        let mut ret = Vec::new();
        let mut visisted = HashSet::new();
        Self::_deepwalk(self, &mut visisted, &mut ret);
        //println!("{ret:?}");
        // for (i, n) in ret.iter().enumerate() {
        //     println!("i:{:?} {:?}", i, n);
        // }
        ret
    }

    pub fn _deepwalk(node: &Self, visisted: &mut HashSet<TensorId>, ret: &mut Vec<Self>) {
        visisted.insert(node.id);
        if node._ctx.is_none() {
            return;
        }

        for n in node._ctx.as_ref().unwrap().parents_ref().iter() {
            if !visisted.contains(&n.id) {
                Self::_deepwalk(n, visisted, ret);
            }
        }
        ret.push(node.clone());
    }

    pub fn backward(&mut self) {
        assert!(
            self.shape().len() == 1,
            "backward can only be called for scalar tensors, but it has shape {}",
            self.shape()
        );
        (*self.grad.lock().unwrap()) = Some(Self::ones([1]));
        let deepwalked = self.deepwalk().into_iter();
        let _deepwalked_len = deepwalked.len();
        for mut t0 in deepwalked.rev() {
            let t0g_clone = (*t0.grad.lock().unwrap())
                .as_ref()
                .expect("t0 should have a grad")
                .buffer
                .clone();
            let grads = match t0._ctx.as_mut().unwrap().backward(t0g_clone) {
                Grad::One(g) => vec![Some(Tensor {
                    dtype: g.dtype.clone(),
                    device: g.device.clone(),
                    buffer: g,
                    require_grad: false,
                    grad: Arc::default(),
                    _ctx: None,
                    id: tensor_id(),
                })],
                Grad::Two(g1, g2) => {
                    let mut out = vec![];
                    out.push(if let Some(g) = g1.as_ref() {
                        // if g.to_vec().iter().any(|n| n.is_nan()) {
                        //     panic!("g has NaN")
                        // } else {
                        //     println!("{:?}", g);
                        // }
                        Some(Tensor {
                            dtype: g.dtype.clone(),
                            device: g.device.clone(),
                            buffer: g.clone(),
                            require_grad: false,
                            grad: Arc::default(),
                            _ctx: None,
                            id: tensor_id(),
                        })
                    } else {
                        None
                    });
                    out.push(if let Some(g) = g2.as_ref() {
                        // if g.to_vec().iter().any(|n| n.is_nan()) {
                        //     panic!("g has NaN")
                        // } else {
                        //     println!("{:?}", g);
                        // }
                        Some(Tensor {
                            dtype: g.dtype.clone(),
                            device: g.device.clone(),
                            buffer: g.clone(),
                            require_grad: false,
                            grad: Arc::default(),
                            _ctx: None,
                            id: tensor_id(),
                        })
                    } else {
                        None
                    });
                    out
                }
            };
            assert!(t0._ctx.as_ref().unwrap().parents_ref().len() == grads.len());
            for (t, g) in t0
                ._ctx
                .as_mut()
                .unwrap()
                .parents_mut()
                .iter()
                .zip(grads.iter())
            {
                if g.is_none() || !t.require_grad {
                    continue;
                }
                let g = g.as_ref().unwrap();
                assert!(
                    t.shape() == g.shape(),
                    "grad shape must match tensor shape, {} != {}",
                    g.shape(),
                    t.shape()
                );
                let mut t_grad = t.grad.lock().unwrap();
                if t_grad.is_none() {
                    *t_grad = Some(g.clone());
                } else {
                    *t_grad = Some(g + (*t_grad).as_ref().unwrap());
                }
                // if (*t_grad)
                //     .as_ref()
                //     .unwrap()
                //     .to_vec()
                //     .iter()
                //     .any(|n| n.is_nan())
                // {
                //     panic!("g has NaN")
                // } else {
                //     println!("{:?}", g);
                // }
            }
            t0._ctx = None;
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let (a, b) = Tensor::_broadcast(&self, &rhs);
        Add {
            need_input_grad: [a.require_grad, b.require_grad],
            ..Default::default()
        }
        .apply(&a, Some(&b), None, None, None)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let (a, b) = Tensor::_broadcast(&self, &rhs);
        Sub {
            need_input_grad: [a.require_grad, b.require_grad],
            ..Default::default()
        }
        .apply(&a, Some(&b), None, None, None)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let (a, b) = Tensor::_broadcast(&self, &rhs);
        Mul {
            need_input_grad: [a.require_grad, b.require_grad],
            ..Default::default()
        }
        .apply(&a, Some(&b), None, None, None)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        let (a, b) = Tensor::_broadcast(&self, &rhs);
        Div {
            need_input_grad: [a.require_grad, b.require_grad],
            ..Default::default()
        }
        .apply(&a, Some(&b), None, None, None)
    }

    pub fn assign(&mut self, x: Self) {
        assert!(self.shape() == x.shape());
        self.buffer = x.buffer;
    }

    pub fn arange(to: f32) -> Self {
        Self::_arange(0., to, 1.)
    }

    pub fn _arange(start: f32, stop: f32, step: f32) -> Self {
        // if stop is None: stop, start = start, 0
        // return Tensor.full((math.ceil((stop-start)/step),), step, **kwargs).cumsum() + (start - step)
        let s = ((stop - start) / step).ceil() as isize;
        Self::full([s], step).cumsum() + (start - step)
    }

    pub fn full<S: Into<Vec<isize>>>(shape: S, const_: impl num_traits::ToBytes) -> Self {
        let shape = shape.into();
        //panic!("in _full to_shape:{}", shape);
        Self::_const(const_)
            .reshape(vec![1; shape.len()])
            .expand(shape)
    }
    pub fn full_like(&self, const_: impl num_traits::ToBytes) -> Self {
        let shape = self.shape();
        Self::_const(const_)
            .reshape(vec![1; shape.len()])
            .expand(shape)
    }

    pub fn cumsum(&self) -> Self {
        self._cumsum(0)
    }

    //return self.transpose(axis,-1).pad2d((self.shape[axis]-1,0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
    pub fn _cumsum(&self, axis: isize) -> Self {
        let axis = if axis < 0 {
            axis + self.shape().dims.len() as isize
        } else {
            axis
        };
        self.transpose(axis, -1)
            .pad2d([self.shape()[axis] as usize - 1, 0], 0)
            ._pool([self.shape()[axis]], 1, 1)
            .sum(-1)
            .transpose(axis, -1)
    }

    // self is pred
    pub fn sparse_categorical_crossentropy(&self, y: &Self) -> Self {
        let loss_mark = y._ne(&y.full_like(-1.0));
        //y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
        let mut y_counter = Self::arange(self.shape()[-1] as f32);
        y_counter = y_counter
            .unsqueeze(0)
            .expand([y.shape().numel(), self.shape()[-1] as usize]);
        // y = ( (y_counter == Y.flatten().reshape(-1, 1)) .where(-1.0, 0) * loss_mask.reshape(-1, 1)) .reshape(*Y.shape, self.shape[-1])
        let mut y_rsh = y.shape();
        y_rsh.dims.push(self.shape()[-1]);

        let yy = (y_counter
            ._eq(&y.flatten().reshape([y.shape().numel(), 1]))
            ._where(-1., 0.)
            * loss_mark.reshape([loss_mark.shape().numel(), 1]))
        .reshape(y_rsh);
        (self.log_softmax() * yy).sum_all() / loss_mark.sum_all()
    }

    pub fn bceloss(&self, y: &Self) -> Self {
        let epsilon = 1e-9f32;
        let num_example = if self.shape().dims.len() == 0 {
            self.shape().dims[0] as f32
        } else {
            1 as f32
        };
        let entrophy =
            (self * &(y * epsilon).log() + ((1f32 - self) * (1f32 - y + epsilon).log())).sum_all();

        (-1f32 / num_example) * entrophy
    }

    pub fn clip(&self, min: f32, max: f32) -> Self {
        self.maximum(&Tensor::from([min]))
            .minimum(&Tensor::from([max]))
    }

    pub fn maximum(&self, x: &Self) -> Self {
        self._lt(x)
            .detach()
            ._where_(x, &self._gt(x).detach()._where_(self, &((self + x) / 2)))
    }
    pub fn minimum(self, x: &Self) -> Self {
        -((-self).maximum(&-x))
    }
    pub fn unsqueeze(&self, dim: isize) -> Self {
        let dim = if dim < 0 {
            (self.shape().len() as isize + dim + 1) as usize
        } else {
            dim as usize
        };
        //TODO: could just insert at position
        let mut shape: Vec<isize> = self.shape().dims[..dim].iter().map(|e| *e).collect();
        shape.push(1);
        shape.extend(self.shape().dims[dim..].iter());
        self.reshape(shape)
    }

    pub fn flatten(&self) -> Self {
        self._flatten(0)
    }
    pub fn _flatten(&self, dim: usize) -> Self {
        // self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))
        if dim == 0 {
            return self.reshape([self.shape().numel()]);
        }
        let mut new_shape: Vec<isize> = self.shape().dims[0..dim].iter().map(|e| *e).collect();
        new_shape.push(self.shape().dims[dim..].iter().product());
        self.reshape(new_shape)
    }

    pub fn _lt(&self, rhs: &Self) -> Self {
        let (a, b) = Tensor::_broadcast(&self, &rhs);
        Less::default().apply(&a, Some(&b), None, None, None)
    }

    pub fn _le(&self, rhs: &Self) -> Self {
        1.0 - (self._gt(&rhs))
    }

    pub fn _gt(&self, rhs: &Self) -> Self {
        let (a, b) = Tensor::_broadcast_r(&self, &rhs);
        Less::default().apply(&a, Some(&b), None, None, None)
    }

    pub fn _ge(&self, rhs: &Self) -> Self {
        1.0 - (self._lt(&rhs))
    }

    pub fn _eq(&self, rhs: &Self) -> Self {
        1.0 - (self._ne(rhs))
    }

    pub fn _ne(&self, rhs: &Self) -> Self {
        self._lt(rhs) + self._gt(rhs)
    }

    pub fn _where(&self, y: TensorDefaultType, z: TensorDefaultType) -> Self {
        let y = Self::_const(y);
        let z = Self::_const(z);
        self._where_(&y, &z)
    }

    pub fn _where_(&self, y: &Self, z: &Self) -> Self {
        let (x_, y_) = Self::_broadcast(self, y);
        let (x, z_) = Self::_broadcast(&x_, z);
        let (y, z) = Self::_broadcast_r(&y_, &z_);
        Where {
            need_input_grad: [self.require_grad, y.require_grad, z.require_grad],
            ..Default::default()
        }
        .apply(&x, Some(&y), Some(&z), None, None)
    }

    // def mean(self, axis=None, keepdim=False):
    //   out = self.sum(axis=axis, keepdim=keepdim)
    //   return out * (math.prod(out.shape)/math.prod(self.shape))
    pub fn mean(&self) -> Self {
        let out = self.sum_all();
        let o_numel = out.shape().numel();
        out * o_numel / self.shape().numel()
    }

    // def abs(self): return self.relu() + (-self).relu()
    pub fn abs(&self) -> Self {
        self.relu() + (-self).relu()
    }

    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    pub fn realize(&self) -> Self {
        self.buffer.realize();
        self.clone()
    }

    pub fn detach(&self) -> Self {
        Self {
            require_grad: false,
            grad: Arc::default(),
            _ctx: None,
            id: tensor_id(),
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            buffer: self.buffer.clone(),
        }
    }
}

//TODO: Tests should be in a macro so that each backend can generate test.

// #[test]
// fn sum_axis() {
//     use crate::prelude::*;
//     let n = 2 * 3;
//     let t = Tensor::<Cpu>::from_shape(
//         (1..n + 1)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [2, 3],
//     );
//     let y = t.sum(1);
//     assert!(vec![6.0f32, 15.0f32] == y.to_vec());
//
//     let n = 4 * 2 * 3 * 3;
//     let t = Tensor::<Cpu>::from_shape(
//         (1..n + 1)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [4, 2, 3, 3],
//     );
//
//     let y = t.sum(0);
//     assert!(
//         vec![
//             112., 116., 120., 124., 128., 132., 136., 140., 144., 148., 152., 156., 160., 164.,
//             168., 172., 176., 180.
//         ] == y.to_vec()
//     );
//
//     let y = t.sum(1);
//     assert!(
//         vec![
//             11., 13., 15., 17., 19., 21., 23., 25., 27., 47., 49., 51., 53., 55., 57., 59., 61.,
//             63., 83., 85., 87., 89., 91., 93., 95., 97., 99., 119., 121., 123., 125., 127., 129.,
//             131., 133., 135.
//         ] == y.to_vec()
//     );
//
//     let y = t.sum(2);
//     assert!(
//         vec![
//             12., 15., 18., 39., 42., 45., 66., 69., 72., 93., 96., 99., 120., 123., 126., 147.,
//             150., 153., 174., 177., 180., 201., 204., 207.
//         ] == y.to_vec()
//     );
//
//     let y = t.sum(3);
//     assert!(
//         vec![
//             6., 15., 24., 33., 42., 51., 60., 69., 78., 87., 96., 105., 114., 123., 132., 141.,
//             150., 159., 168., 177., 186., 195., 204., 213.
//         ] == y.to_vec()
//     );
//
//     let a = Tensor::<Cpu>::from_shape([2., 2., 2.], [3]);
//     assert!(vec![6.] == a.sum_all().to_vec())
// }
//
// #[test]
// fn matmul() {
//     let a = Tensor::<Cpu>::from_shape([1., 2., 3., 4., 5., 6.], [2, 3]);
//     let b = Tensor::<Cpu>::from_shape([10., 11., 20., 21., 30., 31.], [3, 2]);
//     let y = a.matmul(&b);
//     assert!(vec![140., 146., 320., 335.] == y.to_vec());
//
//     let a = Tensor::<Cpu>::from_shape(
//         (1..=4 * 9)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [4, 9],
//     );
//     let b = Tensor::<Cpu>::from_shape(
//         (1..=9 * 2)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [9, 2],
//     );
//     let y = a.matmul(&b);
//     assert!(vec![525., 570., 1254., 1380., 1983., 2190., 2712., 3000.] == y.to_vec())
// }
//
// #[test]
// fn pool() {
//     let n = 9;
//     let a = Tensor::<Cpu>::from_shape(
//         (1..=n * n)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [n, n],
//     );
//     let k = Vec<isize>::from([3, 3]);
//     let y = a._pool(k, 1, 1);
//     assert!(
//         vec![
//             1., 2., 3., 10., 11., 12., 19., 20., 21., 2., 3., 4., 11., 12., 13., 20., 21., 22., 3.,
//             4., 5., 12., 13., 14., 21., 22., 23., 4., 5., 6., 13., 14., 15., 22., 23., 24., 5., 6.,
//             7., 14., 15., 16., 23., 24., 25., 6., 7., 8., 15., 16., 17., 24., 25., 26., 7., 8., 9.,
//             16., 17., 18., 25., 26., 27., 10., 11., 12., 19., 20., 21., 28., 29., 30., 11., 12.,
//             13., 20., 21., 22., 29., 30., 31., 12., 13., 14., 21., 22., 23., 30., 31., 32., 13.,
//             14., 15., 22., 23., 24., 31., 32., 33., 14., 15., 16., 23., 24., 25., 32., 33., 34.,
//             15., 16., 17., 24., 25., 26., 33., 34., 35., 16., 17., 18., 25., 26., 27., 34., 35.,
//             36., 19., 20., 21., 28., 29., 30., 37., 38., 39., 20., 21., 22., 29., 30., 31., 38.,
//             39., 40., 21., 22., 23., 30., 31., 32., 39., 40., 41., 22., 23., 24., 31., 32., 33.,
//             40., 41., 42., 23., 24., 25., 32., 33., 34., 41., 42., 43., 24., 25., 26., 33., 34.,
//             35., 42., 43., 44., 25., 26., 27., 34., 35., 36., 43., 44., 45., 28., 29., 30., 37.,
//             38., 39., 46., 47., 48., 29., 30., 31., 38., 39., 40., 47., 48., 49., 30., 31., 32.,
//             39., 40., 41., 48., 49., 50., 31., 32., 33., 40., 41., 42., 49., 50., 51., 32., 33.,
//             34., 41., 42., 43., 50., 51., 52., 33., 34., 35., 42., 43., 44., 51., 52., 53., 34.,
//             35., 36., 43., 44., 45., 52., 53., 54., 37., 38., 39., 46., 47., 48., 55., 56., 57.,
//             38., 39., 40., 47., 48., 49., 56., 57., 58., 39., 40., 41., 48., 49., 50., 57., 58.,
//             59., 40., 41., 42., 49., 50., 51., 58., 59., 60., 41., 42., 43., 50., 51., 52., 59.,
//             60., 61., 42., 43., 44., 51., 52., 53., 60., 61., 62., 43., 44., 45., 52., 53., 54.,
//             61., 62., 63., 46., 47., 48., 55., 56., 57., 64., 65., 66., 47., 48., 49., 56., 57.,
//             58., 65., 66., 67., 48., 49., 50., 57., 58., 59., 66., 67., 68., 49., 50., 51., 58.,
//             59., 60., 67., 68., 69., 50., 51., 52., 59., 60., 61., 68., 69., 70., 51., 52., 53.,
//             60., 61., 62., 69., 70., 71., 52., 53., 54., 61., 62., 63., 70., 71., 72., 55., 56.,
//             57., 64., 65., 66., 73., 74., 75., 56., 57., 58., 65., 66., 67., 74., 75., 76., 57.,
//             58., 59., 66., 67., 68., 75., 76., 77., 58., 59., 60., 67., 68., 69., 76., 77., 78.,
//             59., 60., 61., 68., 69., 70., 77., 78., 79., 60., 61., 62., 69., 70., 71., 78., 79.,
//             80., 61., 62., 63., 70., 71., 72., 79., 80., 81.
//         ] == y.to_vec(),
//         "{y}"
//     );
// }
//
// #[test]
// fn conv2d() {
//     let a = Tensor::<Cpu>::from_shape(
//         (1..=9 * 9)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [1, 1, 9, 9],
//     );
//     let k = Tensor::<Cpu>::from_shape(
//         (1..=3 * 3)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [1, 1, 3, 3],
//     );
//     let r = a.conv2d(&k);
//     assert!(
//         vec![
//             663., 708., 753., 798., 843., 888., 933., 1068., 1113., 1158., 1203., 1248., 1293.,
//             1338., 1473., 1518., 1563., 1608., 1653., 1698., 1743., 1878., 1923., 1968., 2013.,
//             2058., 2103., 2148., 2283., 2328., 2373., 2418., 2463., 2508., 2553., 2688., 2733.,
//             2778., 2823., 2868., 2913., 2958., 3093., 3138., 3183., 3228., 3273., 3318., 3363.
//         ] == r.to_vec()
//     );
//
//     let (cin, cout, conv) = (3, 3, 3);
//
//     let a2 = Tensor::<Cpu>::from_shape(
//         (1..=cin * 6 * 6)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [cin, 1, 6, 6],
//     );
//     let k2 = Tensor::<Cpu>::from_shape(
//         (1..=cin * conv * conv)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [cin, 1, conv, conv],
//     );
//     let k3 = Tensor::<Cpu>::from_shape(
//         (1..=cout * cin * conv * conv)
//             .map(|e| f32::from_usize(e).unwrap())
//             .collect::<Vec<f32>>(),
//         [cout, cin, conv, conv],
//     );
//     let r = a2.conv2d(&k2).conv2d(&k3);
//
//     assert!(
//         vec![
//             997434., 1058184., 1361934., 1422684., 2458350., 2610954., 3373974., 3526578.,
//             3919266., 4163724., 5386014., 5630472., 3184434., 3245184., 3548934., 3609684.,
//             7952094., 8104698., 8867718., 9020322., 12719754., 12964212., 14186502., 14430960.,
//             5371434., 5432184., 5735934., 5796684., 13445838., 13598442., 14361462., 14514066.,
//             21520242., 21764700., 22986990., 23231448.
//         ] == r.to_vec(),
//         "{r}"
//     );
// }
//
// // macro_rules! close_to_literal {
// //     ($x:tt, $y:tt) => {
// //         x.iter().zip(y.iter()).any(|(x, y)| x / yfm)
// //     };
// // }
//
// #[test]
// fn sparse_categorical_crossentropy() {
//     let y = Tensor::<Cpu>::from_shape([1.0, 2.0], [2]);
//     let out = Tensor::<Cpu>::from_shape([0.05, 0.95, 0., 0.1, 0.8, 0.1], [6]);
//     let loss = out.sparse_categorical_crossentropy(&y);
//     approx_eq!(loss, [1.7302881]);
//     let y = Tensor::<Cpu>::from_shape([-0.0, -0.0, -1., -0.1, -0.2, -0.3], [6]);
//     let out = Tensor::<Cpu>::from_shape([-0.05, -0.95, -0., -0.1, -0.8, -0.1], [6]);
//     let loss = out.sparse_categorical_crossentropy(&y);
//     approx_eq!(loss, [0.6301593]);
// }
//
// #[test]
// fn lt() {
//     let x = Tensor::<Cpu>::from_shape([1., 2., 3., 4., 5.], [5]);
//     let y = Tensor::<Cpu>::from_shape([0., 2., 0., 4., 0.], [5]);
//     let o = x._lt(&y);
//     approx_eq!(o, [0., 0., 0., 0., 0.]);
//     let x = Tensor::<Cpu>::from_shape([1., 0., 3., 0., 5.], [5]);
//     let y = Tensor::<Cpu>::from_shape([0., 2., 0., 4., 0.], [5]);
//     let o = x._lt(&y);
//     approx_eq!(o, [0., 1., 0., 1., 0.]);
// }
//
// #[test]
// fn eq() {
//     let x = Tensor::<Cpu>::from_shape([1., 2., 3., 4., 5.], [5]);
//     let y = Tensor::<Cpu>::from_shape([0., 2., 0., 4., 0.], [5]);
//     let o = x._eq(&y);
//     approx_eq!(o, [0., 1., 0., 1., 0.]);
// }
//
// #[test]
// fn where_test() {
//     let x = Tensor::<Cpu>::from_shape([1., 2., 3., 4., 5.], [5]);
//     let y = Tensor::<Cpu>::from_shape([0., 2., 0., 4., 0.], [5]);
//     let out = x._eq(&y)._where(1.0, 2.0);
//     approx_eq!(out, [2., 1., 2., 1., 2.]);
// }
//
// #[test]
// fn test_softmax() {
//     let x = Tensor::<Cpu>::from_shape([1., 2., 3.], [3]);
//     let (m, e, ss) = x._softmax(-1);
//     approx_eq!(m, [-2., -1., 0.,]);
//     approx_eq!(e, [0.13533531, 0.36787948, 1.,]);
//     approx_eq!(ss, [1.5032148]);
// }
//
// #[test]
// fn max_test() {
//     let x = Tensor::<Cpu>::from_shape(
//         (1..=3 * 3 * 3)
//             .into_iter()
//             .map(|e| e as f32)
//             .collect::<Vec<f32>>(),
//         [3 * 3 * 3],
//     )
//     .reshape([3, 3, 3]);
//     let y = (x * -1.0f32).max_all();
//     approx_eq!(y, [-1.0]);
// }

// #[test]
// fn xor() {
//     struct Xornet {
//         l1: Tensor<Cpu>,
//         l2: Tensor<Cpu>,
//     }
//     impl Xornet {
//         pub fn new() -> Self {
//             Self {
//                 l1: Tensor::<Cpu>::scaled_uniform([2, 10]),
//                 l2: Tensor::<Cpu>::scaled_uniform([10, 1]),
//             }
//         }
//         pub fn forward(&mut self, x: &Tensor<Cpu>) -> Tensor<Cpu> {
//             let mut x = x.matmul(&self.l1).sigmoid();
//             x = x.matmul(&self.l2);
//             x
//         }
//     }
//
//     // loss = (y - out).abs().sum() / y.numel()
//     let mut model = Xornet::new();
//     let mut optim = adam(vec![&mut model.l1, &mut model.l2], 0.1);
//     let x = Tensor::<Cpu>::from_vec([0., 0., 0., 1., 1., 0., 1., 1.], [4, 2]);
//     let y = Tensor::<Cpu>::from_vec([0., 1., 1., 0.], [1, 4]);
//     for _ in 0..100 {
//         let out = model.forward(&x);
//         //let mut loss = (&out - &y).abs().sum_all() / y.numel();
//         let mut loss = &out - &y;
//         loss = (&loss * &loss).mean();
//         optim.zero_grad();
//         println!("loss {:?}", loss.to_vec());
//         loss.backward();
//         optim.step();
//     }
//
//     // let t = Tensor::<Cpu>::from_vec([0., 0.], [2]);
//     // let y = Tensor::<Cpu>::from_vec([0.], [1]);
//     // //println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);
//     //
//     // let t = Tensor::<Cpu>::from_vec([1., 0.], [2]);
//     // let y = Tensor::<Cpu>::from_vec([1.], [1]);
//     // //println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);
//     //
//     // let t = Tensor::<Cpu>::from_vec([0., 1.], [2]);
//     // let y = Tensor::<Cpu>::from_vec([1.], [1]);
//     // //println!("Expected: 1 | Got: {}", model.forward(&t).to_vec()[0]);
//     //
//     // let t = Tensor::<Cpu>::from_vec([1., 1.], [2]);
//     // let y = Tensor::<Cpu>::from_vec([0.], [1]);
//     //println!("Expected: 0 | Got: {}", model.forward(&t).to_vec()[0]);
// }
