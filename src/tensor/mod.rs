pub mod core_ops;
pub mod mlops;
pub mod shape;
use half::f16;
use num_traits::One;
use num_traits::Zero;

use crate::arg::Arg;
use crate::dtype::type_to_dtype;
use crate::dtype::NumType;
use crate::lazy::run_schedule;
use crate::ops::LazyOp;
use crate::ops::Load;
use crate::ops::OpType;
use crate::prelude::*;
use crate::prelude::*;
use crate::tensor::mlops::*;
use crate::tensor::shape::Shape;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::ops::Neg;
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
            buffer: buf.into(),
        }
    }

    pub fn from<V: Into<Vec<TensorDefaultType>>>(data: V) -> Self {
        let data = data.into();
        let buffer = if data.len() == 1 {
            LazyBuffer::_const(data[0], dtype::type_to_dtype::<TensorDefaultType>(), "GPU")
        } else {
            LazyBuffer::from_cpu(data)
        };
        Self {
            require_grad: false,
            _ctx: None,
            id: tensor_id(),
            grad: Arc::new(Mutex::new(None)),
            dtype: buffer.dtype.clone(),
            device: buffer.device.clone(),
            buffer: buffer.into(),
        }
    }

    pub fn contiguous(&self) -> Self {
        Contiguous::default().apply(self, None, None, None, None)
    }

    // TODO: This is probably stuck with generic param.
    //      Or have three verions of this. 1. T, 2. String, 3. Raw bytes
    pub fn to_vec(&self) -> Vec<TensorDefaultType> {
        type T = TensorDefaultType;
        assert!(
            std::any::type_name::<T>().split("::").last().unwrap() == self.dtype(),
            "cannot return Tensor<{}> to Vec<{}>",
            self.dtype(),
            std::any::type_name::<T>().split("::").last().unwrap()
        );
        let buffer = self.realize();
        let mut bytes = (*buffer.buffer.device_buffer)
            .as_ref()
            .expect("buffer not realized")
            .to_cpu();
        let mut ret = vec![];
        for b in bytes
            .windows(std::mem::size_of::<T>())
            .step_by(std::mem::size_of::<T>())
        {
            ret.push(T::from_le_bytes(b.try_into().unwrap()))
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

    pub fn _const(value: impl Display) -> Self {
        let dtype = crate::dtype::name_to_dtype(std::any::type_name::<TensorDefaultType>());
        Self::from_buf(LazyBuffer::_const(value, dtype, "GPU"))
    }

    pub fn const_like<T: NumType>(&self, const_value: T) -> Self {
        Self::_const(const_value)
            .reshape(vec![1; self.shape().len()])
            .expand(self.shape().dims)
    }

    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::_const(0)
            .reshape(vec![1; shape.len()])
            .expand(shape.dims)
    }

    pub fn ones<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::_const(1)
            .reshape(vec![1; shape.len()])
            .expand(shape.dims)
    }

    pub fn rand<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::from_buf(LazyBuffer::new(
            "GPU",
            crate::shape::ShapeTracker::from_shape(&shape.dims),
            OpType::Load(Load::Rand),
            LazyOp::new(OpType::Load(Load::Rand), vec![], None),
            type_to_dtype::<TensorDefaultType>(),
            None,
        ))
        // Self::_load(
        //     OpType::Load(Load::Rand),
        //     shape.numel(),
        //     type_to_dtype::<TensorDefaultType>(),
        //     None,
        //     None,
        // )
        // .reshape(shape)
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
        let shape = shape.into().dims;
        let numel = self.shape().numel() as isize;
        Reshape::default().apply(self, None, None, Some(v![if s == -1 { -numel / shape.iter().product::<isize>()} else { s }, for (i, &s) in shape.iter().enumerate()]), None)
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

    pub fn pad<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: impl NumType) -> Self {
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
            Some(const_value._to_le_bytes()),
        )
        // Tensor {
        //     inner: self.inner.pad(arg, const_value),
        //     require_grad: false,
        //     _ctx: None,
        //     id: tensor_id(),
        //     grad: None,
        // }
    }

    pub fn pad2d<P: Into<Vec<usize>>>(&self, padding: P, const_value: impl NumType) -> Self {
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
        let rl = (self.shape().len() as isize - (padding.len() / 2) as isize);
        let r = if rl < 0 {
            if rl < -(self.shape().len() as isize) {
                0
            } else {
                (self.shape().len() as isize + rl) as usize
            }
        } else {
            if rl > self.shape().len() as isize {
                self.shape().len()
            } else {
                rl as usize
            }
        };
        let mut slice_shape: Vec<(isize, isize)> = self.shape().dims[..r as usize]
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
        ((core::f32::consts::PI / 2.0) - self).sin()
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

    pub fn _sum(&self, axis: Vec<isize>, keepdim: bool) -> Self {
        self._reduce(Sum::default(), axis, keepdim)
    }

    pub fn sum_keepdim(&self, axis: isize) -> Self {
        self._reduce(Sum::default(), [axis].to_vec(), true)
    }

    pub fn sum(&self, axis: isize) -> Self {
        self._reduce(Sum::default(), [axis].to_vec(), false)
    }

    pub fn sum_all(&self) -> Self {
        self._reduce(
            Sum::default(),
            v![i as isize, for i in 0..self.shape().len()],
            false,
        )
    }

    pub fn _reduce<F: 'static + Function>(
        &self,
        mut fxn: F,
        axis: Vec<isize>,
        keepdim: bool,
    ) -> Self {
        let axis_ = if axis.len() > 0 {
            axis
        } else {
            (0..self.shape().len()).map(|i| i as isize).collect()
        };
        let axis_ = v![if x >= 0 { x } else { x + self.shape().len() as isize }, for x in axis_];
        let shape = v![s, for (i, s) in self.shape().dims.iter().enumerate(), if !axis_.contains(&(i as isize))];
        if self.shape().dims.contains(&0) && !shape.contains(&0) {
            let fxn_name = std::any::type_name::<F>().split("::").last().unwrap();
            return Self::full(
                v![if s == 0 { 1 } else { s }, for s in self.shape().dims],
                if fxn_name == "Sum" {
                    0.0
                } else {
                    -f32::INFINITY
                },
            );
        }
        let new_shape = v![if axis_.contains(&(i as isize)) { 1 } else { *s }, for (i, s) in self.shape().dims.iter().enumerate()];
        assert!(!new_shape.is_empty());
        let ret = fxn.apply(self, None, None, Some(new_shape), None);
        if keepdim {
            ret
        } else {
            if shape.len() == 0 {
                ret.reshape([1])
            } else {
                ret.reshape(shape)
            }
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn _max(&self, axis: &[isize], keepdim: bool) -> Self {
        self._reduce(Max::default(), axis.to_vec(), keepdim)
    }

    pub fn max(&self, axis: isize) -> Self {
        self._reduce(Max::default(), [axis].to_vec(), false)
    }

    pub fn max_keepdim(&self, axis: isize) -> Self {
        self._reduce(Max::default(), [axis].to_vec(), true)
    }

    pub fn max_all(&self) -> Self {
        self._reduce(
            Max::default(),
            v![i as isize, for i in 0..self.shape().len()],
            false,
        )
    }

    pub fn _argmax(&self, axis: Option<isize>, keepdim: bool) -> Self {
        if axis.is_none() {
            let idx = (self._eq(&self.max_all()))
                * Self::_arange(self.numel() as f32 - 1., -1., -1.).reshape(self.shape());
            return self.numel() as isize - idx.max_all() - 1;
        }
        let mut axis = axis.unwrap();
        axis = axis + if axis < 0 { self.ndim() as isize } else { axis };
        let m = self._eq(&self.max_keepdim(axis));
        let idx = m * Self::_arange(self.shape()[axis as usize] as f32 - 1., -1., -1.).reshape(
            [
                vec![self.shape()[axis]],
                vec![1; self.ndim() - axis as usize - 1],
            ]
            .concat(),
        );
        if keepdim {
            return self.shape()[axis] - idx.max_keepdim(axis) - 1;
        }
        self.shape()[axis] - idx.max(axis) - 1
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
        if n2 >= 2 {
            w_reshape.extend_from_slice(&w.shape().dims[0..n2 - 2]);
        }
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
        self._max_pool2d(None, None, None)
    }

    pub fn _max_pool2d(
        &self,
        kernel_size: Option<usize>,
        stride: Option<usize>,
        dilation: Option<usize>,
    ) -> Self {
        let kernel_size = kernel_size.unwrap_or(2);
        let stride = stride.unwrap_or(2);
        let dilation = dilation.unwrap_or(1);
        self._pool([kernel_size, kernel_size], stride, dilation)
            ._max(&v![i, for i in (-2..0)], false)
    }


    #[rustfmt::skip]
    pub fn _pool<S: Into<Shape>>(&self, k_: S, stride: usize, dilation: usize) -> Self {
        let self_shape = self.shape();
        let k_ = k_.into().dims.iter().map(|n| *n).collect::<Vec<isize>>();
        let d_ = vec![dilation as isize;k_.len()];
        let s_ = vec![stride as isize;k_.len()];
        assert!(self_shape.len() >= k_.len(), "can't pool {self_shape:?} with {k_:?}");
        assert!(k_.len() == s_.len() && s_.len() == d_.len(), "stride/dilation mismatch kernel:{k_:?} stride:{s_:?} dilation:{d_:?}");
        let slc_prefix: Vec<(isize, isize)> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| (0, *sh))
            .collect();
        let prefix: Vec<isize> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| *sh)
            .collect();
        let i_: Vec<isize> = self_shape.dims[self_shape.len() - k_.len()..]
            .iter()
            .map(|sh| *sh)
            .collect();
        let xup = if k_.iter().zip(s_.iter()).any(|(k, s)| k > s) || d_.iter().any(|d| *d != 1) {
            let o_ = v![(i - d * (k - 1) - 1) / s + 1, for (i, d, k, s) in izip!(i_.iter(), d_.iter(), k_.iter(), s_.iter())];
            let e_: Vec<isize> = k_
                .iter()
                .zip(i_.iter().zip(d_.iter()))
                .map(|(k, (i, d))| f32::ceil((k * (i + d)) as f32 / *i as f32) as isize)
                .collect();
            self.reshape([prefix.clone(), i_.iter().flat_map(|i| [1, *i]).collect()].concat())
                .expand([prefix.clone(), e_.iter().zip(i_.iter()).flat_map(|(e, i)| [*e, *i]).collect()].concat())
                .reshape([prefix.clone(), e_.iter().zip(i_.iter()).map(|(e, i)| *e * *i).collect()].concat())
                .slice(vec![slc_prefix.clone(), v![(0, k*(i+d)), for (k, i, d) in izip!(k_.iter(), i_.iter(), d_.iter())]].concat(), 0)
                .reshape(vec![prefix.clone(), v![[*k, i + d], for (k, i, d) in izip!(k_.iter(), i_.iter(), d_.iter())].concat()].concat())
                .slice(vec![slc_prefix.clone(), v![[(0, *k),(0, o * s)], for (k, o, s) in izip!(k_.iter(), o_.iter(), s_.iter())].concat()].concat(), 0)
                .reshape(vec![prefix.clone(), v![[k, o, s], for (&k, &o, &s) in izip!(k_.iter(), o_.iter(), s_.iter())].concat()].concat())
                .slice(vec![slc_prefix.clone(), v![[(0, *k), (0, *o), (0, 1)], for (k, o) in izip!(k_.iter(), o_.iter())].concat()].concat(), 0)
                .reshape(vec![prefix.clone(), v![[k, o], for (&k, &o) in izip!(k_.iter(), o_.iter())].concat()].concat())
                .permute(vec![v![i, for i in 0..prefix.len()], v![prefix.len() + i * 2 + 1, for i in 0..k_.len()], v![prefix.len() + i * 2, for i in 0..k_.len()]].concat())
        } else {
            let o_ = v![(i+(s-k))/s, for (i, s, k) in izip!(i_.iter(), s_.iter(), k_.iter())];
            let mut xup = self.slice(vec![slc_prefix.clone(), v![(0, o * s), for (&o, &s) in izip!(o_.iter(), s_.iter())]].concat(), 0);
            xup = xup.reshape(vec![prefix.clone(), v![[o, s], for (&o, &s) in izip!(o_.iter(), s_.iter())].concat()].concat());
            xup = xup.slice(vec![slc_prefix.clone(), v![[(0, o), (0, k)], for (&o, &k) in izip!(o_.iter(), k_.iter())].concat()].concat(), 0);

            let mut tmp: Vec<usize> = (0..prefix.len()).into_iter().collect();
            tmp.extend((0..k_.len()).map(|i| prefix.len() + i * 2));
            tmp.extend((0..k_.len()).map(|i| prefix.len() + i * 2 + 1));
            xup.permute(tmp)
        };
        xup
    }

    pub fn slice<A: Into<Vec<(isize, isize)>>>(&self, arg: A, const_value: impl NumType) -> Self {
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

    #[rustfmt::skip]
    pub fn _conv2d<V: Into<Vec<usize>>>(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        groups: usize,
        stride: usize,
        dilation: usize,
        padding: V,
    ) -> Self {
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
      //x = x.reshape(bs, groups, cin, 1, *oyx, *HW)
      //     .expand(bs, groups, cin, rcout, *oyx, *HW)
      //     .permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501
        x = x.reshape(vec![vec![bs, groups,cin, 1], oyx.clone(), hw.clone()].concat())
             .expand(vec![vec![bs, groups, cin, rcout], oyx.clone(), hw.clone()].concat())
             .permute(vec![vec![0,1,3], v![4+i, for i in 0..oyx.len()], vec![2], v![4+oyx.len()+i, for i in 0..hw.len()]].concat());
        // ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx)
        x = (x*weight.reshape(vec![vec![1, groups, rcout], vec![1;oyx.len()], vec![cin], hw.clone()].concat()))._sum(v![-1-i as isize, for i in 0..1+oyx.len()], true).reshape(vec![vec![bs, cout], oyx.clone()].concat());

        if let Some(bias) = bias {
            // bias.reshape(1, -1, *[1] * len(HW))
            x + bias.reshape(vec![vec![1,-1], vec![1;hw.len()]].concat())
        } else {
            x
        }
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
        (*self.grad.lock().unwrap()) = Some(Tensor::_const(1f32));
        let deepwalked = self.deepwalk().into_iter();
        let _deepwalked_len = deepwalked.len();
        for mut t0 in deepwalked.rev() {
            let t0g_clone = (*t0.grad.lock().unwrap())
                .as_ref()
                .expect("t0 should have a grad")
                .buffer
                .clone();
            let grads = match t0._ctx.as_mut().unwrap().backward(&t0g_clone) {
                Grad::One(g) => vec![Some(Tensor {
                    dtype: g.dtype.clone(),
                    device: g.device.clone(),
                    buffer: g.into(),
                    require_grad: false,
                    grad: Arc::default(),
                    _ctx: None,
                    id: tensor_id(),
                })],
                Grad::Two(mut g1, mut g2) => {
                    let mut out = vec![];
                    out.push(if let Some(g) = g1.take() {
                        // if g.to_vec().iter().any(|n| n.is_nan()) {
                        //     panic!("g has NaN")
                        // } else {
                        //     println!("{:?}", g);
                        // }
                        Some(Tensor {
                            dtype: g.dtype.clone(),
                            device: g.device.clone(),
                            buffer: g.into(),
                            require_grad: false,
                            grad: Arc::default(),
                            _ctx: None,
                            id: tensor_id(),
                        })
                    } else {
                        None
                    });
                    out.push(if let Some(g) = g2.take() {
                        // if g.to_vec().iter().any(|n| n.is_nan()) {
                        //     panic!("g has NaN")
                        // } else {
                        //     println!("{:?}", g);
                        // }
                        Some(Tensor {
                            dtype: g.dtype.clone(),
                            device: g.device.clone(),
                            buffer: g.into(),
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
                    "grad shape must match tensor shape, {} != {} \n {} {t}\n {} {g}",
                    g.shape(),
                    t.shape(),
                    t.buffer.id,
                    g.buffer.id,
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
        assert!(Arc::ptr_eq(
            &self.buffer.device_buffer,
            &a.buffer.device_buffer
        ));
        assert!(Arc::ptr_eq(
            &rhs.buffer.device_buffer,
            &b.buffer.device_buffer
        ));
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

    pub fn assign(&mut self, x: Self) -> Self {
        assert!(self.shape() == x.shape());
        self.buffer = x.buffer;
        self.clone()
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

    pub fn full<S: Into<Vec<isize>>>(shape: S, const_: impl NumType) -> Self {
        let shape = shape.into();
        //panic!("in _full to_shape:{}", shape);
        Self::_const(const_)
            .reshape(vec![1; shape.len()])
            .expand(shape)
    }
    pub fn full_like(&self, const_: impl NumType) -> Self {
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
            ._where(-1.0, 0.0)
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

    pub fn _where<N: NumType>(&self, y: N, z: N) -> Self {
        let y = Self::_const(y);
        let z = Self::_const(z);
        self._where_(&y, &z)
    }

    pub fn _where_(&self, z: &Self, y: &Self) -> Self {
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
        if self.shape().dims.contains(&0) {
            return out;
        }
        out * (1f32 / self.numel() as f32)
    }

    // def abs(self): return self.relu() + (-self).relu()
    pub fn abs(&self) -> Self {
        self.relu() + (-self).relu()
    }

    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    pub fn realize(&self) -> Self {
        let mut seen = HashSet::new();
        let mut ret = self.clone();
        if matches!(ret.buffer.lazyop.optype, OpType::Movement(_)) {
            ret = ret + 0;
        }
        run_schedule(ret.buffer.schedule(&mut seen));
        ret.buffer.lazyop.src.clear();
        ret.buffer.lazyop.buffers.clear();
        ret
    }

    pub fn corealize(list: Vec<Tensor>) {
        let mut seen = HashSet::new();
        let mut sched = std::collections::VecDeque::new();
        for t in list {
            sched.extend(t.buffer.schedule(&mut seen));
        }
        run_schedule(sched);
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

    pub fn pow<V: Into<Self>>(&self, x: V, reverse: bool) -> Self {
        let x: Tensor = x.into();
        if x.is_const() && !reverse {
            let cv = x.get_const_val().unwrap();
            if cv < 0. {
                return self.reciprocal().pow(-x, false);
            }
            if [0., 1., 2., 3.].contains(&cv) {
                let mut acc = self.const_like(1);
                for i in 0..cv as usize {
                    acc = &acc * self;
                }
                return acc;
            }
            if cv == 0.5 {
                return self.sqrt();
            }
        }
        let ar = self.abs().log().mul(&x).exp();
        let sign = if !reverse {
            (&x * std::f32::consts::PI).cos()
        } else {
            (self * std::f32::consts::PI).cos()
        };
        let mut base_sign = if !reverse { self.sign() } else { x.sign() };
        if !reverse {
            base_sign = base_sign - (1.5 * (1 - self.sign().abs()));
        } else {
            base_sign = base_sign - (1.5 * (1 - x.sign().abs()));
        }
        base_sign = (&base_sign - 1) / -2;
        ar.mul(&(sign * &base_sign + (1 - &base_sign)))
        //ar.mul(&(sign * &base_sign))
    }

    pub fn sign(&self) -> Self {
        self / &(self.abs() + 1e-12)
    }

    pub fn reciprocal(&self) -> Self {
        1.0 / self
    }

    pub fn is_const(&self) -> bool {
        if self.buffer.lazyop.optype == Load::Const {
            return true;
        }
        false
    }

    pub fn get_const_val(&self) -> anyhow::Result<TensorDefaultType> {
        if !self.is_const() {
            return Err(anyhow::anyhow!("Tensor is not const"));
        }
        Ok(self.buffer.lazyop.args[0]
            .to_str()
            .parse::<TensorDefaultType>()?)
    }
}
