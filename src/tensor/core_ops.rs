use std::{ops::RangeBounds, sync::Arc};

use crate::prelude::*;

use self::dtype::NumType;

impl core::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{:?}Shape:{:?} Stride:{:?} Dtype:{} Device:{} Id:{:?} require_grad:{} grad:{:?}\nctx:{:?}\n",
            self.buffer,
            self.buffer.shape,
            self.buffer.st.strides(),
            self.dtype(),
            self.device(),
            self.id.0,
            self.require_grad,
            self.grad,
            self._ctx,
        )
    }
}

impl core::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{:?}Shape:{:?} Stride:{:?} Dtype:{} Device:{}\n",
            self.buffer,
            self.buffer.shape,
            self.buffer.st.strides(),
            self.dtype(),
            self.device(),
        )
    }
}

macro_rules! core_impl {
    ($op:tt, $fn:tt) => {
        impl core::ops::$op for Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: Self) -> Self::Output {
                Tensor::$fn(&self, &rhs)
            }
        }

        impl core::ops::$op<&Tensor> for Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: &Tensor) -> Self::Output {
                Tensor::$fn(&self, rhs)
            }
        }

        impl core::ops::$op for &Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: Self) -> Self::Output {
                Tensor::$fn(self, rhs)
            }
        }
    };
}

core_impl!(Add, add);
core_impl!(Sub, sub);
core_impl!(Mul, mul);
core_impl!(Div, div);

macro_rules! core_impl_num {
    ($op:tt, $fn:tt, $t:ty, $from:ident) => {
        impl core::ops::$op<$t> for Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: $t) -> Self::Output {
                let rhs = Tensor::_const(rhs);
                Tensor::$fn(&self, &rhs)
            }
        }

        impl core::ops::$op<$t> for &Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: $t) -> Self::Output {
                let rhs = Tensor::_const(rhs);
                Tensor::$fn(self, &rhs)
            }
        }

        impl core::ops::$op<&$t> for Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: &$t) -> Self::Output {
                let rhs = Tensor::_const(*rhs);
                Tensor::$fn(&self, &rhs)
            }
        }

        impl core::ops::$op<&$t> for &Tensor {
            type Output = Tensor;
            fn $fn(self, rhs: &$t) -> Self::Output {
                let rhs = Tensor::_const(*rhs);
                Tensor::$fn(&self, &rhs)
            }
        }

        impl core::ops::$op<Tensor> for $t {
            type Output = Tensor;
            fn $fn(self, rhs: Tensor) -> Self::Output {
                let lhs = Tensor::_const(self);
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl core::ops::$op<Tensor> for &$t {
            type Output = Tensor;
            fn $fn(self, rhs: Tensor) -> Self::Output {
                let lhs = Tensor::_const(*self);
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl core::ops::$op<&Tensor> for $t {
            type Output = Tensor;
            fn $fn(self, rhs: &Tensor) -> Self::Output {
                let lhs = Tensor::_const(self);
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl core::ops::$op<&Tensor> for &$t {
            type Output = Tensor;
            fn $fn(self, rhs: &Tensor) -> Self::Output {
                let lhs = Tensor::_const(*self);
                Tensor::$fn(&lhs, rhs)
            }
        }
    };
}

core_impl_num!(Add, add, f32, from_f32);
core_impl_num!(Sub, sub, f32, from_f32);
core_impl_num!(Mul, mul, f32, from_f32);
core_impl_num!(Div, div, f32, from_f32);

core_impl_num!(Add, add, isize, from_isize);
core_impl_num!(Sub, sub, isize, from_isize);
core_impl_num!(Mul, mul, isize, from_isize);
core_impl_num!(Div, div, isize, from_isize);

impl core::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl core::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl From<f32> for Tensor {
    fn from(value: f32) -> Self {
        Self::_const(value)
    }
}

impl From<isize> for Tensor {
    fn from(value: isize) -> Self {
        Self::_const(value)
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

lazy_static::lazy_static! {
    pub static ref Tensors: Arc<Vec<Tensor>> = Default::default();
}

impl<N: NumType> std::ops::Index<N> for Tensor {
    type Output = Self;

    fn index(&self, index: N) -> &Self::Output {
        let mut index = index.to_isize().unwrap();
        if index < 0 {
            index = self.shape().len() as isize + index;
        };
        let index = index as usize;
        assert!(index < self.shape()[0] as usize);
        let mut slc = v![(0, *s as usize), for (i, s) in self.shape().dims.iter().enumerate()];
        slc[0] = (index, index+1);
        let mut new_tenor = self.shrink(slc);
        if self.shape().len() > 1 {
            let mut new_shape = self.shape().dims.clone();
            new_shape.remove(index);
            new_tenor = new_tenor.reshape(new_shape);
        }
        unsafe {
            let mut tensors = Tensors.clone();
            Arc::get_mut_unchecked(&mut tensors).push(new_tenor);
        }
        Tensors.last().as_ref().unwrap()
    }
}

impl<N: NumType> std::ops::Index<std::ops::Range<N>> for Tensor {
    type Output = Self;
    fn index(&self, range: std::ops::Range<N>) -> &Self::Output {
        let mut s = range.start.to_isize().unwrap();
        if s < 0 {
            s += self.shape().dims[0];
        }
        let mut e = range.end.to_isize().unwrap();
        if e < 0 {
            e += self.shape().dims[0];
        }
        match range.end_bound() {
            std::ops::Bound::Included(_) => e += 1,
            std::ops::Bound::Unbounded => e = self.shape().dims[0],
            _ => (),
        };
        let mut slc = v![(0, *s as usize), for (i, s) in self.shape().dims.iter().enumerate()];
        slc[0] = (s as usize, e as usize);
        let mut new_tenor = self.shrink(slc);
        unsafe {
            let mut tensors = Tensors.clone();
            Arc::get_mut_unchecked(&mut tensors).push(new_tenor);
        }
        Tensors.last().as_ref().unwrap()
    }
}

impl<N: NumType> std::ops::Index<std::ops::RangeTo<N>> for Tensor {
    type Output = Self;
    fn index(&self, range: std::ops::RangeTo<N>) -> &Self::Output {
        let mut s = match range.start_bound() {
            std::ops::Bound::Included(n) | std::ops::Bound::Excluded(n) => n.to_isize().unwrap(),
            std::ops::Bound::Unbounded => 0,
        };
        if s < 0 {
            s += self.shape().dims[0];
        }
        if matches!(range.start_bound(), std::ops::Bound::Excluded(_)) {
            s += 1;
        }
        let mut e = match range.end_bound() {
            std::ops::Bound::Included(n) | std::ops::Bound::Excluded(n) => n.to_isize().unwrap(),
            std::ops::Bound::Unbounded => self.shape().dims[0],
        };
        if e < 0 {
            e += self.shape().dims[0];
        }
        if matches!(range.start_bound(), std::ops::Bound::Included(_)) {
            e += 1;
        }
        let mut slc = v![(0, *s as usize), for (i, s) in self.shape().dims.iter().enumerate()];
        slc[0] = (s as usize, e as usize);
        let mut new_tenor = self.shrink(slc);
        unsafe {
            let mut tensors = Tensors.clone();
            Arc::get_mut_unchecked(&mut tensors).push(new_tenor);
        }
        Tensors.last().as_ref().unwrap()
    }
}

impl<N: NumType> std::ops::Index<std::ops::RangeInclusive<N>> for Tensor {
    type Output = Self;
    fn index(&self, range: std::ops::RangeInclusive<N>) -> &Self::Output {
        let mut s = match range.start_bound() {
            std::ops::Bound::Included(n) | std::ops::Bound::Excluded(n) => n.to_isize().unwrap(),
            std::ops::Bound::Unbounded => 0,
        };
        if s < 0 {
            s += self.shape().dims[0];
        }
        if matches!(range.start_bound(), std::ops::Bound::Excluded(_)) {
            s += 1;
        }
        let mut e = match range.end_bound() {
            std::ops::Bound::Included(n) | std::ops::Bound::Excluded(n) => n.to_isize().unwrap(),
            std::ops::Bound::Unbounded => self.shape().dims[0],
        };
        if e < 0 {
            e += self.shape().dims[0];
        }
        if matches!(range.start_bound(), std::ops::Bound::Included(_)) {
            e += 1;
        }
        let mut slc = v![(0, *s as usize), for (i, s) in self.shape().dims.iter().enumerate()];
        slc[0] = (s as usize, e as usize);
        let mut new_tenor = self.shrink(slc);
        unsafe {
            let mut tensors = Tensors.clone();
            Arc::get_mut_unchecked(&mut tensors).push(new_tenor);
        }
        Tensors.last().as_ref().unwrap()
    }
}

impl std::ops::Index<std::ops::RangeFull> for Tensor {
    type Output = Self;
    fn index(&self, range: std::ops::RangeFull) -> &Self::Output {
        let s = 0;
        let e = self.shape().dims[0];
        let mut slc = v![(0, *s as usize), for (i, s) in self.shape().dims.iter().enumerate()];
        slc[0] = (s as usize, e as usize);
        let mut new_tenor = self.shrink(slc);
        unsafe {
            let mut tensors = Tensors.clone();
            Arc::get_mut_unchecked(&mut tensors).push(new_tenor);
        }
        Tensors.last().as_ref().unwrap()
    }
}
