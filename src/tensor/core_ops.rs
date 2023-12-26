use crate::prelude::*;

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
