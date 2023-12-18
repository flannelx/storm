use crate::{dtype::NumType, prelude::*};

impl<T: NumType> core::fmt::Debug for Tensor<T> {
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

impl<T: NumType> core::fmt::Display for Tensor<T> {
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
        impl<T: NumType> core::ops::$op for Tensor<T> {
            type Output = Tensor<T>;
            fn $fn(self, rhs: Self) -> Self::Output {
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &Tensor<T>) -> Self::Output {
                Tensor::$fn(&self, rhs)
            }
        }

        impl<T: NumType> core::ops::$op for &Tensor<T> {
            type Output = Tensor<T>;
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
        impl<T: NumType> core::ops::$op<$t> for Tensor<T> {
            type Output = Tensor<T>;
            fn $fn(self, rhs: $t) -> Self::Output {
                let rhs = Tensor::_const(T::$from(rhs).unwrap());
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<$t> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $fn(self, rhs: $t) -> Self::Output {
                let rhs = Tensor::_const(T::$from(rhs).unwrap());
                Tensor::$fn(self, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<&$t> for Tensor<T> {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &$t) -> Self::Output {
                let rhs = Tensor::_const(T::$from(*rhs).unwrap());
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<&$t> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &$t) -> Self::Output {
                let rhs = Tensor::_const(T::$from(*rhs).unwrap());
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<Tensor<T>> for $t {
            type Output = Tensor<T>;
            fn $fn(self, rhs: Tensor<T>) -> Self::Output {
                let lhs = Tensor::_const(T::$from(self).unwrap());
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<Tensor<T>> for &$t {
            type Output = Tensor<T>;
            fn $fn(self, rhs: Tensor<T>) -> Self::Output {
                let lhs = Tensor::_const(T::$from(*self).unwrap());
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<&Tensor<T>> for $t {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &Tensor<T>) -> Self::Output {
                let lhs = Tensor::_const(T::$from(self).unwrap());
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl<T: NumType> core::ops::$op<&Tensor<T>> for &$t {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &Tensor<T>) -> Self::Output {
                let lhs = Tensor::_const(T::$from(*self).unwrap());
                Tensor::$fn(&lhs, rhs)
            }
        }
    };
}

core_impl_num!(Add, add, f32, from_f32);
core_impl_num!(Sub, sub, f32, from_f32);
core_impl_num!(Mul, mul, f32, from_f32);
core_impl_num!(Div, div, f32, from_f32);

core_impl_num!(Add, add, i32, from_i32);
core_impl_num!(Sub, sub, i32, from_i32);
core_impl_num!(Mul, mul, i32, from_i32);
core_impl_num!(Div, div, i32, from_i32);

core_impl_num!(Add, add, usize, from_usize);
core_impl_num!(Sub, sub, usize, from_usize);
core_impl_num!(Mul, mul, usize, from_usize);
core_impl_num!(Div, div, usize, from_usize);

core_impl_num!(Add, add, isize, from_isize);
core_impl_num!(Sub, sub, isize, from_isize);
core_impl_num!(Mul, mul, isize, from_isize);
core_impl_num!(Div, div, isize, from_isize);

impl<T: NumType> core::ops::Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl<T: NumType> core::ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}
