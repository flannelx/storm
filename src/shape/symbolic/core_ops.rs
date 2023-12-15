use super::{num, Node, ArcNode};

macro_rules! impl_core {
    ($op: tt, $op_fn: tt, $fn: ident) => {
        impl core::ops::$op for ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                self.$fn(rhs.clone())
            }
        }

        impl core::ops::$op<&ArcNode> for ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: &ArcNode) -> Self::Output {
                self.$fn(rhs.clone())
            }
        }

        impl core::ops::$op for &ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                self.$fn(rhs.clone())
            }
        }

        impl core::ops::$op<ArcNode> for &ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: ArcNode) -> Self::Output {
                self.$fn(rhs.clone())
            }
        }

        impl core::ops::$op<isize> for ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: isize) -> Self::Output {
                self.$fn(num(rhs))
            }
        }

        impl core::ops::$op<isize> for &ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: isize) -> Self::Output {
                self.$fn(num(rhs))
            }
        }

        impl core::ops::$op<ArcNode> for isize {
            type Output = ArcNode;

            fn $op_fn(self, rhs: ArcNode) -> Self::Output {
                num(self).$fn(rhs)
            }
        }
    };

    ($op: tt, $op_fn: tt, $fn: ident, $opt: ident) => {
        impl core::ops::$op for ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                self.$fn(rhs.clone(), $opt)
            }
        }

        impl core::ops::$op<&ArcNode> for ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: &ArcNode) -> Self::Output {
                self.$fn(rhs.clone(), $opt)
            }
        }

        impl core::ops::$op for &ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                self.$fn(rhs.clone(), $opt)
            }
        }

        impl core::ops::$op<isize> for ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: isize) -> Self::Output {
                self.$fn(num(rhs), $opt)
            }
        }

        impl core::ops::$op<isize> for &ArcNode {
            type Output = ArcNode;

            fn $op_fn(self, rhs: isize) -> Self::Output {
                self.$fn(num(rhs), $opt)
            }
        }

        impl core::ops::$op<ArcNode> for isize {
            type Output = ArcNode;

            fn $op_fn(self, rhs: ArcNode) -> Self::Output {
                num(self).$fn(rhs, $opt)
            }
        }
    }
}

impl_core!(Add, add, _add);
impl_core!(Sub, sub, _sub);
impl_core!(Mul, mul, _mul);
impl_core!(Div, div, _div, None);
impl_core!(Rem, rem, _mod);

impl core::ops::Neg for ArcNode {
    type Output = ArcNode;

    fn neg(self) -> Self::Output {
        self * num(-1)
    }
}

impl core::ops::Neg for &ArcNode {
    type Output = ArcNode;

    fn neg(self) -> Self::Output {
        self * num(-1)
    }
}

impl core::fmt::Display for ArcNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key())
    }
}

// impl core::fmt::Debug for ArcNode {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.render(Arc::new(CStyle), Some("DEBUG"), false))
//     }
// }

impl PartialEq for dyn Node {
    fn eq(&self, other: &Self) -> bool {
        if self.is_num() && other.is_num() {
            return self.num_val().unwrap() == other.num_val().unwrap();
        }
        self.key() == other.key()
    }
}

impl Eq for dyn Node {}

impl std::hash::Hash for dyn Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

