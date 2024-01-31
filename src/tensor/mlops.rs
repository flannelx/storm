#![allow(unused_variables, dead_code)]

use crate::{
    ops::{Binary, Reduce, Ternary, Unary},
    prelude::*,
    tensor::TensorId,
};
use dyn_clone::DynClone;

pub fn argsort<V: Into<Vec<isize>>>(shape: V) -> Vec<isize> {
    let shape = shape.into();
    let mut out = (0..shape.len()).into_iter().collect::<Vec<_>>();
    out.sort_by_key(|&i| &shape[i]);
    out.iter().map(|i| *i as isize).collect()
}

pub fn shape_to_axis(old_shape: &[isize], new_shape: &[isize]) -> Vec<usize> {
    assert!(old_shape.len() == new_shape.len());
    let mut ret = Vec::new();
    for (i, (o, d)) in old_shape.iter().zip(new_shape.iter()).enumerate() {
        if o != d {
            ret.push(i as usize)
        }
    }
    ret
}

#[derive(Debug, Clone)]
pub struct Ctx(pub(crate) Vec<Tensor>);

impl Default for Ctx {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl Ctx {
    fn contains(&self, id: TensorId) -> bool {
        self.iter().any(|t| t.id == id)
    }
}

impl core::ops::Deref for Ctx {
    type Target = Vec<Tensor>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Ctx {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait Function: DynClone + core::fmt::Debug {
    fn type_name(&self) -> String {
        let full_name = std::any::type_name::<Self>().to_string();
        let splited: Vec<&str> = full_name.split(&['<', '>'][..]).collect();
        let function = splited[0]
            .split("::")
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let backend = splited[1]
            .split("::")
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        format!("{}<{}>", function.last().unwrap(), backend.last().unwrap())
    }
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer;
    fn backward(&mut self, grad: &LazyBuffer) -> Grad;
    fn parents_mut(&mut self) -> &mut Ctx;
    fn parents_ref(&self) -> &Ctx;
    fn apply(
        &mut self,
        x: &Tensor,
        y: Option<&Tensor>,
        z: Option<&Tensor>,
        shape: Option<Vec<isize>>,
        const_: Option<Vec<u8>>,
    ) -> Tensor
    where
        Self: 'static + Sized,
    {
        // These steps are done before this. Function::default().apply()
        // self.device = device
        // self.needs_input_grad = [t.requires_grad for t in tensors]
        // self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
        //
        let ctx = self;
        let ret_buffer = ctx.forward(
            &x.buffer,
            y.map(|t| &t.buffer),
            z.map(|t| &t.buffer),
            shape.as_deref(),
            const_,
        );
        let require_grad = x.require_grad
            || y.is_some_and(|t| t.require_grad)
            || z.is_some_and(|t| t.require_grad);
        // if self.require_grad: self.parents = tensors
        if require_grad {
            ctx.parents_mut().push(x.clone());
            if let Some(t) = y {
                ctx.parents_mut().push(t.clone());
            }
            if let Some(t) = z {
                ctx.parents_mut().push(t.clone());
            }
        }
        Tensor {
            device: ret_buffer.device.clone(),
            dtype: ret_buffer.dtype.clone(),
            buffer: ret_buffer.into(),
            require_grad,
            _ctx: if require_grad {
                Some(dyn_clone::clone_box(&*ctx))
            } else {
                None
            },
            id: super::tensor_id(),
            grad: std::sync::Arc::default(),
        }
    }
}

dyn_clone::clone_trait_object!(Function);

#[derive(Debug, Clone)]
pub enum Grad {
    One(LazyBuffer),
    Two(Option<LazyBuffer>, Option<LazyBuffer>),
}

// impl core::fmt::Debug for Grad {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match &self {
//             Grad::Contiguous(x)
//             | Grad::Sin(x)
//             | Grad::Log(x)
//             | Grad::Exp(x)
//             | Grad::Sqrt(x)
//             | Grad::Max(x)
//             | Grad::Sum(x)
//             | Grad::Sigmoid(x)
//             | Grad::Relu(x) => write!(f, "{x:?}"),
//             Grad::Add(x, y) | Grad::Sub(x, y) | Grad::Mul(x, y) | Grad::Div(x, y) => {
//                 write!(f, "x:{x:?}\ny:{y:?}")
//             }
//         }
//     }
// }

#[derive(Clone, Debug)]
pub struct Contiguous {
    pub(crate) ctx: Ctx,
}

impl Default for Contiguous {
    fn default() -> Self {
        Self {
            ctx: Ctx::default(),
        }
    }
}

impl Function for Contiguous {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        x.contiguous()
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(grad.clone())
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        todo!()
    }

    fn parents_ref(&self) -> &Ctx {
        todo!()
    }
    // fn forward(
    //     &mut self,
    //     x: &B,
    //     _: Option<&B>,
    //     _: Option<&B>,
    //     _: Option<Shape>,
    //     _: Option<B::Dtype>,
    // ) -> B {
    //     x.contiguous()
    // }
    //
    // fn backward(&mut self, grad: B) -> Grad {
    //     Grad::One(grad)
    // }
    //
    // fn parents_mut(&mut self) -> &mut Ctx {
    //     &mut self.ctx
    // }
    //
    // fn parents_ref(&self) -> &Ctx {
    //     &self.ctx
    // }
}

#[derive(Clone, Debug)]
pub struct Sin {
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Sin {
    fn default() -> Self {
        Self {
            x: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Sin {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.x = Some(x.clone());
        x.e(Unary::Sin, &[], None)
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let x = self.x.as_ref().unwrap();
        Grad::One(
            x.const_like(core::f32::consts::PI / 2.0)
                .e(Binary::Sub, &[x.clone()], None)
                .e(Unary::Sin, &[], None)
                .e(Binary::Mul, &[grad.clone()], None),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Log {
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Log {
    fn default() -> Self {
        Self {
            x: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Log {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.x = Some(x.clone());
        x.e(Unary::Log2, &[], None).e(
            Binary::Mul,
            &[x.const_like(2.0f32.log(core::f32::consts::E))],
            None,
        )
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(grad.e(Binary::Div, &[self.x.as_ref().unwrap().clone()], None))
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Exp {
    pub(crate) ret: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Exp {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Exp {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        let ret = x
            .e(
                Binary::Mul,
                &[x.const_like(1f32 / 2.0f32.ln())],
                None,
            )
            .e(Unary::Exp2, &[], None);
        self.ret = Some(ret.clone());
        ret
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(self.ret.as_ref().unwrap().e(Binary::Mul, &[grad.clone()], None))
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sqrt {
    pub(crate) ret: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Sqrt {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Sqrt {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.ret = Some(x.e(Unary::Sqrt, &[], None));
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let ret = self.ret.as_ref().unwrap();
        Grad::One(grad.e(
            Binary::Div,
            &[ret.e(Binary::Mul, &[ret.const_like(2.0f32)], None)],
            None,
        ))
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}
//FIXME: Both Sum/Max reduce op is using a hack on shape param in forward.
#[derive(Clone, Debug)]
pub struct Sum {
    pub(crate) input_shape: Option<Vec<isize>>,
    pub(crate) ctx: Ctx,
}

impl Default for Sum {
    fn default() -> Self {
        Self {
            input_shape: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Sum {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        // let shape = shape.unwrap();
        // self.input_shape = Some(x.shape);
        // if shape.len() == 1 {
        //     return x.sum(None, false);
        // }
        // let (keepdim, axis) = if shape.len() - x.shape().len() == 1 {
        //     //TODO: hack, need change
        //     (true, *shape.dims.last().unwrap())
        // } else {
        //     (false, shape.dims.iter().position(|e| *e == 0).unwrap())
        // };
        // x.sum(Some(axis as isize), keepdim)
        let shape = shape.unwrap();
        self.input_shape = Some(x.shape.clone());
        x.r(Reduce::Sum, shape)
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        // let input_shape = self
        //     .input_shape
        //     .as_ref()
        //     .expect("Sum bwd should have a input_shape");
        // if input_shape.len() > grad.shape.len() {
        //     let mut new_grad_shape = grad.shape;
        //     for _ in 0..input_shape.len() - grad.shape.len() {
        //         new_grad_shape.push(1);
        //     }
        //     grad = grad.reshape(&new_grad_shape);
        // }
        Grad::One(
            grad.expand(
                &self
                    .input_shape
                    .as_ref()
                    .expect("Sum bwd should have a input_shape"),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Max {
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) ret: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Max {
    fn default() -> Self {
        Self {
            x: None,
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Max {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        let shape = shape.unwrap();
        // if shape.len() == 1 {
        //     return x.sum(None, false);
        // }
        // let (keepdim, axis) = if shape.len() - x.shape().len() == 1 {
        //     //TODO: hack, need change
        //     (true, *shape.dims.last().unwrap())
        // } else {
        //     (false, shape.dims.iter().position(|e| *e == 0).unwrap())
        // };
        // self.x = Some(x.clone());
        // if !keepdim {
        //     panic!("please use reshape to remove dim")
        // }
        // self.ret = Some(x.max(Some(axis as isize), true));
        // self.ret.as_ref().unwrap().clone()
        self.x = Some(x.clone());
        let ret = x.r(Reduce::Max, shape);
        self.ret = Some(ret.clone());
        ret
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let x_ref = self.x.as_ref().unwrap();
        let ret_ref = self.ret.as_ref().unwrap();
        let max_is_1s = x_ref.const_like(1.0).e(
            Binary::Sub,
            &[x_ref.e(Binary::Cmplt, &[ret_ref.expand(&x_ref.shape)], None)],
            None,
        );
        // let mut div = max_is_1s.clone();
        // for (i, (msh, gsh)) in max_is_1s
        //     .shape()
        //     .dims
        //     .iter()
        //     .zip(grad.shape().dims.iter())
        //     .enumerate()
        // {
        //     if msh != gsh {
        //         div = div.sum(Some(i as isize), true);
        //     }
        // }
        // let div = div.expand(x_ref.shape());
        let div = max_is_1s
            .r(Reduce::Sum, &grad.shape)
            .expand(&self.x.as_ref().unwrap().shape);
        Grad::One(max_is_1s.e(Binary::Div, &[div], None).e(
            Binary::Mul,
            &[grad.expand(&self.x.as_ref().unwrap().shape)],
            None,
        ))
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Less {
    pub(crate) ctx: Ctx,
}

impl Default for Less {
    fn default() -> Self {
        Self {
            ctx: Ctx::default(),
        }
    }
}

impl Function for Less {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        x.e(
            Binary::Cmplt,
            &[y.expect("Less fwd op expects rhs").clone()],
            None,
        )
    }

    fn backward(&mut self, _grad: &LazyBuffer) -> Grad {
        unreachable!("Less op can not do backward pass")
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Add {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) y: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Add {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Add {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        x.e(
            Binary::Add,
            &[y.expect("Add fwd op expects rhs").clone()],
            None,
        )
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let x = if self.need_input_grad[0] {
            Some(grad.clone())
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(grad.clone())
        } else {
            None
        };
        Grad::Two(x, y)
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sub {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) y: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Sub {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Sub {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        x.e(
            Binary::Sub,
            &[y.expect("Sub fwd op expects rhs").clone()],
            None,
        )
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let x = if self.need_input_grad[0] {
            Some(grad.clone())
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(grad.const_like(0.0).e(Binary::Sub, &[grad.clone()], None))
        } else {
            None
        };
        Grad::Two(x, y)
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Mul {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) y: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Mul {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Mul {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.x = Some(x.clone());
        self.y = Some(y.expect("Mul fwd op expects rhs").clone());
        x.e(
            Binary::Mul,
            &[y.expect("Nul fwd op expects rhs").clone()],
            None,
        )
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let y = if self.need_input_grad[0] {
            Some(
                self.y
                    .as_ref()
                    .unwrap()
                    .e(Binary::Mul, &[grad.clone()], None),
            )
        } else {
            None
        };
        let x = if self.need_input_grad[1] {
            Some(
                self.x
                    .as_ref()
                    .unwrap()
                    .e(Binary::Mul, &[grad.clone()], None),
            )
        } else {
            None
        };
        Grad::Two(y, x)
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Div {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) y: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Div {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Div {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.x = Some(x.clone());
        self.y = Some(y.expect("Div fwd op expects rhs").clone());
        x.e(
            Binary::Div,
            &[y.expect("Div fwd op expects rhs").clone()],
            None,
        )
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let x = if self.need_input_grad[0] {
            Some(grad.e(Binary::Div, &[self.y.as_ref().unwrap().clone()], None))
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            let y_ref = self.y.as_ref().unwrap();
            Some(
                grad.e(Unary::Neg, &[], None)
                    .e(Binary::Mul, &[self.x.as_ref().unwrap().clone()], None)
                    .e(
                        Binary::Div,
                        &[y_ref.e(Binary::Mul, &[y_ref.clone()], None)],
                        None,
                    ),
            )
        } else {
            None
        };
        Grad::Two(x, y)
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sigmoid {
    pub(crate) ret: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Sigmoid {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.ret = Some(
            x.const_like(1.0).e(
                Binary::Div,
                &[x.const_like(1.0).e(
                    Binary::Add,
                    &[x.e(
                        Binary::Mul,
                        &[x.const_like(-1.0 / 2.0f32.log(core::f32::consts::E))],
                        None,
                    )
                    .e(Unary::Exp2, &[], None)],
                    None,
                )],
                None,
            ),
        );
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        let ret_ref = self.ret.as_ref().unwrap();
        Grad::One(
            ret_ref
                .e(
                    Binary::Mul,
                    &[ret_ref.const_like(1.0).e(
                        Binary::Sub,
                        &[self.ret.as_ref().unwrap().clone()],
                        None,
                    )],
                    None,
                )
                .e(Binary::Mul, &[grad.clone()], None),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Relu {
    pub(crate) ret: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
}

impl Default for Relu {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Relu {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.ret = Some(x.e(Binary::Max, &[x.const_like(0)], None));
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(
            self.ret
                .as_ref()
                .unwrap()
                .const_like(0)
                .e(Binary::Cmplt, &[self.ret.as_ref().unwrap().clone()], None)
                .e(Binary::Mul, &[grad.clone()], None),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}
// --------------------------------- Tenary
#[derive(Clone, Debug)]
pub struct Where {
    pub(crate) x: Option<LazyBuffer>,
    pub(crate) ctx: Ctx,
    pub(crate) need_input_grad: [bool; 3],
}

impl Default for Where {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false, false],
            x: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Where {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.x = Some(x.clone());
        x.e(
            Ternary::Where,
            &[
                y.expect("Where expect y").clone(),
                z.expect("Where expect z").clone(),
            ],
            None,
        )
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        // return None, \
        //        self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
        //        self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None
        let x = self.x.as_ref().expect("where bwd should have x now");
        Grad::Two(
            if self.need_input_grad[1] {
                Some(x.e(Ternary::Where, &[grad.clone(), grad.const_like(0)], None))
            } else {
                None
            },
            if self.need_input_grad[2] {
                Some(x.e(Ternary::Where, &[grad.const_like(0), grad.clone()], None))
            } else {
                None
            },
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}
// -------------------------------- Movement -------------------------------------
#[derive(Clone, Debug)]
pub struct Expand {
    pub(crate) input_shape: Option<Vec<isize>>,
    pub(crate) ctx: Ctx,
}

impl Default for Expand {
    fn default() -> Self {
        Self {
            input_shape: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Expand {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        self.input_shape = Some(x.shape.clone());
        x.expand(shape.expect("Expand mlops expect a shape"))
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(
            grad.r(
                Reduce::Sum,
                self.input_shape
                    .as_ref()
                    .expect("Expand bwd expects a input shape"),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Reshape {
    pub(crate) input_shape: Option<Vec<isize>>,
    pub(crate) ctx: Ctx,
}

impl Default for Reshape {
    fn default() -> Self {
        Self {
            input_shape: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Reshape {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        assert!(!shape.as_ref().unwrap().is_empty());
        self.input_shape = Some(x.shape.clone());
        x.reshape(shape.expect("Reshape mlops expect a shape"))
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(
            grad.reshape(
                self.input_shape
                    .as_ref()
                    .expect("Reshape backward should already have a shape"),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Permute {
    pub(crate) permute_order: Option<Vec<isize>>,
    pub(crate) ctx: Ctx,
}

impl Default for Permute {
    fn default() -> Self {
        Self {
            permute_order: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Permute {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        let order = shape
            .expect("Permute mlops expect a permute order")
            .to_vec();
        self.permute_order = Some(order);
        x.permute(self.permute_order.as_ref().unwrap())
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(
            grad.permute(&argsort::<&[isize]>(
                self.permute_order
                    .as_ref()
                    .expect("Permute bwd order should not be empty")
                    .as_ref(),
            )),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

//NOTE: Pad/Shrink stores in Vec<(usize, usize)>, so we flatten that into a vec<usize> when using
//      this, such that we dont need a new param in this forwrad()
#[derive(Clone, Debug)]
pub struct Pad {
    pub(crate) narg: Option<Vec<isize>>,
    pub(crate) ctx: Ctx,
}

impl Default for Pad {
    fn default() -> Self {
        Self {
            narg: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Pad {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        let flatten_p = shape.unwrap();
        let mut narg = Vec::new();
        let mut arg = Vec::new();
        for (sh, p) in x.shape.iter().zip(flatten_p.windows(2).step_by(2)) {
            narg.push(vec![p[0], sh + p[0]]);
            arg.push(vec![p[0], p[1]]);
        }
        assert!(
            narg.len() == x.shape.len(),
            "Pad fwd: Something is wrong when creating Vec<(usize, usize)>, padding:{:?} x:{:?}",
            narg,
            x.shape
        );
        let narg = narg.concat();
        let arg = arg.concat();
        self.narg = Some(narg.clone());
        x.pad(&arg)
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(
            grad.shrink(
                self.narg
                    .as_ref()
                    .expect("Reshape backward should already have a shape"),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Shrink {
    pub(crate) narg: Option<Vec<isize>>,
    pub(crate) ctx: Ctx,
}

impl Default for Shrink {
    fn default() -> Self {
        Self {
            narg: None,
            ctx: Ctx::default(),
        }
    }
}

impl Function for Shrink {
    fn forward(
        &mut self,
        x: &LazyBuffer,
        y: Option<&LazyBuffer>,
        z: Option<&LazyBuffer>,
        shape: Option<&[isize]>,
        const_: Option<Vec<u8>>,
    ) -> LazyBuffer {
        let flatten_p = shape.unwrap();
        let mut narg = Vec::new();
        let mut padding = Vec::new();
        for (sh, p) in x.shape.iter().zip(flatten_p.windows(2).step_by(2)) {
            narg.push(vec![p[0], sh - p[1]]);
            padding.push(vec![p[0], p[1]]);
        }
        let narg = narg.concat();
        let padding = padding.concat();
        self.narg = Some(narg.clone());
        x.shrink(&padding)
    }

    fn backward(&mut self, grad: &LazyBuffer) -> Grad {
        Grad::One(
            grad.pad(
                self.narg
                    .as_ref()
                    .expect("Reshape backward should already have a shape"),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx {
        &self.ctx
    }
}

// #[test]
// fn mlop_sin() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.sin();
//     approx_eq!(
//         x,
//         [
//             -0.841471,
//             0.9092974,
//             -0.14112,
//             -0.7568025,
//             0.9589243,
//             -0.2794155,
//             -0.6569866,
//             0.98935825,
//             -0.41211846
//         ]
//     );
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [
//             0.54030222,
//             -0.41614679,
//             -0.9899925,
//             -0.65364379,
//             0.28366235,
//             0.96017021,
//             0.75390244,
//             -0.14549987,
//             -0.91113013
//         ]
//     );
// }
//
// #[test]
// fn mlop_relu() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.relu();
//
//     approx_eq!(x, [0., 2., 0., 4., 0., 6., 0., 8., 0.]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [0., 1., 0., 1., 0., 1., 0., 1., 0.]
//     );
//
//     let mut t =
//         Tensor::<Cpu>::from_shape([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.relu();
//     approx_eq!(x, [1., 0., 3., 0., 5., 0., 7., 0., 9.]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [1., 0., 1., 0., 1., 0., 1., 0., 1.]
//     );
// }
//
// #[test]
// fn mlop_log() {
//     let mut t = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.log();
//     approx_eq!(
//         x,
//         [
//             0., 0.6931472, 1.0986123, 1.3862944, 1.6094378, 1.7917595, 1.9459102, 2.0794415,
//             2.1972246
//         ]
//     );
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [1., 0.5, 0.33333333, 0.25, 0.2, 0.16666667, 0.14285714, 0.125, 0.11111111]
//     );
// }
//
// #[test]
// fn mlop_exp() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.exp();
//
//     approx_eq!(
//         x,
//         [
//             0.36787945,
//             7.3890557,
//             0.04978707,
//             54.5981315,
//             0.006737951,
//             403.42868,
//             0.00091188244,
//             2980.9558,
//             0.0001234099
//         ]
//     );
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [
//             0.36787945,
//             7.38905573,
//             0.04978707,
//             54.59813835,
//             0.006737951,
//             403.42868042,
//             0.00091188244,
//             2980.95586367,
//             0.00012340993
//         ]
//     );
// }
//
// #[test]
// fn mlop_sqrt() {
//     let mut t = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.sqrt();
//     approx_eq!(
//         x,
//         [1., 1.4142135, 1.7320508, 2., 2.236068, 2.4494898, 2.6457512, 2.828427, 3.]
//     );
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [
//             0.5, 0.3535534, 0.28867514, 0.25, 0.22360679, 0.20412414, 0.18898224, 0.1767767,
//             0.16666667
//         ]
//     );
// }
//
// #[test]
// fn mlop_sigmoid() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.sigmoid();
//     approx_eq!(
//         x,
//         [
//             0.26894143,
//             0.880797,
//             0.04742588,
//             0.98201376,
//             0.0066928547,
//             0.9975274,
//             0.0009110517,
//             0.99966466,
//             0.0001233947
//         ]
//     );
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [
//             0.19661194,
//             0.10499363,
//             0.04517666,
//             0.01766273,
//             0.00664806,
//             0.0024664658,
//             0.00091022166,
//             0.00033522327,
//             0.00012337948
//         ]
//     );
// }
//
// #[test]
// fn mlop_sum() {
//     let mut t = Tensor::<Cpu>::from_shape(
//         [
//             0.26894143,
//             0.880797,
//             0.04742588,
//             0.98201376,
//             0.0066928547,
//             0.9975274,
//             0.0009110517,
//             0.99966466,
//             0.0001233947,
//         ],
//         [3, 3],
//     );
//     t.require_grad = true;
//     let mut x = t.sum_all();
//     approx_eq!(x, [4.184098]);
//     x.backward();
//     assert!(t.shape() == t.grad.lock().unwrap().as_ref().unwrap().shape());
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [1., 1., 1., 1., 1., 1., 1., 1., 1.]
//     );
// }
//
// #[test]
// fn mlop_max() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.max_all();
//     approx_eq!(x, [8.0]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [0., 0., 0., 0., 0., 0., 0., 1., 0.]
//     );
//
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.max(0);
//     approx_eq!(x, [4.0, 8.0, 6.0]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [0., 0., 0., 1., 0., 1., 0., 1., 0.]
//     );
//
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.max(1);
//     approx_eq!(x, [2.0, 6.0, 8.0]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [0., 1., 0., 0., 0., 1., 0., 1., 0.]
//     );
//
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     t.require_grad = true;
//     let x = t.max(2);
//     approx_eq!(x, [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [1., 1., 1., 1., 1., 1., 1., 1., 1.]
//     );
// }
//
// #[test]
// fn mlop_less() {
//     let t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     let b = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
//     let x = t._lt(&b);
//     approx_eq!(x, [1., 0., 1., 0., 1., 0., 1., 0., 1.]);
//     // less has no bwd pass
// }
//
// #[test]
// fn mlop_add() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     let mut b = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
//     t.require_grad = true;
//     b.require_grad = true;
//     let x = &t + &b;
//     approx_eq!(x, [0., 4., 0., 8., 0., 12., 0., 16., 0.]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [1., 1., 1., 1., 1., 1., 1., 1., 1.]
//     );
// }
//
// #[test]
// fn mlop_sub() {
//     let mut t =
//         Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
//     let mut b = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
//     t.require_grad = true;
//     b.require_grad = true;
//     let x = &t - &b;
//     approx_eq!(x, [-2., 0., -6., 0., -10., 0., -14., 0., -18.]);
//     x.sum_all().backward();
//     approx_eq!(
//         t.grad.lock().unwrap().as_ref().unwrap(),
//         [1., 1., 1., 1., 1., 1., 1., 1., 1.]
//     );
// }
//
// #[test]
// fn mlop_mul() {
//     let a = Tensor::<Cpu>::empty([3]).const_like(4.0);
//     let b = Tensor::<Cpu>::empty([3]).const_like(0.5);
//     let out = a * b;
//     approx_eq!(out, [2.0, 2.0, 2.0]);
// }
//
// #[test]
// fn mlop_div() {
//     todo!()
// }
//
// #[test]
// fn mlop_where() {
//     todo!()
// }
//
// #[test]
// fn mlop_expand() {
//     todo!()
// }
//
// #[test]
// fn mlop_reshape() {
//     todo!()
// }
//
// #[test]
// fn mlop_permute() {
//     todo!()
// }
//
// #[test]
// fn mlop_pad() {
//     todo!()
// }
//
// #[test]
// fn mlop_shrink() {
//     todo!()
// }
//
// #[test]
// fn mlop_flip() {
//     todo!()
// }
