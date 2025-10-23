#![feature(get_mut_unchecked)]

use storm::dtype::{Dtype, NumType};
use storm::shape::Shape;
use storm::shape::list::ShapeVec;

use std::{
    ops::{Add, Deref, DerefMut},
    sync::Arc,
};

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpT {
    #[default]
    Nop,

    Add,
    Sub,
    Mul,
    Div,
    Mod,

    Reshape,
    Stride,

    Const,

    Load,
    Store,
}

#[derive(Debug, Clone)]
pub enum Arg {
    Shape(ShapeVec),
    Const(String),
}

#[derive(Clone, Default)]
pub struct Op_ {
    t: OpT,
    dtype: Dtype,
    children: Vec<Op>,
    src: Vec<Op>,
    args: Vec<Arg>,
    buffer: Option<*mut std::ffi::c_void>,
}

impl std::fmt::Debug for Op_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Op_")
            .field("t", &self.t)
            // .field("children", &self.children)
            // .field("src", &self.children)
            .field("args", &self.args)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Op(Arc<Op_>);

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for Op {
    type Target = Op_;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Op {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Arc::get_mut_unchecked(&mut self.0) }
    }
}

unsafe impl Sync for Op {}
unsafe impl Send for Op {}

impl Op {
    fn add_src(&self, parents: &[&Self]) {
        let mut c = self.0.clone();
        unsafe {
            Arc::get_mut_unchecked(&mut c)
                .src
                .extend(parents.iter().map(|x| (**x).clone()))
        }
    }

    fn add_child(&self, children: &[&Self]) {
        let mut c = self.0.clone();
        unsafe {
            Arc::get_mut_unchecked(&mut c)
                .children
                .extend(children.iter().map(|x| (**x).clone()))
        }
    }

    fn const_<N: NumType>(value: N) -> Self {
        Op(Arc::new(Op_ {
            t: OpT::Const,
            args: vec![Arg::Const(value.to_string())],
            ..Default::default()
        }))
    }

    fn add(&self, rhs: &Self) -> Self {
        let ret = Op(Arc::new(Op_ {
            t: OpT::Add,
            src: vec![self.clone(), rhs.clone()],
            ..Default::default()
        }));
        let p = &[rhs];
        self.add_child(p);
        rhs.add_child(p);
        ret
    }

    fn toposort(&self) -> Vec<Self> {
        use std::collections::VecDeque;
        let mut ret: Vec<Self> = Vec::new();
        let mut stack: VecDeque<(&Self, bool)> = [(self, false)].into();
        while !stack.is_empty() {
            let (node, visited) = stack.pop_back().unwrap();
            if ret.contains(node) {
                continue;
            }
            if !visited {
                stack.push_back((node, true));
                for n in node.src.iter().rev() {
                    stack.push_back((n, false))
                }
            } else {
                ret.push(node.clone())
            }
        }
        ret
    }

    fn render(&self) -> String {
        let nodes = self.toposort();
        for n in nodes.iter() {
            if n.t == OpT::Add {
            }
        }
        dbg!(nodes);
        "".into()
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    op: Op,
    shape: Shape,
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            op: self.op.add(&rhs.op),
            shape: self.shape.clone(),
        }
    }
}

impl Tensor {
    fn const_<S: Into<ShapeVec>>(val: f32, shape: S) -> Self {
        Tensor {
            op: Op::const_(val),
            shape: shape.into().to_shape(),
        }
    }
    fn ones<S: Into<ShapeVec>>(shape: S) -> Self {
        Self::const_(1.0, shape)
    }
}

fn main() {
    let a = Tensor::ones(&[3, 3]);
    let b = Tensor::ones(&[3, 3]);
    let c = a + b;
    println!("{}", c.op.render());
}
