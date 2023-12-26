use crate::{
    codegen::kernel::Buffers,
    ops::{LazyOp, OpType},
    prelude::*,
};

#[derive(Clone, Debug, Hash, Eq)]
pub enum Arg {
    Str(String),
    OpType(OpType),
    Op(LazyOp),
    Buffer(Buffers),
    Usize(usize),
    Idx(isize),
    Shape(Vec<isize>),
}

impl Arg {
    pub fn shape(&self) -> Vec<isize> {
        match self {
            Arg::Shape(s) => s.clone(),
            t => panic!("Can not to_shape() {t:?}"),
        }
    }
    pub fn to_str(&self) -> String {
        match self {
            Arg::Str(s) => s.clone(),
            Arg::Idx(i) => i.to_string(),
            t => panic!("Can not to_str() {t:?}"),
        }
    }

    pub fn to_idx(&self) -> isize {
        match self {
            Arg::Idx(s) => *s,
            t => panic!("Can not to_idx() {t:?}"),
        }
    }

    pub fn to_op(&self) -> OpType {
        match self {
            Arg::OpType(op) => op.clone(),
            t => panic!("Can not to_op() {t:?}"),
        }
    }

    pub fn to_buf(&self) -> Buffers {
        match self {
            Arg::Buffer(buf) => buf.clone(),
            t => panic!("Can not to_buf() {t:?}"),
        }
    }
}

impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Arg::Str(a), Arg::Str(b)) => a == b,
            (Arg::OpType(a), Arg::OpType(b)) => a == b,
            (Arg::Op(a), Arg::Op(b)) => a == b,
            (Arg::Buffer(a), Arg::Buffer(b)) => a == b,
            //(Arg::Num(a), Arg::Num(b)) => a == b,
            (Arg::Usize(a), Arg::Usize(b)) => a == b,
            (Arg::Idx(a), Arg::Idx(b)) => a == b,
            (Arg::Shape(a), Arg::Shape(b)) => a == b,
            (Arg::Str(a), Arg::Idx(b)) => a.parse::<isize>().is_ok_and(|n| n == *b),
            (Arg::Idx(a), Arg::Str(b)) => b.parse::<isize>().is_ok_and(|n| n == *a),
            (a, b) => {
                println!("ARG Compare: {a:?} != {b:?} ??");
                false
            }
        }
    }
}

impl PartialEq<OpType> for Arg {
    fn eq(&self, other: &OpType) -> bool {
        match self {
            Arg::OpType(op) => op == other,
            _ => false,
        }
    }
}
