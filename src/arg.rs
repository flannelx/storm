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
    Num(Vec<u8>), // in little-endian bytes, for devices
    Usize(usize),
    Idx(isize),
}

impl Arg {
    pub fn to_str(&self) -> String {
        match self {
            Arg::Str(s) => s.clone(),
            t => panic!("Can not to_str() {t:?}"),
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

    pub fn to_num<T: dtype::NumType>(&self) -> T {
        match self {
            Arg::Num(bytes) => T::from_le_bytes(bytes),
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
            (Arg::Num(a), Arg::Num(b)) => a == b,
            (Arg::Usize(a), Arg::Usize(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialEq<str> for Arg {
    fn eq(&self, other: &str) -> bool {
        match self {
            Arg::Str(s) => s == other,
            _ => false,
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
