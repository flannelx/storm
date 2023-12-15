use crate::{ops::OpType, prelude::*};

#[derive(Clone, Debug, Hash, Eq)]
pub enum Arg {
    Str(String),
    Op(OpType),
    Buf(LazyBuffer),
    Num(Vec<u8>), // in little-endian bytes
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
            Arg::Op(op) => op.clone(),
            t => panic!("Can not to_op() {t:?}"),
        }
    }

    pub fn to_buf(&self) -> LazyBuffer {
        match self {
            Arg::Buf(buf) => buf.clone(),
            t => panic!("Can not to_buf() {t:?}"),
        }
    }

    pub fn to_num<T: dtype::FromBytes>(&self) -> T {
        match self {
            Arg::Num(bytes) => T::from_le_bytes(bytes),
            t => panic!("Can not to_buf() {t:?}"),
        }
    }
}

impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        self == other
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
            Arg::Op(op) => op == other,
            _ => false,
        }
    }
}
