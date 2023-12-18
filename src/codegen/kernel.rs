use std::hash::Hash;

use crate::prelude::*;
use crate::{
    dtype,
    lazy::{LOArc, LazyBuffer},
    ops::{LazyOp, OpType, DEVICE},
    shape::ShapeTracker,
};

// @dataclass(frozen=True)
// class MemBuffer:
//   idx: int
//   dtype: DType
//   st: ShapeTracker
//
// @dataclass(frozen=True)
// class ConstBuffer:
//   val: Union[int, float]
//   dtype: DType
//   st: ShapeTracker
//
// @dataclass(frozen=True)
// class ScheduleItem:
//   ast: LazyOp
//   out: LazyBuffer
//   inputs: Tuple[LazyBuffer, ...]
//   var_vals: Dict[Variable, int]

#[derive(Debug)]
pub struct MemBuffer {
    pub idx: usize,
    pub dtype: dtype::Dtype,
    pub st: ShapeTracker,
}

#[derive(Debug)]
pub struct ConstBuffer {
    pub val: String,
    pub dtype: dtype::Dtype,
    pub st: ShapeTracker,
}

pub struct LocalBuffer {
    pub name: String,
    pub size: usize,
    pub dtype: dtype::Dtype,
    pub realized: bool,
}

impl core::fmt::Display for LocalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "localbuffer<{}[{}]>", self.name, self.size)
    }
}

#[derive(Clone, Debug)]
pub struct LinearizerOptions {
    support_float4: bool,
    support_float4_alu: bool,
    has_local: bool,
    global_max: Option<Vec<isize>>,
    local_max: Option<Vec<isize>>,
}

#[derive(Clone, Debug)]
pub struct Kenrel {
    ast: LazyOp,
    opts: LinearizerOptions,
    bufs: Vec<LazyBuffer>,
    reduceop: Option<LazyOp>,
}

#[allow(unused_variables)]
impl Kenrel {
    pub fn new(ast: LazyOp, opts: Option<LinearizerOptions>) -> Self {
        // let ast = if ast.optype == Movement::Reshape {
        //     ast.src[0].clone().to_lo()
        // } else {
        //     ast.clone()
        // };
        //assert!(matches!(ast.optype, BufferOps))
        let opts = opts.unwrap_or(DEVICE.linearizer_opts());
        let reduceop = v![x, for x in ast.get_lazyops(), if matches!(x.optype, OpType::Reduce(_))];
        todo!()
    }
}
