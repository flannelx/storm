#![allow(non_snake_case)]
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::arg::Arg;
use crate::dtype::{_bool, float16, float32, int32};
use crate::lazy::{LazyBufferId, STArc};
use crate::ops::{Binary, Ternary, Unary};
use crate::prelude::*;
use crate::renderer::cstyle::uops_to_cstyle;
use crate::shape::symbolic::{iter_idxs, num, var, ArcNode, NodeOp};
use crate::tensor::shape::Shape;
use crate::{
    dtype,
    lazy::{LOArc, LazyBuffer},
    ops::{self, LazyOp, OpType, DEVICE},
    shape::ShapeTracker,
};

use super::linearizer::{LinearizerOptions, UOp, UOps};

#[derive(Debug, Clone, PartialEq)]
pub enum ConstNum {
    Int(i128),
    Float(f32),
}

impl std::fmt::Display for ConstNum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = match self {
            ConstNum::Int(n) => n.to_string(),
            ConstNum::Float(n) => format!("{:?}",n),
        };
        write!(f, "{n}")
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemBuffer {
    pub idx: usize,
    pub dtype: dtype::Dtype,
    pub st: ShapeTracker,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstBuffer {
    pub val: String,
    pub dtype: dtype::Dtype,
    pub st: ShapeTracker,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalBuffer {
    pub name: String,
    pub size: usize,
    pub dtype: dtype::Dtype,
    pub realized: Option<Box<Buffers>>,
}

impl core::fmt::Display for LocalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "localbuffer<{}[{}]>", self.name, self.size)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Buffers {
    MemBuffer(MemBuffer),
    ConstBuffer(ConstBuffer),
    LazyBuffer(LazyBuffer),
    LocalBuffer(LocalBuffer),
}

impl Buffers {
    pub fn dtype(&self) -> Dtype {
        match self {
            Buffers::MemBuffer(b) => b.dtype.clone(),
            Buffers::ConstBuffer(b) => b.dtype.clone(),
            Buffers::LazyBuffer(b) => b.dtype.clone(),
            Buffers::LocalBuffer(b) => b.dtype.clone(),
        }
    }

    pub fn st(&self) -> ShapeTracker {
        match self {
            Buffers::MemBuffer(b) => b.st.clone(),
            Buffers::ConstBuffer(b) => b.st.clone(),
            Buffers::LazyBuffer(b) => (*b.st).clone(),
            Buffers::LocalBuffer(b) => panic!("Local buffer does not have shapetracker {b:?}"),
        }
    }

    pub fn idx(&self) -> usize {
        match self {
            Buffers::MemBuffer(b) => b.idx,
            t => panic!("{t:?} does not have a idx"),
        }
    }
}

macro_rules! BuffersFrom {
    ($type:tt) => {
        impl From<$type> for Buffers {
            fn from(value: $type) -> Self {
                Buffers::$type(value)
            }
        }

        impl Buffers {
            fn $type(self) -> $type {
                match self {
                    Buffers::$type(value) => value,
                    t => panic!("{t:?}"),
                }
            }
        }
    };
}

BuffersFrom!(MemBuffer);
BuffersFrom!(ConstBuffer);
BuffersFrom!(LazyBuffer);
BuffersFrom!(LocalBuffer);

#[derive(Clone, Debug)]
pub struct Kernel {
    pub ast: LazyOp,
    pub opts: LinearizerOptions,
    pub bufs: Vec<Buffers>,
    pub reduceop: Option<LazyOp>,
    pub earlybufs: Vec<Buffers>,
    pub full_buf_idx: usize,
    pub sts: Vec<ShapeTracker>, // This might need deepclone
    pub local_dims: isize,
    pub upcasted: isize,
    pub group_for_reduce: Vec<isize>,
    pub dont_use_locals: bool,
    pub local_alias: HashMap<usize, LocalBuffer>,
}

impl Kernel {
    pub fn new(ast: LazyOp, opts: Option<LinearizerOptions>) -> Self {
        // assert!(
        //     matches!(ast.optype, OpType::Buffer(_)),
        //     "Kernel must have a store as the output, got {ast:?}"
        // );
        let opts = opts.unwrap_or(DEVICE.linearizer_opts());
        let reduceop = v![x, for x in ast.get_lazyops(), if matches!(x.optype, OpType::Reduce(_))]
            .into_iter()
            .next();
        let mut bufs = v![b.to_buf(), for b in x.args, for x in ast.get_lazyops(), if matches!(x.optype, OpType::Buffer(_))];
        bufs.dedup();
        let earlybufs = if let Some(op) = &reduceop {
            v![b.to_buf(),for b in x.args, for x in op.get_lazyops(), if matches!(x.optype, OpType::Buffer(_))]
        } else {
            vec![]
        };
        let full_buf_idx = if earlybufs.len() > 0 {
            bufs.iter().position(|p| p == &earlybufs[0]).unwrap()
        } else {
            0
        };
        let sts = v![x.st(), for x in bufs.iter()];
        // full_shape(): @properties: self.sts[self.full_buf_index].shape
        let reduce = v![(i as isize, sh1, sh2), for (i, (sh1, sh2)) in izip!(sts[full_buf_idx].shape(), sts[0].shape()).enumerate()];
        let permute = vec![
            v![*i, for (i, s, n) in reduce.iter(), if s == n],
            v![*i, for (i, s, n) in reduce.iter(), if s != n],
        ]
        .concat();
        let mut ret = Self {
            ast,
            opts,
            bufs: v![b.into(), for b in bufs],
            reduceop,
            earlybufs: v![b.into(), for b in earlybufs],
            full_buf_idx,
            sts,
            local_dims: 0,
            upcasted: 0,
            group_for_reduce: vec![],
            dont_use_locals: false,
            local_alias: HashMap::new(),
            //applied_opts: vec![],
        };
        ret.reshape_and_permute(None, Some(permute));
        // # parameters for optimization
        // self.applied_opts: List[Opt] = []
        // self.local_alias: Dict[int, LocalBuffer] = {}
        // self.tensor_core: Optional[TensorCore] = None
        // self.dont_use_locals: bool = False
        ret.simplify_ones();
        ret.simplify_merge_adjacent();
        // # cache
        // self.applied_opts_cache: Optional[List[Opt]] = None
        ret
    }

    pub fn reshape_and_permute(
        &mut self,
        new_shape_fxn: Option<Box<dyn Fn(Vec<isize>) -> Vec<isize>>>,
        axis: Option<Vec<isize>>,
    ) {
        let mut new_sts = vec![];
        for st in self.sts.iter() {
            let mut new_st = st.clone();
            if let Some(ref fxn) = new_shape_fxn {
                new_st = new_st.reshape(&fxn(new_st.shape()));
            }
            if let Some(a) = &axis {
                new_st = new_st.permute(&a);
            }
            new_sts.push(new_st);
        }
        self.sts = new_sts;
    }

    pub fn shape_len(&self) -> isize {
        self.sts[0].shape().len() as isize
    }

    pub fn full_shape(&self) -> Vec<isize> {
        self.sts[self.full_buf_idx].shape()
    }

    pub fn first_reduce(&self) -> isize {
        if self.shape_len() < self.upcasted {
            return 0;
        }
        v![x!=y, for (x,y) in izip!(self.sts[0].shape()[..(self.shape_len()-self.upcasted) as usize].iter(),self.full_shape()[..(self.shape_len()-self.upcasted)as usize].iter())]
            .iter()
            .position(|&t| t == true)
            .unwrap_or((self.shape_len()-self.upcasted) as usize) as isize
    }

    pub fn simplify_ones(&mut self) -> bool {
        if self.shape_len() == 0 {
            return false;
        }
        let all_ones = v![if s==1 { 1 } else { 0 }, for s in self.full_shape()];
        let first_reduce = self.first_reduce();
        self.local_dims -= all_ones
            [(first_reduce - self.local_dims) as usize..first_reduce as usize]
            .iter()
            .sum::<isize>();
        self.upcasted -= all_ones[(self.shape_len() - self.upcasted) as usize..]
            .iter()
            .sum::<isize>();
        let ret = all_ones.iter().sum::<isize>() > 0;
        self.reshape_and_permute(
            Some(Box::new(move |shape: Vec<isize>| -> Vec<isize> {
                v![*x, for (i,x) in shape.iter().enumerate(), if all_ones[i] != 1]
            })),
            None,
        );
        ret
    }

    pub fn simplify_merge_adjacent(&mut self) {
        if self.shape_len() == 0 {
            return;
        }

        let shapes = v![x.shape(), for x in self.sts.iter()];
        let strides = v![x.real_strides(false), for x in self.sts.iter()];

        let mut rets = v![vec![(shapes[j][0], strides[j][0])], for j in 0..shapes.len()];
        for i in 1..shapes[0].len() {
            let mut can_merge = vec![];
            for j in 0..shapes.len() {
                if let Some(stride) = strides[j][i] {
                    //can_merge.append(strides[j][i] is not None and ((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*cast(int, strides[j][i])) or (strides[j][i] == 0 and rets[j][-1][1] == 0)))
                    #[rustfmt::skip]
                    can_merge.push((stride != 0 && rets[j].last().unwrap().1.is_some_and(|n| n == shapes[j][i] * stride)) || (stride == 0 && rets[j].last().unwrap().1.is_some_and(|n| n == 0)),
                    );
                } else {
                    can_merge.push(false);
                }
            }
            let mergeable = can_merge.iter().all(|t| *t) && i as isize != self.first_reduce();
            for j in 0..shapes.len() {
                if mergeable {
                    *rets[j].last_mut().unwrap() =
                        (rets[j].last().unwrap().0 * shapes[j][i], strides[j][i]);
                } else {
                    rets[j].push((shapes[j][i], strides[j][i]));
                }
            }
        }

        for (i, x) in rets[..self.sts.len()].iter().enumerate() {
            self.sts[i] = self.sts[i].reshape(&v![y.0, for y in x]);
        }
    }

    pub fn global_dims(&self) -> isize {
        self.first_reduce() - self.local_dims
    }

    pub fn upcasted_axis(&self, i: isize) -> Vec<Vec<isize>> {
        todo!()
    }

    pub fn output_shape(&self) -> Vec<isize> {
        self.sts[0].shape()
    }
}

lazy_static::lazy_static! {
    pub static ref KERNEL_CNT: Mutex<HashMap<String, usize>> = Default::default();
}
