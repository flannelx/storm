#![allow(non_snake_case)]
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use itertools::Itertools;

use crate::arg::Arg;
use crate::codegen::linearizer::cartesian_product;
use crate::dtype::{_bool, float16, float32, int32, NumType};
use crate::lazy::{LazyBufferId, STArc};
use crate::ops::{Binary, Buffer, Reduce, Ternary, Unary};
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
            ConstNum::Float(n) => format!("{:?}", n),
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
            Buffers::LazyBuffer(b) => b.st.clone(),
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum OptOps {
    UPCAST,
    UPCASTMID,
    UNROLL,
    LOCAL,
    LASTLOCAL,
    GROUP,
    GROUPTOP,
    NOLOCALS,
    PADTO,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opt {
    op: OptOps,
    axis: Option<isize>,
    amt: Option<isize>,
}

impl Default for Opt {
    fn default() -> Self {
        Self {
            op: OptOps::LOCAL,
            axis: None,
            amt: None,
        }
    }
}

lazy_static::lazy_static! {
    pub static ref KERNEL_CNT: Mutex<HashMap<String, usize>> = Default::default();
}

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
    pub applied_opts: Vec<Opt>,
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
        let reduce = v![(i as isize, sh1, sh2), for (i, (sh1, sh2)) in izip!(sts[full_buf_idx].shape_vec(), sts[0].shape_vec()).enumerate()];
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
            applied_opts: vec![],
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
                let new_shape = fxn(new_st.shape_vec());
                new_st = new_st.reshape(&new_shape);
            }
            if let Some(a) = &axis {
                new_st = new_st.permute(&a);
            }
            new_sts.push(new_st);
        }
        self.sts = new_sts;
    }

    pub fn shape_len(&self) -> isize {
        self.sts[0].shape_vec().len() as isize
    }

    pub fn full_shape(&self) -> Shape {
        self.sts[self.full_buf_idx].shape()
    }

    pub fn first_reduce(&self) -> isize {
        if self.shape_len() < self.upcasted {
            return 0;
        }
        v![x!=y, for (x,y) in izip!(self.sts[0].shape_vec()[..(self.shape_len()-self.upcasted) as usize].iter(),self.full_shape().dims[..(self.shape_len()-self.upcasted)as usize].iter())]
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

        let shapes = v![x.shape_vec(), for x in self.sts.iter()];
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

    pub fn upcasted_axis(&self, i: isize) -> Vec<(isize, Option<isize>, isize)> {
        // return list(zip(self.sts[i].shape[self.shape_len-self.upcasted:],
        //                 self.sts[i].real_strides()[self.shape_len-self.upcasted:],
        //                 [x!=y for x,y in zip(self.sts[0].shape[self.shape_len-self.upcasted:], self.full_shape[self.shape_len-self.upcasted:])]))

        let i = if i < 0 {
            (self.sts.len() as isize + i) as usize
        } else {
            i as usize
        };
        v![(a, b.clone(), c), for (a, b, c) in izip!(
            self.sts[i].shape()[self.shape_len()-self.upcasted..].to_vec(),
            self.sts[i].real_strides(false)[(self.shape_len()-self.upcasted) as usize..].into_iter(),
            v![if x!=y { 1 } else { 0 }, for (x, y) in izip!(self.sts[0].shape()[self.shape_len()-self.upcasted..].iter(), self.full_shape()[self.shape_len()-self.upcasted..].iter())]
            )
        ]
    }

    pub fn output_shape(&self) -> Vec<isize> {
        self.sts[0].shape_vec()
    }

    pub fn apply_opt(&mut self, opt: Opt) {
        use OptOps::*;
        assert!(
            !self.dont_use_locals
                || !matches!(opt.op, LOCAL | LASTLOCAL | GROUP | GROUPTOP | UPCASTMID),
            "not using locals"
        );
        self.applied_opts.push(opt.clone());
        let mut axis = -1;
        if let Some(opt_axis) = opt.axis {
            //axis = opt.axis + (self.first_reduce if opt.op == OptOps.UNROLL else (self.first_reduce+len(self.group_for_reduce) if opt.op in [OptOps.GROUP, OptOps.GROUPTOP] else 0))  # noqa: E501
            axis = opt_axis
                + (if opt.op == UNROLL {
                    self.first_reduce()
                } else {
                    if matches!(opt.op, GROUP | GROUPTOP) {
                        self.first_reduce() + self.group_for_reduce.len() as isize
                    } else {
                        0
                    }
                });
        }
        let mut amt = -1;
        if let Some(opt_amt) = opt.amt {
            if opt_amt != 0 {
                amt = opt_amt;
            } else {
                amt = self.full_shape()[axis];
                assert!(amt != 1, "shift/padto of amt 1 or Node is meaningless");
                if opt.op != PADTO {
                    assert!(self.full_shape()[axis] % amt == 0)
                }
            }
        }
        match opt.op {
            LOCAL | LASTLOCAL => {
                assert!(self.opts.has_local, "target does not support local");
                assert!(axis < self.first_reduce(), "can't local a reduce");
                if opt.op == LOCAL {
                    // assert not tensor cores
                    self.shift_to(axis, amt, None, Some(self.first_reduce()));
                } else {
                    self.shift_to(axis, amt, None, Some(self.first_reduce() + self.local_dims));
                }
                self.local_dims += 1;
            }
            GROUP | GROUPTOP => {
                assert!(
                    self.opts.has_local && self.opts.has_share,
                    "target does not support local or shared mem"
                );
                assert!(
                    axis >= self.first_reduce() + self.group_for_reduce.len() as isize
                        && axis < self.shape_len() - self.upcasted,
                    "must be reduce axis to group"
                );
                //assert!(not self.tensor_core, "can't group with tensor cores");
                self.shift_to(
                    axis,
                    amt,
                    Some(opt.op == GROUPTOP),
                    Some(self.first_reduce() + self.group_for_reduce.len() as isize),
                );
                self.group_for_reduce.push(amt);
            }
            UNROLL => {
                assert!(
                    axis < self.shape_len() - self.upcasted,
                    "can't upcasted already upcasted"
                );
                assert!(amt <= 32, "don't unroll more than 32");
                self.shift_to(axis, amt, None, None);
                self.upcast();
            }
            UPCAST => {
                assert!(axis < self.first_reduce(), "upcast is for non-reduce");
                assert!(amt <= 8, "don't upcast more than 8");
                self.shift_to(axis, amt, None, None);
                self.upcast();
            }
            UPCASTMID => {
                let axes = self.sts[0].unit_stride_axes(false);
                assert!(axes.len() == 1, "wrong number of stride 1 axis: {axis:?}");
                assert!(axes[0] == axis, "wrong axis");
                assert!(amt == 4, "don't upcast mid anything but 4");
                self.shift_to(
                    axis,
                    amt,
                    None,
                    Some(self.first_reduce() + self.group_for_reduce.len() as isize),
                );
                self.group_for_reduce.push(amt);
            }
            NOLOCALS => {
                assert!(self.opts.has_local && self.dont_use_locals, "NOLOCALS is meaningless if target does not support local or already not using locals");
                assert!(
                    self.local_dims == 0 && self.group_for_reduce.len() == 0,
                    "can't have no locals with locals"
                );
                self.dont_use_locals = true;
            }
            PADTO => {
                assert!(axis < self.first_reduce(), "cannot pad a reduce axis");
                let mut padded = false;
                for (i, st) in self.sts.iter_mut().enumerate() {
                    assert!(
                        st.shape()[axis] > amt / 2,
                        "pad adds more than double the work"
                    );
                    let ru = roundup(st.shape()[axis], amt);
                    if ru - st.shape()[axis] > 0 {
                        *st = st.pad(
                            &vec![
                                vec![(0, 0); axis.to_usize().unwrap_or(0)],
                                vec![(0, ru)],
                                vec![
                                    (0, 0);
                                    (st.shape().len() as isize - axis - 1)
                                        .to_usize()
                                        .unwrap_or(0)
                                ],
                            ]
                            .concat(),
                        );
                        padded = true;
                    }
                }
                assert!(padded, "nothing was padded");
            }
            _ => panic!(),
        }
        self.simplify_ones();
    }

    pub fn upcast(&mut self) {
        assert!(
            self.full_shape()[-1isize] != 1,
            "can't upcast a dimension with size 1"
        );
        self.upcasted += 1;
    }

    pub fn shift_to(
        &mut self,
        axis: isize,
        amount: isize,
        top: Option<bool>,
        insert_before: Option<isize>,
    ) {
        let top = top.unwrap_or(false);
        let mut insert_before = insert_before.unwrap_or(self.shape_len());
        let move_axis = if top { axis } else { axis + 1 };
        if move_axis < insert_before {
            insert_before += 1;
        }
        self.reshape_and_permute(
            Some(Box::new(move |x: Vec<isize>| -> Vec<isize> {
                let axis = if axis < 0 {
                    (axis + x.len() as isize) as usize
                } else {
                    axis as usize
                };
                vec![
                    x[..axis].to_vec(),
                    if x[axis] > 1 {
                        if top {
                            vec![amount, x[axis] / amount]
                        } else {
                            vec![x[axis] / amount, amount]
                        }
                    } else {
                        vec![1, 1]
                    },
                    x[axis + 1..].to_vec(),
                ]
                .concat()
            })),
            Some(
                vec![
                    v![i, for i in 0..insert_before, if i != move_axis],
                    vec![move_axis],
                    v![i, for i in insert_before..self.shape_len()+1, if i != move_axis],
                ]
                .concat(),
            ),
        );
    }

    pub fn float4_axis(&self, i: isize) -> Vec<isize> {
        let i = if i < 0 {
            self.sts.len() as isize + i
        } else {
            i
        } as usize;
        v![x-(self.shape_len()-self.upcasted), for x in self.sts[i].unit_stride_axes(false), if x >= self.shape_len()-self.upcasted && self.sts[i].shape()[x]%4 == 0]
    }

    pub fn upcast_in_mid_reduce_axes(&self) -> Vec<isize> {
        v![j, for j in self.first_reduce()..self.group_for_reduce.len() as isize, if self.full_shape()[j] == self.sts[0].shape()[j]]
    }

    pub fn shape_offsets(&self, i: isize) -> Vec<Vec<isize>> {
        //def shape_offsets(self, i:int): return itertools.product(*[list(range(cast(int, s))) for s in self.sts[i].shape[self.shape_len-self.upcasted:][::-1]]) if self.upcasted > 0 else [tuple()]  # noqa: E501
        let i = if i < 0 {
            self.sts.len() as isize + i
        } else {
            i
        } as usize;
        if self.upcasted > 0 {
            cartesian_product(
                v![v![i, for i in 0..*s], for s in self.sts[i].shape()[self.shape_len()-self.upcasted..].into_iter().rev()],
            )
        } else {
            vec![vec![]]
        }
    }

    pub fn hand_coded_optim(&mut self) {
        use OptOps::*;
        let MV_BLOCKSIZE = getenv("MV_BLOCKSIZE", 4);
        let MV_THREADS_PER_ROW = getenv("MV_THREADS_PER_ROW", 8);
        let MV_ROWS_PER_THREAD = getenv("MV_ROWS_PER_THREAD", 4);

        // if self.opts.has_local and getenv("MV",1) != 0 and
        // (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and
        // self.reduceop and self.reduceop.op == ReduceOps.SUM and len(self.full_shape) >= 2 and
        // self.opts.has_shared and (mulop:=self.reduceop.src[0]).op == BinaryOps.MUL and mulop.src[0].op == BufferOps.LOAD and mulop.src[1].op == BufferOps.LOAD:
        if self.opts.has_local
            && getenv("NV", 1) != 0
            && (MV_BLOCKSIZE > 1 || MV_THREADS_PER_ROW > 1 || MV_ROWS_PER_THREAD > 1)
            && self
                .reduceop
                .as_ref()
                .is_some_and(|r| r.optype == Reduce::Sum)
            && self.full_shape().len() >= 2
            && self.opts.has_share
        {
            let mulop = &self.reduceop.as_ref().unwrap().src[0];
            if !(mulop.optype() == Binary::Mul
                && mulop.src()[0].optype() == ops::Buffer::Load
                && mulop.src()[1].optype() == ops::Buffer::Load)
            {
                return;
            }
            fn has_expanded_axis(shape: &[isize], strides: &[Option<isize>]) -> bool {
                v![*s >1 && st.is_some_and(|n| n==0), for (s, st) in izip!(shape, strides)]
                    .iter()
                    .any(|&s| s)
            }
            let mulop_src = mulop.src();
            let st0 = &self.sts[self
                .bufs
                .iter()
                .position(|a| a.eq(&mulop_src[0].lo().args[0].to_buf()))
                .unwrap()];
            let st1 = &self.sts[self
                .bufs
                .iter()
                .position(|a| a.eq(&mulop_src[1].lo().args[0].to_buf()))
                .unwrap()];
            let strides0: Vec<Option<isize>> = st0.real_strides(false);
            let strides1: Vec<Option<isize>> = st1.real_strides(false);
            if strides0[self.first_reduce() as usize].is_some_and(|n| n == 1)
                && !(has_expanded_axis(&st0.shape().dims, &strides0))
                && has_expanded_axis(&st1.shape().dims, &strides1)
            {
                for global_idx in 0..self.global_dims() {
                    if self.full_shape()[self.first_reduce()] % MV_THREADS_PER_ROW == 0
                        && self.full_shape()[global_idx] % (MV_BLOCKSIZE * MV_ROWS_PER_THREAD) == 0
                    {
                        if MV_THREADS_PER_ROW > 1 {
                            self.apply_opt(Opt {
                                op: GROUP,
                                axis: Some(0),
                                amt: Some(MV_THREADS_PER_ROW),
                            });
                        }
                        if MV_BLOCKSIZE > 1 {
                            self.apply_opt(Opt {
                                op: LOCAL,
                                axis: Some(global_idx),
                                amt: Some(MV_BLOCKSIZE),
                            });
                        }
                        if MV_ROWS_PER_THREAD > 1 {
                            self.apply_opt(Opt {
                                op: UPCAST,
                                axis: Some(global_idx),
                                amt: Some(MV_ROWS_PER_THREAD),
                            });
                        }
                        return;
                    }
                }
            }
        }

        //TODO: OPENCL -54(CL_DEVICE_NOT_FOUND) error when using barrier
        if DEVICE.name() != "OPENCL" {
            if self.opts.has_local && self.opts.has_share {
                if self.float4_axis(0).len() == 0
                    && self.first_reduce() <= 2
                    && self.first_reduce() + 1 <= self.shape_len()
                    && prod(&self.sts[0].shape()[..self.first_reduce()]) <= 2048
                {
                    for sz in if prod(&self.sts[0].shape()[..self.first_reduce()]) <= 32 {
                        vec![256, 16]
                    } else {
                        vec![16]
                    } {
                        if all(
                            &v![st.shape()[self.first_reduce()] % sz == 0 || st.shape()[self.first_reduce()] == 1, for st in self.sts.iter()],
                        ) {
                            self.apply_opt(Opt {
                                op: GROUPTOP,
                                axis: Some(0),
                                amt: Some(sz),
                            });
                            break;
                        }
                    }
                }
            }

            if self.group_for_reduce.len() > 0 {
                return;
            }
        }

        let mut to_upcast: Vec<isize> = vec![];
        for axis in 0..self.first_reduce() {
            if self.full_shape()[axis] <= 7
                && any(&v![st.axis_is_masked(axis), for st in self.sts.iter()])
                && prod(&self.full_shape()[self.shape_len() - self.upcasted..])
                    * prod(&v![self.full_shape()[*j], for j in to_upcast.iter()])
                    * self.full_shape()[axis]
                    <= 7 * 7
            {
                to_upcast.push(axis);
            }
        }

        for axis in to_upcast.iter().rev() {
            self.apply_opt(Opt {
                op: UPCAST,
                axis: Some(*axis),
                amt: Some(0),
            })
        }

        let mut upcasted_axis: HashSet<isize> = HashSet::new();
        while prod(&self.sts[0].shape()[..self.first_reduce()]) >= 1024 {
            let mut xb_choices = vec![];
            for x in cartesian_product(vec![v![i, for i in 0..self.first_reduce()], vec![3, 4]]) {
                let [axis, upcast_amount] = x[..] else {
                    panic!("{:?}", x)
                };
                if !upcasted_axis.contains(&axis)
                    && self.full_shape()[axis] % upcast_amount == 0
                        //any(st.views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index, st in enumerate(self.sts)):  # noqa: E501
                    && any(
                        &v![st.views.last().as_ref().unwrap().strides[axis as usize] == 0 && !any(&v![x.1.is_some_and(|s| s == 0), for x in self.upcasted_axis(buf_index as isize)]), for (buf_index, st) in self.sts.iter().enumerate()],
                    )
                {
                    xb_choices.push(
                        (
                            v![if st.views[st.views.len()-1].strides[axis as usize]>0 { 1 } else { 0 } , for st in self.sts.iter()].iter().sum::<isize>(),
                            v![st.views[st.views.len()-1].strides[axis as usize], for st in self.sts.iter()].iter().sum::<isize>(),
                            axis,
                            upcast_amount
                        )
                    )
                }
            }
            if xb_choices.len() > 0 {
                xb_choices.sort();
                self.apply_opt(Opt {
                    op: UPCAST,
                    axis: Some(xb_choices[0].2),
                    amt: Some(xb_choices[0].3),
                });
                upcasted_axis.insert(xb_choices[0].2);
            } else {
                break;
            }
        }

        //if self.first_reduce < (self.shape_len-self.upcasted) and
        //(len(list(self.shape_offsets(self.full_buf_index))) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))) and
        //(self.upcasted == 0 or prod(self.full_shape[-self.upcasted:]) < 64):  # noqa: E501

        if self.first_reduce() < (self.shape_len() - self.upcasted)
            && (self.shape_offsets(self.full_buf_idx as isize).len() <= 4
                || !any(&v![r.2 > 0, for r in self.upcasted_axis(self.full_buf_idx as isize)]))
            && (self.upcasted == 0 || prod(&self.full_shape()[-self.upcasted..]) < 64)
        {
            let s1 = self.full_unupcasted_shape()[-1];
            if s1 <= 32isize {
                self.apply_opt(Opt {
                    op: UNROLL,
                    axis: Some(
                        self.full_unupcasted_shape().len() as isize - 1 - self.first_reduce(),
                    ),
                    amt: Some(0),
                });
                if self.first_reduce() < (self.shape_len() - self.upcasted)
                    && s1 <= 3isize
                    && self.full_unupcasted_shape()[-1] <= 3isize
                {
                    self.apply_opt(Opt {
                        op: UNROLL,
                        axis: Some(
                            self.full_unupcasted_shape().len() as isize - 1 - self.first_reduce(),
                        ),
                        amt: Some(0),
                    });
                }
            } else {
                for splits in [4] {
                    if self.full_unupcasted_shape()[-1] % splits == 0isize {
                        self.apply_opt(Opt {
                            op: UNROLL,
                            axis: Some(
                                self.full_unupcasted_shape().len() as isize
                                    - 1
                                    - self.first_reduce(),
                            ),
                            amt: Some(splits),
                        });
                    }
                }
            }
        }

        for splits in [4] {
            if self.upcasted == 0
                && self.full_unupcasted_shape().len() > 0
                && self.full_unupcasted_shape()[-1] % splits == 0isize
            {
                self.apply_opt(Opt {
                    op: UPCAST,
                    axis: Some(self.full_unupcasted_shape().len() as isize - 1),
                    amt: Some(splits),
                });
            }
        }

        //         && self.local_dims == 0
        // if self.opts.has_local {
        //     if getenv("NOLOCALS", 0) == 1
        //         && self.group_for_reduce.is_empty()
        //     {
        //         self.apply_opt(Opt {
        //             op: NOLOCALS,
        //             axis: None,
        //             amt: None,
        //         })
        //     } else {
        //         //local_axis_ranking = [(any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))), axis) for axis in range(len(self.full_shape[:self.first_reduce]))]  # noqa: E501
        //         let local_axis_ranking = v![(if any(&v![self.sts[buf_index].views.last().as_ref().unwrap().strides[axis] == 0, for buf_index in 0..self.sts.len()]) { 1isize } else { 0isize }, axis as isize), for axis in 0..self.full_shape()[..self.first_reduce()].len()];
        //         let mut sorted_local_axis_ranking = local_axis_ranking.clone();
        //         sorted_local_axis_ranking.sort_by_cached_key(|x| (-x.0, -x.1));
        //         let mut to_local: Vec<(isize, isize)> = vec![];
        //         for (_, axis) in sorted_local_axis_ranking {
        //             let local_size = prod(&v![*sz, for (_, sz) in to_local.iter()]);
        //             //local_sz: Optional[int] = next((x for x in ([32] * (axis == 0) + [16, 8, 4, 3, 2]) if self.full_shape[axis] % x == 0 and local_size * x <= 128), None)  # noqa: E501
        //             let local_sz = v![x, for x in vec![vec![32;if axis == 0isize { 1 } else { 0 }], vec![16, 8, 4, 3, 2]].concat(), if self.full_shape()[axis] % x == 0 && local_size * x <= 128];
        //             if local_sz.len() > 0 {
        //                 to_local.push((axis, local_sz[0]));
        //             }
        //         }
        //         let mut deleted_shape = 0;
        //         for (mut axis, local_sz) in to_local[..to_local.len().min(3)].iter().sorted() {
        //             axis -= deleted_shape;
        //             let will_deleted_shape = *local_sz == self.full_shape()[axis];
        //             println!("{} {}", axis, local_sz);
        //             self.apply_opt(Opt {
        //                 op: LOCAL,
        //                 axis: Some(axis),
        //                 amt: Some(*local_sz),
        //             });
        //             if will_deleted_shape {
        //                 deleted_shape += 1;
        //             }
        //         }
        //     }
        // }
    }

    pub fn full_unupcasted_shape(&self) -> Shape {
        self.full_shape()[..self.shape_len() - self.upcasted]
            .to_vec()
            .into()
    }
}
