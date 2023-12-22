#![allow(non_snake_case)]
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::sync::Mutex;

use crate::arg::Arg;
use crate::device::opencl::CLRenderer;
use crate::dtype::{_bool, float16, float32, int32};
use crate::lazy::STArc;
use crate::ops::{Binary, Ternary, Unary};
use crate::prelude::*;
use crate::renderer::cstyle::uops_to_cstyle;
use crate::shape::symbolic::{num, var, ArcNode};
use crate::tensor::shape::Shape;
use crate::{
    dtype,
    lazy::{LOArc, LazyBuffer},
    ops::{self, LazyOp, OpType, DEVICE},
    shape::ShapeTracker,
};

use super::linearizer::{UOp, UOps};

pub enum ConstNum {
    Int(i32),
    Float(f32),
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
pub struct LinearizerOptions {
    support_float4: bool,
    support_float4_alu: bool,
    has_local: bool,
    global_max: Option<Vec<isize>>,
    local_max: Option<Vec<isize>>,
}

impl Default for LinearizerOptions {
    fn default() -> Self {
        Self {
            support_float4_alu: false,
            support_float4: false,
            has_local: true,
            global_max: None,
            local_max: None,
        }
    }
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
        println!("\nfull_buf_idx: {full_buf_idx} kenrel bufs {bufs:?}\n");
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
        println!("\nafter reshape_and_permute >>> {:?}\n", ret.sts);
        // # parameters for optimization
        // self.applied_opts: List[Opt] = []
        // self.local_alias: Dict[int, LocalBuffer] = {}
        // self.tensor_core: Optional[TensorCore] = None
        // self.dont_use_locals: bool = False
        ret.simplify_ones();
        println!("\nsimplify_ones >>> {:?}\n", ret.sts);
        ret.simplify_merge_adjacent();
        println!("\nsimplify_merge_adjacent  >>> {:?}\n", ret.sts);
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

    fn shape_len(&self) -> isize {
        self.sts[0].shape().len() as isize
    }

    fn full_shape(&self) -> Vec<isize> {
        self.sts[self.full_buf_idx].shape()
    }

    fn first_reduce(&self) -> isize {
        if self.shape_len() < self.upcasted {
            return 0;
        }
        v![x!=y, for (x,y) in izip!(self.sts[0].shape()[..(self.shape_len()-self.upcasted) as usize].iter(),self.full_shape()[..(self.shape_len()-self.upcasted)as usize].iter())]
            .iter()
            .position(|&t| t == true)
            .unwrap_or((self.shape_len()-self.upcasted) as usize) as isize
    }

    fn simplify_ones(&mut self) -> bool {
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

    fn simplify_merge_adjacent(&mut self) {
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
                        (rets[j].last().unwrap().0 * shapes[j][j], strides[j][i]);
                } else {
                    rets[j].push((shapes[j][i], strides[j][i]));
                }
            }
        }

        for (i, x) in rets[..self.sts.len()].iter().enumerate() {
            self.sts[i] = self.sts[i].reshape(&v![y.0, for y in x]);
        }
    }

    fn global_dims(&self) -> isize {
        self.first_reduce() - self.local_dims
    }

    fn upcasted_axis(&self, i: isize) -> Vec<Vec<isize>> {
        todo!()
    }

    fn output_shape(&self) -> Vec<isize> {
        self.sts[0].shape()
    }
}

#[derive(Debug)]
pub struct Linearizer {
    // ast: LazyOp,
    // opts: LinearizerOptions,
    // bufs: Vec<LazyBuffer>,
    // reduceop: Option<LazyOp>,
    // earlybufs: Vec<LazyBuffer>,
    // full_buf_idx: usize,
    // sts: Vec<ShapeTracker>, // This might need deepclone
    // local_dims: isize,
    // upcasted: isize,
    // group_for_reduce: Vec<isize>,
    pub kernel: Kernel,
    pub uops: Vec<UOp>,
    pub buf_ops: Vec<Option<UOp>>,
    pub loop_ops: HashMap<String, UOp>,
    pub name: String,
    pub saved_exprs: HashMap<(UOps, Dtype, Vec<UOp>, Vec<Arg>), UOp>,
    pub global_size: Option<Vec<usize>>,
    pub local_size: Option<Vec<usize>>,
}

impl Linearizer {
    pub fn new(ast: LazyOp, opts: Option<LinearizerOptions>) -> Self {
        Self {
            kernel: Kernel::new(ast, opts),
            uops: Default::default(),
            buf_ops: Default::default(),
            loop_ops: Default::default(),
            name: "".into(),
            saved_exprs: HashMap::new(),
            global_size: None,
            local_size: None,
        }
    }

    pub fn linearize(&mut self) {
        // # no new opts and we already ran? skip relinearizing
        //   if self.applied_opts == self.applied_opts_cache: return self
        let mut sts_backup = self.kernel.sts.clone();
        let mut gfr_backup = self.kernel.group_for_reduce.clone();
        let mut upc_backup = self.kernel.upcasted;
        self.saved_exprs = HashMap::new();
        self.uops = vec![];
        self.buf_ops = vec![None; self.kernel.bufs.len()];
        self.loop_ops = HashMap::new();
        for i in 0..self.kernel.bufs.len() {
            let buf = &self.kernel.bufs[i];
            if let Buffers::MemBuffer(buffer) = buf {
                let a = Arg::Str(format!("data{}", buffer.idx));
                let uop = self.uop_default(
                    UOps::DEFINE_GLOBAL,
                    Some(buffer.dtype.clone()),
                    vec![],
                    vec![a],
                );
                self.buf_ops[i] = Some(uop);
            }
        }
        // # add var vals
        // for var in vars_from_ast(self.ast):
        //   assert var.expr is not None
        //   self.loop_uops[var.expr] = self.uop(UOps.DEFINE_GLOBAL, dtypes.int32, (), var
        // for lb in self.local_alias.values():
        //   self.buf_uops[self.bufs.index(lb)] = self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), (lb.name, self.sts[self.bufs.index(lb)].size()))

        // self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+len(self.group_for_reduce)]) + [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        // self.bufs.append(LocalBuffer("temp", self.sts[-1].size()))
        // self.buf_uops.append(self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), ("temp", self.sts[-1].size())))
        //
        // self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[
        // self.global_dims:self.global_dims+self.local_dims+len(self.group_for_reduce)
        // ]) + [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce)
        // + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        if !self.kernel.group_for_reduce.is_empty() {
            self.kernel.sts.push(ShapeTracker::from_shape(
                &vec![
                    &vec![1; self.kernel.global_dims() as usize],
                    &self.kernel.full_shape()[self.kernel.global_dims() as usize
                        ..(self.kernel.global_dims() + self.kernel.local_dims) as usize
                            + self.kernel.group_for_reduce.len()],
                    &vec![
                        1;
                        (self.kernel.shape_len()
                            - self.kernel.upcasted
                            - self.kernel.group_for_reduce.len() as isize)
                            as usize
                    ],
                    &v![x[0].into(), for x in self.kernel.upcasted_axis(0)],
                ]
                .concat(),
            ));
            self.kernel.bufs.push(
                LocalBuffer {
                    name: "tmp".into(),
                    size: self.kernel.sts.last().unwrap().size() as usize - 1,
                    dtype: float32,
                    realized: None,
                }
                .into(),
            );
            //self.buf_uops.append(self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), ("temp", self.sts[-1].size())))

            let a1 = Arg::Str("temp".into());
            let a2 = Arg::Usize(self.kernel.sts.last().unwrap().size() as usize);
            let buf_uop = self.uop_default(UOps::DEFINE_LOCAL, Some(float32), vec![], vec![a1, a2]);
            self.buf_ops.push(Some(buf_uop))
        }

        let name = if self.kernel.reduceop.is_some() {
            "r_".to_string()
        } else {
            "E_".to_string()
        } + &v![x.to_string(), for x in self.kernel.full_shape()].join("_");

        // # name the function something unique
        let mut c = *KERNEL_CNT.lock().unwrap().entry(name.clone()).or_default();
        c += 1;
        let suffix = if c > 1 {
            "n".to_string() + &c.to_string()
        } else {
            "".to_string()
        };
        self.name = name + &suffix;
        let (global_idx, loop_global_idxs) = get_grouped_dims(
            "gidx",
            0,
            &self.kernel.full_shape()[..self.kernel.global_dims() as usize],
            if self.kernel.opts.has_local { 3 } else { 0 },
        );
        let (local_idxs, loop_local_idxs) = get_grouped_dims(
            "lidx",
            self.kernel.global_dims() as usize,
            &self.kernel.full_shape()[self.kernel.global_dims() as usize
                ..self.kernel.first_reduce() as usize + self.kernel.group_for_reduce.len()],
            if self.kernel.opts.has_local { 3 } else { 0 },
        );
        let full_upcast_idxs = v![var("", 0, s-1), for s in self.kernel.full_shape()[(self.kernel.shape_len()-self.kernel.upcasted) as usize..].iter()];
        let upcast_idxs = v![var("", 0, s-1), for s in self.kernel.output_shape()[(self.kernel.shape_len()-self.kernel.upcasted) as usize..].iter()];

        self.global_size = None;
        self.local_size = None;

        if self.kernel.dont_use_locals {
            self.global_size =
                Some(v![x.max().unwrap() as usize + 1, for x in loop_local_idxs.iter().rev()]);
            let mut extend_loop_uops = HashMap::new();
            for (i, x) in loop_global_idxs.iter().enumerate() {
                extend_loop_uops.insert(
                    x.expr().unwrap().to_string(),
                    self.uop_default(
                        UOps::SPECIAL,
                        Some(int32),
                        vec![],
                        vec![
                            Arg::Usize(loop_global_idxs.len() - 1 - i),
                            Arg::Str(x.expr().unwrap().replace("gidx", "idx")),
                            Arg::Usize(x.max().unwrap() as usize + 1),
                        ],
                    ),
                );
            }
            self.loop_ops.extend(extend_loop_uops);
        } else if self.kernel.opts.has_local {
            self.global_size =
                Some(v![(x.max().unwrap() + 1 ) as usize, for x in loop_global_idxs.iter().rev()]);
            self.local_size =
                Some(v![(x.max().unwrap() + 1 ) as usize, for x in loop_local_idxs.iter().rev()]);

            let mut extend_loop_uops = HashMap::new();
            for (i, x) in loop_global_idxs.iter().enumerate() {
                extend_loop_uops.insert(
                    x.expr().unwrap().to_string(),
                    self.uop_default(
                        UOps::SPECIAL,
                        Some(int32),
                        vec![],
                        vec![
                            Arg::Usize(loop_global_idxs.len() - 1 - i),
                            Arg::Str(x.expr().unwrap().to_string()),
                            Arg::Usize(x.max().unwrap() as usize + 1),
                        ],
                    ),
                );
            }

            let mut extend_loop_uops = HashMap::new();
            for (i, x) in loop_local_idxs.iter().enumerate() {
                extend_loop_uops.insert(
                    x.expr().unwrap().to_string(),
                    self.uop_default(
                        UOps::SPECIAL,
                        Some(int32),
                        vec![],
                        vec![
                            Arg::Usize(loop_local_idxs.len() - 1 - i),
                            Arg::Str(x.expr().unwrap().to_string()),
                            Arg::Usize(x.max().unwrap() as usize + 1),
                        ],
                    ),
                );
            }
        } else {
            self.render_loop(&vec![loop_global_idxs.clone(), loop_local_idxs.clone()].concat());
        }

        let mut loaded_buffser: HashMap<Buffers, Vec<UOp>> = HashMap::new();
        let mut acc: Vec<UOp> = vec![];
        let mut load_cache: HashMap<String, UOp> = HashMap::new();

        let mut fake_reduce_idxs: Vec<ArcNode> = vec![];

        println!(">>> KENREL: {:?}", self.kernel.ast);

        if let Some(reduceop) = &self.kernel.reduceop {
            let reduce_idxs = v![var(&format!("ridx{i}"), 0, self.kernel.full_shape()[i]-1), for i in self.kernel.first_reduce() as usize +self.kernel.group_for_reduce.len()..(self.kernel.shape_len()-self.kernel.upcasted) as usize];
            let fake_reduce_idxs = v![x*0, for x in reduce_idxs.iter()];
            let mut seen = HashSet::new();
            let idxs_tmp = vec![
                global_idx.clone(),
                local_idxs.clone(),
                fake_reduce_idxs.clone(),
                upcast_idxs.clone(),
            ]
            .concat();
            let mut idxs_gl = vec![];
            for (i, idx) in idxs_tmp.into_iter().enumerate() {
                if !seen.contains(&idx) {
                    idxs_gl.push(idx.clone());
                }
                seen.insert(idx);
            }
            let acc = self.global_load(
                0,
                idxs_gl,
                Some(get_reduce_acc(
                    reduceop.optype.clone(),
                    self.kernel.bufs[0].dtype(),
                )),
                None,
            );
            // if self.kernel.tensor_core.is_some() {
            // }

            let loop_ctx = self.render_loop(&reduce_idxs);

            //let mut locals_to_store = vec![];
            // TODO: local_alias i dont see it gets `appended` anyway
            //
            // for i in self.local_alias:
            //   localbuf_idx = self.bufs.index(self.local_alias[i])
            //   buf_idxs = [idx*0 if s == 0 else idx for idx,s in zip(global_idxs+local_idxs+reduce_idxs+full_upcast_idxs,self.sts[i].real_strides())]
            //   if self.tensor_core:
            //     min_alias_idx = min(self.local_alias.keys())
            //     replace_input_idxs = calc_tc_idxs(self.tensor_core.thread_local_sizes[i-min_alias_idx], self.tensor_core.thread_local_aliases[i-min_alias_idx])  # noqa: E501
            //     for n in range(len(self.tensor_core.threads)):
            //       buf_idxs[self.first_reduce-len(self.tensor_core.threads)+n] = replace_input_idxs[n] # replace locals
            //     for n in range(len(replace_input_idxs)-len(self.tensor_core.threads)):
            //       buf_idxs[self.shape_len-self.upcasted+n] = replace_input_idxs[len(self.tensor_core.threads)+n] # replace upcasts
            //   if DEBUG >= 3: print(f"{localbuf_idx} alias {i}: idxs=", buf_idxs)
            //   ll = self.global_load(i, buf_idxs)
            //   locals_to_store.append((localbuf_idx, buf_idxs, ll))

            //if self.kernel.tensor_core //TODO:

            if self.kernel.group_for_reduce.len() > 0 {
                let fake_global_idx = v![x*0, for x in global_idx.iter()];
            }
        }

        // println!("LOOP TOKENS: {:?}", loop_ctx);
        // println!(
        //     "RENDERED->: {:?}",
        //     uops_to_cstyle(CLRenderer::default().renderer, &self.name, loop_ctx)
        // );

        todo!(
            "{}\n{:?} {:?}\n{:?} {:?}\n{:?}\n{:?}\n",
            self.name,
            global_idx,
            loop_global_idxs,
            local_idxs,
            loop_local_idxs,
            full_upcast_idxs,
            upcast_idxs,
        );
    }

    pub fn _const(&mut self, val: String, dtype: Dtype, insert_before: Option<isize>) -> UOp {
        self.uop(
            UOps::CONST,
            Some(dtype),
            vec![],
            vec![Arg::Str(val)],
            true,
            insert_before,
            true,
        )
    }

    pub fn const_default(&mut self, val: String) -> UOp {
        let dtype = int32;
        let insert_before = None;
        self.uop(
            UOps::CONST,
            Some(dtype),
            vec![],
            vec![Arg::Str(val)],
            true,
            insert_before,
            true,
        )
    }

    pub fn render_loop(&mut self, xx: &[ArcNode]) -> Vec<UOp> {
        let mut new_loops = HashMap::new();
        for x in xx {
            if !x.is_num() && x.expr().is_some() {
                let min = self.const_default(x.min().unwrap().to_string());
                let max = self.const_default(x.max().unwrap().to_string());
                new_loops.insert(
                    x.expr().unwrap().to_string(),
                    self.uop_default(UOps::LOOP, Some(int32), vec![min, max], vec![]),
                );
            }
        }
        let ret = new_loops.values().map(|x| x.clone()).collect();
        self.loop_ops.extend(new_loops);
        ret
    }

    pub fn uop_default(
        &mut self,
        uop: UOps,
        dtype: Option<Dtype>,
        vin: Vec<UOp>,
        arg: Vec<Arg>,
    ) -> UOp {
        self.uop(uop, dtype, vin, arg, true, None, true)
    }

    pub fn uop(
        &mut self,
        mut uop: UOps,
        mut dtype: Option<Dtype>,
        mut vin: Vec<UOp>,
        mut arg: Vec<Arg>,
        mut cachable: bool,
        mut insert_before: Option<isize>,
        mut simplify: bool,
    ) -> UOp {
        println!("UOP called {:?}", uop);
        match uop {
            UOps::ALU => {
                //assert!(dtype.as_ref().unwrap() == &_bool);
                let upcast_type = if matches!(arg[0].to_op(), OpType::Ternary(Ternary::Mulacc)) {
                    float32
                } else {
                    //WARN: Should this be using jax types promo?
                    v![x.dtype.as_ref().unwrap().clone(), for x in vin.iter(), if x.dtype.is_some()]
                        .into_iter()
                        .max()
                        .unwrap()
                };
                dtype = Some(upcast_type);
            }
            _ => (),
        }
        if simplify {
            match uop {
                UOps::PHI if vin.len() == 2 => return vin[1].to_owned(),
                UOps::GEP if vin[0].uop == UOps::CONST => {
                    return self._const(vin[0].args[0].to_str(), dtype.unwrap(), insert_before)
                }
                UOps::CAST => {
                    if v![0, for x in vin.iter(), if x.uop == UOps::CONST].len() > 0
                        && v![0,  for x in vin.iter(), if &vin[0] == x].len() > 1
                    {
                        return self._const(vin[0].args[0].to_str(), dtype.unwrap(), insert_before);
                    }
                }
                UOps::ALU => {
                    let op = arg[0].to_op();
                    if op == Binary::Add
                        && vin[1].uop == UOps::ALU
                        && vin[1].args[0].to_op() == Unary::Neg
                    {
                        //return self.uop(UOps.ALU, dtype, (vin[0], vin[1].vin[0]), BinaryOps.SUB, cachable=cachable, insert_before=insert_before)
                        return self.uop(
                            UOps::ALU,
                            dtype,
                            vec![vin[0].to_owned(), vin[1].vin[0].to_owned()],
                            vec![Arg::OpType(OpType::Binary(Binary::Sub))],
                            cachable,
                            insert_before,
                            simplify,
                        );
                    }
                    if op == Unary::Neg && vin[0].uop == UOps::CONST {
                        return self._const(
                            "-".to_string() + &vin[0].args[0].to_str(),
                            dtype.unwrap(),
                            insert_before,
                        );
                    }
                    if op == Ternary::Where && vin[1] == vin[2] {
                        return vin[1].to_owned();
                    }

                    for x in 0..1 {
                        if op == Binary::Add
                            && vin[x].uop == UOps::CONST
                            && &vin[x].args[0].to_str() == "0.0"
                        {
                            return vin[1 - x].to_owned();
                        }
                        if op == Binary::Mul
                            && vin[x].uop == UOps::CONST
                            && &vin[x].args[0].to_str() == "1.0"
                        {
                            return vin[1 - x].to_owned();
                        }
                        if op == Binary::Mul
                            && vin[x].uop == UOps::CONST
                            && &vin[x].args[0].to_str() == "0.0"
                        {
                            return vin[x].to_owned();
                        }
                    }
                    if op == Binary::Sub
                        && vin[1].uop == UOps::CONST
                        && &vin[1].args[0].to_str() == "0.0"
                    {
                        return vin[0].to_owned();
                    }
                    if op == Binary::Div
                        && vin[1].uop == UOps::CONST
                        && &vin[1].args[0].to_str() == "1.0"
                    {
                        return vin[0].to_owned();
                    }
                }
                _ => (),
            }
        }
        //TODO: this could be really expansive.
        let ret = UOp {
            uop: uop.clone(),
            dtype: dtype.clone(),
            vin: vin.clone(),
            args: arg.clone(),
        };
        let key = &(uop, dtype.unwrap(), vin, arg);
        if let Some(expr) = self.saved_exprs.get(key) {
            if cachable
                && (insert_before.is_none()
                    || self
                        .uops
                        .iter()
                        .position(|e| e == expr)
                        .is_some_and(|i| i as isize <= insert_before.unwrap()))
            {
                return expr.to_owned();
            }
        };
        if let Some(i) = insert_before {
            self.uops.insert(i as usize, ret.clone());
        };
        if cachable {
            self.saved_exprs.insert(key.clone(), ret.clone());
        }
        ret
    }

    fn global_load(
        &self,
        i: usize,
        idxs: Vec<ArcNode>,
        acc: Option<ConstNum>,
        barrier: Option<UOp>,
    ) -> Vec<UOp> {
        let buf = &self.kernel.bufs[i];
        let localtype = buf.dtype();
        let const_ = if let Buffers::ConstBuffer(acc) = buf {
            if localtype.is_int() {
                ConstNum::Int(acc.val.parse::<i32>().unwrap())
            } else {
                ConstNum::Float(acc.val.parse::<f32>().unwrap())
            }
        } else {
            acc.unwrap()
        };

        let mut amt = 1;
        let mut dim: Option<isize> = None;
        let upcast_dim = self.get_upcast_dim(i);
        let float4_expand = idxs[0].expand(None);
        if upcast_dim.len() == 1 && (float4_expand.len() == 4 || float4_expand.len() == 2) {
            dim = Some(upcast_dim[0]);
            amt = float4_expand.len();
        }

        println!("\nidxs >>>> {:?}", idxs);
        let expand_vars = v![rename_var(idx.expand_idx(), &format!("_uidx{j}")), for (j, idx) in idxs.iter().enumerate()];
        println!("expand_vars >>>> {:?}", expand_vars);
        let fake_idxs = v![idx.substitute(&HashMap::from([(idx.expand_idx(), ev.clone())])), for (idx, ev) in izip!(idxs.iter(), expand_vars.iter())];
        println!("fake_idxs >>>> {:?}", fake_idxs);
        let (mut g_idx, mut g_valid) = if let Some(d) = dim {
            let d = d as usize;
            let (mut gidx, mut gvalid) = self.kernel.sts[i].expr_idxs(Some(
                vec![
                    &fake_idxs[..d],
                    &[float4_expand[0].clone()],
                    &fake_idxs[d + 1..],
                ]
                .concat(),
            ));
            if (&gidx / &num(amt as isize) * num(amt as isize)).render_default()
                != gvalid.render_default()
            {
                (gidx, gvalid) = self.kernel.sts[i].expr_idxs(Some(fake_idxs.clone()));
                amt = 1;
                dim = None;
            }
            (gidx, gvalid)
        } else {
            self.kernel.sts[i].expr_idxs(Some(fake_idxs.clone()))
        };
        if amt > 1 {
            //TODO: localtype.vectorize()
        }
        println!("g_idx >>>> {:?}", g_idx);
        println!("g_vlid >>>> {:?}", g_valid);
        todo!()
    }

    fn global_store(&self, i: usize, idxs: Vec<ArcNode>, store: Option<UOp>) -> Vec<UOp> {
        let buf = &self.kernel.bufs[i];
        let buf_uop = &self.buf_ops[i];
        assert!(buf_uop.is_some(), "buffer {i} wasn't UOped");
        let expanded_node = v![idx.nodes(), for idx in idxs.iter()];
        todo!()
    }

    fn get_upcast_dim(&self, i: usize) -> Vec<isize> {
        let should_upcast = self.kernel.opts.support_float4
            && matches!(self.kernel.bufs[i].dtype(), float32 | float16);
        v![x, for x in self.kernel.sts[i].unit_stride_axes(false), if should_upcast && x >= self.kernel.shape_len()-self.kernel.upcasted && Shape::from(self.kernel.sts[i].shape())[x] > 1]
    }
}

lazy_static::lazy_static! {
    pub static ref KERNEL_CNT: Mutex<HashMap<String, usize>> = Default::default();
}

fn get_grouped_dims(
    prefix: &str,
    start_dim: usize,
    local_dims: &[isize],
    maxdim: usize,
) -> (Vec<ArcNode>, Vec<ArcNode>) {
    let tmp = if local_dims.len() > maxdim {
        let mut tmp = local_dims[0..maxdim - 1].to_vec();
        tmp.push(local_dims[maxdim - 1..].iter().product::<isize>());
        tmp
    } else {
        vec![]
    };
    let mut local_idxs = v![var(&format!("{}{}",prefix, start_dim+i), 0, s-1), for (i, s) in if local_dims.len() > maxdim { &tmp } else {local_dims}.iter().enumerate()];
    let loop_local_idxs = local_idxs.clone();
    if maxdim != 0 && local_dims.len() > maxdim {
        let mut dd = local_idxs[maxdim].clone();
        let mut nli = vec![];
        for s in local_dims[maxdim - 1..].iter().rev() {
            nli.push(dd.clone() % num(*s));
            dd = dd / num(*s);
        }
        local_idxs = local_idxs[0..maxdim - 1].to_vec();
        local_idxs.extend(nli.into_iter().rev());
    }
    (local_idxs, v![x, for x in loop_local_idxs, if !x.is_num()])
}

pub fn get_reduce_acc(op: OpType, dtype: Dtype) -> ConstNum {
    if op == ops::Reduce::Sum {
        return if dtype.is_float() {
            ConstNum::Float(0.0)
        } else {
            ConstNum::Int(0)
        };
    };
    if op == ops::Reduce::Max {
        return if dtype.is_float() {
            ConstNum::Float(-f32::MAX)
        } else {
            ConstNum::Int(-i32::MAX)
        };
    };
    unreachable!();
}

pub fn rename_var(v: ArcNode, expr: &str) -> ArcNode {
    if v.is_num() {
        return v;
    }
    var(expr, v.min().unwrap(), v.max().unwrap())
}
