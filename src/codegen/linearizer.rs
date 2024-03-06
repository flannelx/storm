use itertools::Itertools;

use crate::dtype::{_bool, float16, float32, int32};
use crate::shape::shapetracker::strides_for_shape;
use crate::{ops, prelude::*};
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use std::sync::Arc;

use crate::arg::Arg;
use crate::codegen::kernel::{Buffers, LocalBuffer, KERNEL_CNT};
use crate::ops::{Binary, LazyOp, LazyOpSrc, Reduce, Ternary, Unary};
use crate::shape::symbolic::{iter_idxs, none_var, num, var, ArcNode, NodeOp};
use crate::shape::ShapeTracker;
use crate::tensor::shape::Shape;
use crate::{dtype, lazy::LazyBuffer, ops::OpType};

use super::kernel::{ConstNum, Kernel, Opt};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct UOsId(pub(crate) usize);

pub(crate) fn uop_id() -> UOsId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UOsId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
unsafe impl Send for UOsId {}
unsafe impl Sync for UOsId {}

impl core::fmt::Display for UOsId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UOps {
    LOOP,
    END,
    SPECIAL,
    DEFINE_GLOBAL,
    DEFINE_LOCAL,
    DEFINE_ACC,
    LOAD,
    STORE,
    CONST,
    BARRIER,
    ALU,
    WMMA,
    CAST,
    GEP,
    PHI,
    IF,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct UOp {
    pub(crate) uop: UOps,
    pub(crate) dtype: Option<dtype::Dtype>,
    pub(crate) vin: Vec<UOp>,
    pub(crate) args: Vec<Arg>,
    pub(crate) id: UOsId,
}

// impl core::fmt::Display for UOp {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "{:.4} {:<20?}: {} {:<32?} {:?}",
//             self.num,
//             self.uop,
//             if self.dtype.is_some() {
//                 format!("{:?}", self.dtype.as_ref().unwrap())
//             } else {
//                 format!("{:<25}", "")
//             },
//             self.vin.iter().map(|x| x.num).collect::<Vec<usize>>(),
//             self.args
//         )
//     }
// }

#[derive(Clone, Debug)]
pub struct LinearizerOptions {
    pub support_float4: bool,
    pub support_float4_alu: bool,
    pub has_local: bool,
    pub has_share: bool,
    pub global_max: Option<Vec<isize>>,
    pub local_max: Option<Vec<isize>>,
}

impl Default for LinearizerOptions {
    fn default() -> Self {
        Self {
            support_float4_alu: false,
            support_float4: false,
            has_local: true,
            has_share: true,
            global_max: None,
            local_max: None,
        }
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
    pub buf_uops: Vec<Option<UOp>>,
    pub loop_uops: HashMap<String, UOp>,
    pub name: String,
    pub saved_exprs: HashMap<(UOps, Option<Dtype>, Vec<UOp>, Vec<Arg>), UOp>,
    pub global_size: Option<Vec<usize>>,
    pub local_size: Option<Vec<usize>>,
    pub load_cache: HashMap<String, UOp>,
    pub applied_opts_cache: Vec<Opt>,
}

pub fn _limit_size(mut x: Vec<isize>, max_size: Vec<isize>) -> Vec<isize> {
    let mut new_shape = x;
    for i in 0..new_shape.len() {
        let mut next_idx = (i + 1) % new_shape.len();
        while new_shape[i] > max_size[i] {
            new_shape[i] /= 2;
            if new_shape[next_idx] > max_size[next_idx] {
                next_idx = (next_idx + 1) % new_shape.len();
            }
            new_shape[next_idx] *= 2;
        }
    }
    new_shape
}

impl Linearizer {
    pub fn new(ast: LazyOp, opts: Option<LinearizerOptions>) -> Self {
        Self {
            kernel: Kernel::new(ast, opts),
            uops: Default::default(),
            buf_uops: Default::default(),
            loop_uops: Default::default(),
            name: "".into(),
            saved_exprs: HashMap::new(),
            global_size: None,
            local_size: None,
            load_cache: HashMap::new(),
            applied_opts_cache: vec![],
        }
    }

    pub fn upcast(&mut self) {
        self.kernel.upcasted += 1;
    }

    pub fn limit_dims_to_max(&mut self, global_max: &[isize], local_max: &[isize]) {
        if self.kernel.global_dims() > 0 {
            let fl = self.kernel.full_shape().len();
            let g_dim = self.kernel.global_dims();
            if global_max.len() > 0 {
                let tmp = vec![
                    if g_dim > global_max.len() as isize {
                        vec![]
                    } else {
                        global_max[..self.kernel.global_dims() as usize].to_vec()
                    },
                    if local_max.len() > 0 {
                        local_max[..self.kernel.local_dims as usize].to_vec()
                    } else {
                        vec![]
                    },
                ]
                .concat();
                if global_max.iter().max()
                    < self.kernel.full_shape()[..self.kernel.global_dims()]
                        .iter()
                        .max()
                {
                    //self.reshape_and_permute(lambda x: self._limit_size(x, tmp + [math.inf] * (len(self.full_shape)-len(tmp))), None)
                    let tl = tmp.len();
                    self.kernel.reshape_and_permute(
                        Some(Box::new(move |x: Vec<isize>| {
                            _limit_size(x, vec![isize::MAX; fl - tl])
                        })),
                        None,
                    );
                }
                for i in 0..(self.kernel.global_dims() - 1) as usize {
                    if i < global_max.len() && self.kernel.full_shape()[i] > global_max[i] {
                        let mut order = (0..fl as isize).collect::<Vec<isize>>();
                        order.swap(i, (self.kernel.global_dims() - 1) as usize);
                        self.kernel.reshape_and_permute(None, Some(order));
                    }
                }
            }
        }
    }

    #[rustfmt::skip]
    pub fn linearize(&mut self) {
        if self.kernel.applied_opts.len() > 0 && self.kernel.applied_opts == self.applied_opts_cache {
            return;
        }
        let mut sts_backup = self.kernel.sts.clone();
        let mut gfr_backup = self.kernel.group_for_reduce.clone();
        let mut upc_backup = self.kernel.upcasted;
        self.saved_exprs = HashMap::new();
        self.uops = vec![];
        self.buf_uops = vec![None; self.kernel.bufs.len()];
        self.loop_uops = HashMap::new();

        // limit dims if need
        if let Some(gm) = &self.kernel.opts.global_max && let Some(lm) = &self.kernel.opts.local_max {
            let gm = gm.clone();
            let lm = lm.clone();
            self.limit_dims_to_max(&gm, &lm);
        }


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
                self.buf_uops[i] = Some(uop);
            }
        }
        // # add var vals
        // for var in vars_from_ast(self.ast):
        //   assert var.expr is not None
        //   self.loop_uops[var.expr] = self.uop(UOps.DEFINE_GLOBAL, dtypes.int32, (), var
        // for lb in self.local_alias.values():
        //   self.buf_uops[self.bufs.index(lb)] = self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), (lb.name, self.sts[self.bufs.index(lb)].size()))

        // self.sts.append(ShapeTracker.from_shape(tuple(
        // [1] * self.global_dims +
        // list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+len(self.group_for_reduce)]) +
        // [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce) +
        // [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        //
        // self.bufs.append(LocalBuffer("temp", self.sts[-1].size()))
        // self.buf_uops.append(self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), ("temp", self.sts[-1].size())))
        //
        // self.sts.append(ShapeTracker.from_shape(
        // tuple([1] * self.global_dims +
        // list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+len(self.group_for_reduce)]) +
        // [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce) +
        // [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        if self.kernel.group_for_reduce.len() > 0 {
            self.kernel.sts.push(ShapeTracker::from_shape(
                    &vec![
                        &vec![1; self.kernel.global_dims() as usize],
                        &self.kernel.full_shape()[self.kernel.global_dims()..self.kernel.global_dims() + self.kernel.local_dims + self.kernel.group_for_reduce.len() as isize],
                        &vec![1;(self.kernel.shape_len() - self.kernel.upcasted - self.kernel.group_for_reduce.len() as isize - self.kernel.first_reduce()) as usize],
                        &v![x.0, for x in self.kernel.upcasted_axis(0)],
                    ].concat(),
            ));
            let tmp_buf = LocalBuffer {
                name: "tmp".into(),
                size: self.kernel.sts.last().unwrap().size() as usize - 1,
                dtype: float32,
                realized: None,
            };
            self.kernel.bufs.push(tmp_buf.clone().into());
            let uop = self.uop(
                UOps::DEFINE_LOCAL,
                Some(float32),
                vec![],
                vec![Arg::Str("tmp".into()), Arg::Idx(self.kernel.sts[self.kernel.sts.len()-1].size())],
                true,
                None,
                true,
            );
            self.buf_uops.push(Some(uop));
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
            &self.kernel.full_shape()[..self.kernel.global_dims()],
            if self.kernel.opts.has_local { 3 } else { 0 },
        );
        //local_idxs, loop_local_idxs = get_grouped_dims("lidx", self.global_dims, self.full_shape[self.global_dims:self.first_reduce+len(self.group_for_reduce)], 3 if self.opts.has_local else 0)  # noqa: E501
        let (local_idxs, loop_local_idxs) = get_grouped_dims(
            "lidx",
            self.kernel.global_dims() as usize,
            &self.kernel.full_shape()[self.kernel.global_dims()
                ..self.kernel.first_reduce() + self.kernel.group_for_reduce.len() as isize],
            if self.kernel.opts.has_local { 3 } else { 0 },
        );
        let full_upcast_idxs = v![none_var(0, s-1), for s in self.kernel.full_shape()[(self.kernel.shape_len()-self.kernel.upcasted) as usize..].iter()];
        let mut upcast_idxs = v![none_var(0, s-1), for s in self.kernel.output_shape()[(self.kernel.shape_len()-self.kernel.upcasted) as usize..].iter()];

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
            self.loop_uops.extend(extend_loop_uops);
        } else if self.kernel.opts.has_local {
            self.global_size =
                Some(v![(x.max().unwrap() + 1 ) as usize, for x in loop_global_idxs.iter().rev()]);
            self.local_size = Some(
                v![(x.max().unwrap() + 1 ) as usize, for x in loop_local_idxs.iter().rev()],
            );

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
            self.loop_uops.extend(extend_loop_uops);

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
            self.loop_uops.extend(extend_loop_uops);
        } else {
            self.render_loop(&vec![loop_global_idxs.clone(), loop_local_idxs.clone()].concat());
        }

        let mut loaded_buffers: HashMap<Buffers, Vec<UOp>> = HashMap::new();
        let mut acc: Vec<UOp> = vec![];
        self.load_cache = HashMap::new();

        let mut fake_reduce_idxs: Vec<ArcNode> = vec![];

        if let Some(reduceop) = &self.kernel.reduceop {
            let optype = reduceop.optype.clone();
            let reduce_idxs = v![var(&format!("ridx{i}"), 0, self.kernel.full_shape()[i]-1), for i in self.kernel.first_reduce() as usize +self.kernel.group_for_reduce.len()..(self.kernel.shape_len()-self.kernel.upcasted) as usize];
            fake_reduce_idxs = v![x*0, for x in reduce_idxs.iter()];
            acc = self.global_load(
                0,
                vec![
                    global_idx.clone(),
                    local_idxs.clone(),
                    fake_reduce_idxs.clone(),
                    upcast_idxs.clone(),
                ]
                .concat(),
                Some(get_reduce_acc(
                    reduceop.optype.clone(),
                    self.kernel.bufs[0].dtype(),
                )),
                None,
            );
            // if self.kernel.tensor_core.is_some() {
            // }

            //println!("reduce idx len {}", reduce_idxs.len());
            let mut loop_ctx = self.render_loop(&reduce_idxs);

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
            //else if lcaols_to_store

            // Load early bufs
            //loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[i]) if i in self.local_alias else i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs})  # noqa: E501
            let iter_ = v![(i, b.clone()), for (i, b) in  self.kernel.bufs.iter().enumerate().skip(1), if self.kernel.earlybufs.contains(b)];

            //lb_ex = {b:self.global_load(i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs}
            let lb_ex = v![(b, self.global_load(i as isize, vec![global_idx.clone(), local_idxs.clone(), reduce_idxs.clone(), full_upcast_idxs.clone()].concat(), None, None)), for (i, b) in iter_];
            loaded_buffers.extend(lb_ex);
            // run early ast with reduce
            //panic!("self.kernel.reduceop {:?}", self.kernel.reduceop.as_ref().unwrap().optype);
            self.ast_parse(
                self.kernel.reduceop.clone().unwrap(),
                &mut acc,
                Some(&self.acc_offset(self.kernel.full_buf_idx as isize)),
                &loaded_buffers,
                true,
                Some(&loop_ctx),
                None,
            );

            self.load_cache.clear();

            if self.kernel.group_for_reduce.len() > 0 {
                let fake_global_idx = v![x*0, for x in global_idx.iter()];
                let stores = self.global_store(
                    -1,
                    vec![
                        fake_global_idx.clone(),
                        local_idxs.clone(),
                        fake_reduce_idxs.clone(),
                        upcast_idxs.clone(),
                    ]
                    .concat(),
                    acc.clone(),
                );
                let mut barrier = self.uop(UOps::BARRIER, None, stores, vec![], false, None, true);
                if self.kernel.opts.has_local {
                    let mut fake_idxs = v![num(0), for _ in 0..self.kernel.sts.last().as_ref().unwrap().shape().len()];
                    //fake_idxs[self.global_dims+self.local_dims:self.global_dims+len(local_idxs)] = local_idxs[self.local_dims:]
                    fake_idxs = vec![fake_idxs[..(self.kernel.global_dims()+self.kernel.local_dims) as usize].to_vec(), local_idxs[self.kernel.local_dims as usize..].to_vec(), fake_idxs[(self.kernel.global_dims()+local_idxs.len() as isize) as usize..].to_vec()].concat();
                    let if_cond = self.render((self.kernel.sts.last().as_ref().unwrap().expr_idxs(Some(fake_idxs)).0.lt(num(1))));
                    barrier = self.uop(UOps::IF, None, vec![if_cond, barrier], vec![],false, None, true);
                }
                //end_local_idxs = [Variable(f"tidx{i}", 0, self.full_shape[i]-1 if i >= self.first_reduce and i not in self.upcast_in_mid_reduce_axes else 0) for i in range(0, self.first_reduce+len(self.group_for_reduce))]  # noqa: E501
                let mut end_local_idxs = v![var(&format!("tidx{i}"), 0, if i >= self.kernel.first_reduce() && !self.kernel.upcast_in_mid_reduce_axes().contains(&i) { self.kernel.full_shape()[i]-1 } else { 0 }), for i in 0..self.kernel.first_reduce()+self.kernel.group_for_reduce.len() as isize];
                let mut local_idxs = vec![local_idxs[..self.kernel.local_dims as usize].to_vec(), end_local_idxs[(self.kernel.global_dims() + self.kernel.local_dims) as usize..].to_vec()].concat();

                for j in self.kernel.upcast_in_mid_reduce_axes() {
                    self.kernel.reshape_and_permute(None, Some(vec![v![i, for i in 0..self.kernel.shape_len(), if i != j], vec![j]].concat()));
                    self.upcast();
                    self.kernel.group_for_reduce.pop();
                    local_idxs.pop();
                    end_local_idxs.pop();
                    upcast_idxs = v![var(&format!("_uidx{i}"), 0, s-1), for (i, s) in self.kernel.output_shape()[(self.kernel.shape_len()-self.kernel.upcasted) as usize..].iter().enumerate()];
                }
                acc = self.global_load(-1, vec![fake_global_idx.clone(), local_idxs.clone(),fake_reduce_idxs.clone(),upcast_idxs.clone()].concat(), Some(get_reduce_acc(optype.clone(), float32)), None);
                loop_ctx = self.render_loop(&end_local_idxs);
                loaded_buffers.insert(self.kernel.bufs[self.kernel.bufs.len()-1].clone(), self.global_load(-1, vec![fake_global_idx.clone(), local_idxs.clone(),fake_reduce_idxs.clone(),upcast_idxs.clone()].concat(), None, Some(barrier)));
                self.ast_parse(LazyOp::new(optype.clone(), vec![LazyOp::new(OpType::Buffer(ops::Buffer::Load), vec![], Some(vec![Arg::Buffer(self.kernel.bufs[self.kernel.bufs.len()-1].clone())])).into()], None), &mut acc, Some(&self.acc_offset(-1)), &loaded_buffers, true, Some(&loop_ctx), None);
                self.load_cache.clear();
                local_idxs = vec![local_idxs[..self.kernel.local_dims as usize].to_vec(), v![num(0), for _ in 0..self.kernel.group_for_reduce.len()]].concat();
            }
        }

        // load late bufs
        let iter_ = v![(i, b.clone()), for (i, b) in  self.kernel.bufs.iter().enumerate(), if !self.kernel.earlybufs.contains(b) && i != 0 && !matches!(b, Buffers::LocalBuffer(_))];
        loaded_buffers.extend(v![(b, self.global_load(i as isize, vec![global_idx.clone(), local_idxs.clone(), fake_reduce_idxs.clone(), upcast_idxs.clone()].concat(), None, None)), for (i, b) in iter_]);
        let val = self.ast_parse(
            self.kernel.ast.src[0].lo().clone(),
            &mut acc,
            None,
            &loaded_buffers,
            false,
            None,
            None,
        );
        let val = self.global_store(
            0,
            vec![
                global_idx.clone(),
                local_idxs.clone(),
                fake_reduce_idxs.clone(),
                upcast_idxs.clone(),
            ]
            .concat(),
            val,
        );
        //println!("{:?}", val);
        // println!();
        // for v in val.iter() {
        //     println!("\n{v:?}");
        // }

        self.optimize_uops();

        // loop fix doesnt seem to work. so this is to push out anything that doesnt need to depend
        // on loop, will be replacing loop fix
        // let mut _loop = None;
        // let mut loop_child = HashSet::new();
        // let mut before_loop = vec![];
        // let mut after_loop = vec![];
        // 'out: for u in self.uops.iter() {
        //     if _loop.is_some() {
        //         let u_child = self.get_recursive_children(u);
        //         for u in u_child.iter() {
        //             if u.uop == UOps::LOOP {
        //                 continue 'out;
        //             }
        //         }
        //     }
        //     if u.uop == UOps::LOOP {
        //         _loop = Some(u);
        //         loop_child = self.get_recursive_children(u);
        //     } else if u.uop == UOps::END {
        //         _loop = None;
        //     }
        // }

        self.kernel.sts = sts_backup;
        self.kernel.group_for_reduce = gfr_backup;
        self.kernel.upcasted = upc_backup;

        // if self.uops.last().as_ref().unwrap().uop == UOps::END {
        //     let mut uu = vec![];
        //     for u in self.uops.iter().rev() {
        //     }
        //     panic!();
        // }

        self.applied_opts_cache = self.kernel.applied_opts.clone();
    }

    pub fn remove_childless_uops(&mut self) {
        let UOPS_W_SIDE_EFFECTS = HashSet::from([UOps::STORE, UOps::BARRIER, UOps::DEFINE_GLOBAL]);
        loop {
            let mut has_child = HashSet::new();
            for ru in self.uops.iter() {
                for vu in ru.vin.iter() {
                    has_child.insert(vu);
                }
            }
            let mut nu = v![x, for x in self.uops.iter(), if has_child.contains(x) || UOPS_W_SIDE_EFFECTS.contains(&x.uop)];
            if nu.len() == self.uops.len() {
                break;
            }
            self.uops = nu;
        }
    }

    pub fn optimize_uops(&mut self) {
        let mut acc_scope: HashMap<&UOp, Vec<&UOp>> = HashMap::new();
        for u in self.uops.iter() {
            if u.uop == UOps::PHI {
                acc_scope
                    .entry(&u.vin[0])
                    .or_default()
                    .extend(u.vin[2..].iter().collect::<Vec<&UOp>>())
            }
        }

        // fix loop scope
        let mut loop_stack = vec![vec![]];
        for u in self.uops.iter() {
            if loop_stack[loop_stack.len() - 1].is_empty() {
                loop_stack.last_mut().unwrap().push(u.clone())
            } else if u.uop == UOps::LOOP {
                loop_stack.push(vec![u.clone()])
            } else if !matches!(u.uop, UOps::CONST | UOps::ALU | UOps::CAST | UOps::LOAD) {
                loop_stack.last_mut().unwrap().push(u.clone())
            } else {
                let parents =
                    HashSet::<&UOp>::from_iter(get_recursive_parents(u, &mut acc_scope, true));
                if any(&v![u.uop == UOps::DEFINE_LOCAL, for u in parents.iter()]) {
                    loop_stack.last_mut().unwrap().push(u.clone());
                } else {
                    'out: for i in (0..loop_stack.len()).rev() {
                        if i == 0 {
                            loop_stack[i].push(u.clone());
                            break;
                        }
                        for x in loop_stack[i].iter() {
                            if parents.contains(x) {
                                loop_stack[i].push(u.clone());
                                break 'out;
                            }
                        }
                    }
                }
            }
        }
        self.uops = loop_stack.concat();

        self.remove_childless_uops();

        // add uops.end
        for i in 0..self.uops.len() {
            let u = &self.uops[i];
            if u.uop == UOps::LOOP {
                let uops_idxs: HashMap<&UOp, usize> =
                    HashMap::from_iter(v![(u, i), for (i, u) in self.uops.iter().enumerate()]);
                let mut inb = 0;
                for u in self.get_recursive_children(&u) {
                    inb = inb.max(uops_idxs[u])
                }
                inb += 1;
                self.uop(
                    UOps::END,
                    None,
                    vec![u.clone()],
                    vec![],
                    false,
                    Some(inb as isize),
                    true,
                );
            } else if u.uop == UOps::IF {
                self.uop(UOps::END, None, vec![u.clone()], vec![], false, None, true);
            }
        }
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
    pub fn const_idx(&mut self, val: String, insert_before: Option<isize>) -> UOp {
        self.uop(
            UOps::CONST,
            Some(int32),
            vec![],
            vec![Arg::Idx(val.parse::<isize>().unwrap())],
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
            //if !x.is_num() && x.expr().is_some() {
            let min = self.const_default(x.min().unwrap().to_string());
            let max = self.const_default((x.max().unwrap() + 1).to_string());
            new_loops.insert(
                x.expr().unwrap_or("").to_string(),
                self.uop(
                    UOps::LOOP,
                    Some(int32),
                    vec![min, max],
                    vec![],
                    false,
                    None,
                    true,
                ),
            );
            //}
        }
        let ret = new_loops.values().map(|x| x.clone()).collect();
        self.loop_uops.extend(new_loops);
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
                            true,
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
            id: uop_id(),
        };
        let key = &(uop, dtype, vin, arg);
        if let Some(expr) = self.saved_exprs.get(key) {
            if cachable
                && self.uops.iter().position(|e| *e == *expr).is_some_and(|i| {
                    i as isize <= insert_before.unwrap_or(self.uops.len() as isize)
                })
            {
                return expr.to_owned();
            }
        };
        if let Some(i) = insert_before {
            self.uops.insert(i as usize, ret.clone());
        } else {
            self.uops.push(ret.clone());
        }
        if cachable {
            self.saved_exprs.insert(key.clone(), ret.clone());
        }
        ret
    }

    fn global_load(
        &mut self,
        i: isize,
        idxs: Vec<ArcNode>,
        acc: Option<ConstNum>,
        mut barrier: Option<UOp>,
    ) -> Vec<UOp> {
        let buf_i = if i < 0 {
            self.kernel.bufs.len() as isize + i
        } else {
            i
        } as usize;
        let sts_i = if i < 0 {
            self.kernel.sts.len() as isize + i
        } else {
            i
        } as usize;
        let buf_uops_i = if i < 0 {
            self.buf_uops.len() as isize + i
        } else {
            i
        } as usize;
        let buf = &self.kernel.bufs[buf_i];
        let buf_string = format!("{:?}", buf);
        let localtype = buf.dtype();
        let const_ = if let Buffers::ConstBuffer(acc) = buf {
            if localtype.is_int() {
                Some(ConstNum::Int(acc.val.parse::<i128>().unwrap()))
            } else {
                Some(ConstNum::Float(acc.val.parse::<f32>().unwrap()))
            }
        } else {
            acc.clone()
        };

        let mut amt = 1;
        let mut dim: Option<isize> = None;
        let upcast_dim = self.get_upcast_dim(i);
        if upcast_dim.len() == 1 {
            let float4_expand = idxs[0].expand(None);
            if (float4_expand.len() == 4 || float4_expand.len() == 2) {
                dim = Some(upcast_dim[0]);
                amt = float4_expand.len();
            }
        }

        let expand_vars = v![rename_var(idx.expand_idx(), &format!("_uidx{j}")), for (j, idx) in idxs.iter().enumerate()];
        let fake_idxs = v![
        {
            let eidx = idx.expand_idx();
            if eidx.is_var() {
                idx.substitute(&HashMap::from([(idx.expand_idx(), ev.clone())]))
            } else {
                idx.clone()
            }
        }
        ,for (idx, ev) in izip!(idxs.iter(), expand_vars.iter())];
        let (mut g_idx, mut g_valid) = if let Some(d) = dim {
            let d = d as usize;
            let (mut gidx, mut gvalid) = self.kernel.sts[sts_i].expr_idxs(Some(
                vec![
                    &fake_idxs[..d],
                    //            &[float4_expand[0].clone()],
                    &fake_idxs[d + 1..],
                ]
                .concat(),
            ));
            if (&gidx / &num(amt as isize) * num(amt as isize)).render_default()
                != gvalid.render_default()
            {
                (gidx, gvalid) = self.kernel.sts[sts_i].expr_idxs(Some(fake_idxs.clone()));
                amt = 1;
                dim = None;
            }
            (gidx, gvalid)
        } else {
            self.kernel.sts[sts_i].expr_idxs(Some(fake_idxs.clone()))
        };
        // if amt > 1 {
        //     //TODO: localtype.vectorize()
        // }
        let (e_idxs, e_valids) = (
            g_idx.expand(Some(expand_vars.clone())),
            g_valid.expand(Some(expand_vars.clone())),
        );

        let mut ret = vec![];
        let mut invalid_value = Some(if buf.dtype().is_int() {
            ConstNum::Int(0)
        } else {
            ConstNum::Float(0.0)
        });
        for (idx, valid, rep_idx) in izip!(e_idxs.iter(), e_valids.iter(), iter_idxs(&expand_vars))
        {
            let (this_const, idx, valid) = if valid.max().unwrap() == 0 {
                (invalid_value.clone(), num(0), num(1))
            } else {
                (const_.clone(), idx.clone(), valid.clone())
            };
            let key = format!(
                "{:?}{localtype}{:?}{:?}{}{}",
                acc.as_ref(),
                this_const,
                buf_string,
                idx.render_default(),
                valid.render_default()
            );
            if !self.load_cache.contains_key(&key) {
                if acc.is_some() {
                    let tmp = self.uop(
                        UOps::DEFINE_ACC,
                        Some(localtype.clone()),
                        vec![],
                        vec![Arg::Str(this_const.unwrap().to_string())],
                        false,
                        None,
                        true,
                    );
                    self.load_cache.insert(key.clone(), tmp);
                } else if let Some(_const) = this_const {
                    let tmp = self._const(_const.to_string(), localtype.clone(), None);
                    self.load_cache.insert(key.clone(), tmp);
                    if valid.min().unwrap() == 0 && valid.max().unwrap() == 1 {
                        let valid_render = self.render(valid.clone());
                        let cache_tmp = self.load_cache[&key].clone();
                        let const_tmp = self._const(
                            invalid_value.as_ref().unwrap().to_string(),
                            localtype.clone(),
                            None,
                        );
                        let tmp = self.uop_default(
                            UOps::ALU,
                            Some(localtype.clone()),
                            vec![valid_render, cache_tmp, const_tmp],
                            vec![Arg::OpType(OpType::Ternary(Ternary::Where))],
                        );
                        self.load_cache.insert(key.clone(), tmp);
                    }
                } else {
                    assert!(
                        self.buf_uops[buf_uops_i].is_some(),
                        "buffer {i} wasn't UOped"
                    );
                    let buf_uop = self.buf_uops[buf_uops_i].clone().unwrap();
                    // WARN: This seem to be always empty
                    // valid_tuple = (valid.render(self.render_ops, self), self.const(invalid_value, localtype)) if valid.min == 0 else tuple()
                    // panic!("{:?}", idx.render_default());
                    let rendered_idx = self.render(idx);

                    //valid_tuple = (valid.render(self.render_ops, self), self.const(invalid_value, localtype)) if valid.min == 0 else tuple()
                    let valid_tuple = if valid.min().unwrap() == 0 {
                        vec![
                            self.render(valid),
                            self._const(
                                invalid_value.as_ref().unwrap().to_string(),
                                localtype.clone(),
                                None,
                            ),
                        ]
                    } else {
                        vec![]
                    };
                    //panic!();
                    let mut vin = vec![vec![buf_uop, rendered_idx], valid_tuple];
                    if let Some(bb) = barrier.take() {
                        vin.push(vec![bb])
                    }
                    let tmp =
                        self.uop_default(UOps::LOAD, Some(localtype.clone()), vin.concat(), vec![]);
                    self.load_cache.insert(key.clone(), tmp);
                }
            }
            if let Some(d) = dim {
                ret.push(self.uop_default(
                    UOps::GEP,
                    Some(localtype.clone()),
                    vec![self.load_cache[&key].clone()],
                    vec![Arg::Str(rep_idx[d as usize].to_string())],
                ))
            } else {
                ret.push(self.load_cache[&key].clone())
            }
        }
        ret
    }

    fn global_store(&mut self, i: isize, idxs: Vec<ArcNode>, store: Vec<UOp>) -> Vec<UOp> {
        let buf_i = if i < 0 {
            self.kernel.bufs.len() as isize + i
        } else {
            i
        } as usize;
        let buf_uops_i = if i < 0 {
            self.buf_uops.len() as isize + i
        } else {
            i
        } as usize;
        let sts_i = if i < 0 {
            self.kernel.sts.len() as isize + i
        } else {
            i
        } as usize;
        let buf = &self.kernel.bufs[buf_i];
        let buf_uop = self.buf_uops[buf_uops_i].clone();
        assert!(buf_uop.is_some(), "buffer {i} wasn't UOped");
        let expanded_node = v![idx.expand(None), for idx in idxs.iter()];
        let mut _idxs = v![x.into_iter().rev().collect::<Vec<ArcNode>>(), for x in cartesian_product(expanded_node.clone().into_iter().rev().collect())];
        if _idxs.len() == 0 {
            _idxs = vec![vec![]];
        }
        let store_offset = v![(i, s), for (i, s) in izip!(_idxs, store)];
        // let upcast_dim = self.get_upcast_dim(i);
        // if upcast_dim.len() == 1 && matches!(expanded_node[upcast_dim[0] as usize].len(), 2 | 4) {
        //     //TODO: float4
        // }
        let mut stores = vec![];
        for (idx, var) in store_offset.iter() {
            let (idx, valid) = self.kernel.sts[sts_i].expr_idxs(Some(idx.to_vec()));
            let render_idx = self.render(idx);
            if valid.min().unwrap() == 1 {
                stores.push(self.uop_default(
                    UOps::STORE,
                    None,
                    vec![buf_uop.clone().unwrap(), render_idx, var.clone()],
                    vec![],
                ));
            } else {
                let valid_rendered = self.render(valid);
                stores.push(self.uop_default(
                    UOps::STORE,
                    None,
                    vec![
                        buf_uop.clone().unwrap(),
                        render_idx,
                        var.clone(),
                        valid_rendered,
                    ],
                    vec![],
                ));
            }
        }
        stores
    }

    fn get_upcast_dim(&self, i: isize) -> Vec<isize> {
        let buf_i = if i < 0 {
            self.kernel.bufs.len() as isize + i
        } else {
            i
        } as usize;
        let sts_i = if i < 0 {
            self.kernel.sts.len() as isize + i
        } else {
            i
        } as usize;
        let should_upcast = self.kernel.opts.support_float4
            && matches!(self.kernel.bufs[buf_i].dtype(), float32 | float16);
        v![x, for x in self.kernel.sts[sts_i].unit_stride_axes(false), if should_upcast && x >= self.kernel.shape_len()-self.kernel.upcasted && Shape::from(self.kernel.sts[sts_i].shape_vec())[x] > 1]
    }

    fn ast_parse(
        &mut self,
        mut x: LazyOp,
        acc: &mut [UOp],
        offs: Option<&[isize]>,
        loaded_buffers: &HashMap<Buffers, Vec<UOp>>,
        do_reduce: bool,
        loop_ctx: Option<&[UOp]>,
        cache: Option<&mut HashMap<LazyOp, Vec<UOp>>>,
    ) -> Vec<UOp> {
        let mut map = HashMap::new();
        let cache = cache.unwrap_or(&mut map);
        if cache.contains_key(&x) {
            return cache[&x].clone();
        }
        match &x.optype {
            OpType::Buffer(_) => return loaded_buffers[&x.args[0].to_buf()].clone(),
            OpType::Reduce(b) => {
                if !do_reduce {
                    return acc.to_vec();
                }
                match b {
                    Reduce::Sum => {
                        // TODO: UnaryOps.Cast
                        // if x.op == ReduceOps.SUM and x.src[0].__class__ is LazyOp and x.src[0].op == UnaryOps.CAST and x.src[0].src[0].__class__ is LazyOp and x.src[0].src[0].op == BinaryOps.MUL:  # noqa: E501
                        //   x = LazyOp(TernaryOps.MULACC, x.src[0].src[0].src, x.arg)
                        if x.src[0].optype() == Binary::Mul {
                            x = LazyOp::new(
                                OpType::Ternary(Ternary::Mulacc),
                                x.src[0].src(),
                                Some(x.args),
                            );
                        }
                    }
                    _ => (),
                }
            }
            OpType::Unary(Unary::Cast) => {
                let Arg::Dtype(ref to_dtype) = x.args[0] else {
                    panic!("Cast op lazyop arg[0] should be a dtype")
                };
                //return [self.uop(UOps.CAST, self.get_base_dtype(x.arg[0]), (u,), x.arg) for u in self.ast_parse(x.src[0], acc, offs, loaded_buffers)]
                return v![self.uop_default(UOps::CAST, Some(to_dtype.clone()), vec![u], x.args.clone()), for u in self.ast_parse(x.src[0].lo().clone(), acc, offs, loaded_buffers, do_reduce, loop_ctx, None)];
            }
            _ => (),
        }
        let values = v![self.ast_parse(v.lo().clone(), acc, offs, loaded_buffers, do_reduce, loop_ctx, Some(cache)), for v in x.src.iter()];
        let ops = HashMap::from([
            (OpType::Reduce(Reduce::Sum), OpType::Binary(Binary::Add)),
            (OpType::Reduce(Reduce::Max), OpType::Binary(Binary::Max)),
            (
                OpType::Ternary(Ternary::Mulacc),
                OpType::Ternary(Ternary::Mulacc),
            ),
        ]);
        let mut ret = vec![];
        let mut values_transpose = vec![vec![None; values.len()]; values[0].len()];
        for r in 0..values.len() {
            for c in 0..values[0].len() {
                values_transpose[c][r] = Some(&values[r][c]);
            }
        }
        let values_transpose = values_transpose
            .iter()
            .map(|r| r.iter().map(|n| n.unwrap().clone()).collect::<Vec<UOp>>())
            .collect::<Vec<Vec<UOp>>>();
        if ops.contains_key(&x.optype) {
            //panic!();
            let input_acc = acc.to_vec();
            for (val, &off) in izip!(values_transpose.iter(), offs.as_ref().unwrap().iter()) {
                let off = off as usize;
                acc[off] = self.uop_default(
                    UOps::ALU,
                    None,
                    vec![val.clone(), vec![acc[off].clone()]].concat(),
                    vec![Arg::OpType(ops[&x.optype].clone())],
                );
                ret.push(acc[off].clone());
            }
            for off in 0..acc.len() {
                if input_acc[off] != acc[off] {
                    acc[off] = self.uop_default(
                        UOps::PHI,
                        input_acc[off].dtype.clone(),
                        vec![
                            vec![input_acc[off].clone(), acc[off].clone()],
                            if loop_ctx.is_some() {
                                loop_ctx.as_ref().unwrap().to_vec()
                            } else {
                                vec![]
                            },
                        ]
                        .concat(),
                        vec![],
                    );
                }
            }
        } else {
            ret = v![self.uop(UOps::ALU, if x.optype == Binary::Cmplt { Some(_bool)} else {Some(float32)}, val.clone(),vec![Arg::OpType(x.optype.clone())], true, None, true), for val in values_transpose.iter()];
        }
        cache.insert(x.clone(), ret.clone());
        ret
    }

    fn get_recursive_children<'a>(&'a self, x: &'a UOp) -> HashSet<&UOp> {
        let mut deps = HashSet::from([x]);
        let mut size = 0;
        while size != deps.len() {
            for u in self.uops.iter() {
                size = deps.len();
                for x in u.vin.iter() {
                    if u.uop == UOps::PHI && deps.contains(x) {
                        deps.insert(u);
                        break;
                    }
                }
            }
        }
        deps
    }

    fn replace_op(&mut self, old: &UOp, new: &UOp) {
        for u in self.uops.iter_mut() {
            u.vin = v![if x == old { new.clone() } else {x.clone() }, for x in u.vin.iter()];
        }
        self.uops
            .remove(self.uops.iter().position(|u| u == old).unwrap());
    }

    fn acc_offset(&self, i: isize) -> Vec<isize> {
        if self.kernel.upcasted == 0 {
            return vec![0];
        }
        let upcasted_i = self.kernel.upcasted_axis(i);
        let r_upcasted_i =
            upcasted_i
                .iter()
                .rev()
                .map(|c| c.clone())
                .collect::<Vec<(isize, Option<isize>, isize)>>();

        //acc_strides = [x*(1-upcasted_i[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in upcasted_i[::-1])))]
        let acc_strides = v![x*(1-r_upcasted_i[i].2), for (i, x) in strides_for_shape(&v![if *r > 0 { 1 } else { *s }, for (s, _, r) in r_upcasted_i.iter()]).into_iter().enumerate()];
        //return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(upcasted_i[::-1])])]
        v![t.iter().sum::<isize>(), for t in cartesian_product(v![v![y*acc_strides[i], for y in 0..x.0], for (i, x) in r_upcasted_i.iter().enumerate()])]
    }
}

fn get_grouped_dims(
    prefix: &str,
    start_dim: usize,
    local_dims: &[isize],
    maxdim: usize,
) -> (Vec<ArcNode>, Vec<ArcNode>) {
    //local_idxs = loop_local_idxs = [Variable(f"{prefix}{start_dim+i}", 0, s-1) for i,s in enumerate(local_dims[0:maxdim-1] + (prod(local_dims[maxdim-1:]),) if len(local_dims) > maxdim else local_dims)]  # noqa: E501
    let mut iter = if local_dims.len() > maxdim {
        vec![
            local_dims[..maxdim - 1].to_vec(),
            vec![prod(&local_dims[maxdim - 1..].to_vec())],
        ]
        .concat()
    } else {
        local_dims.to_vec()
    };
    let mut local_idxs = v![var(&format!("{}{}",prefix, start_dim+i), 0, s-1), for (i, s) in iter.into_iter().enumerate()];
    let loop_local_idxs = local_idxs.clone();
    if maxdim != 0 && local_dims.len() > maxdim {
        let mut dd = local_idxs[maxdim - 1].clone();
        let mut nli = vec![];
        for s in local_dims[maxdim - 1..].iter().rev() {
            nli.push(&dd % num(*s));
            dd = dd / num(*s);
        }
        local_idxs = local_idxs[0..maxdim - 1].to_vec();
        local_idxs.extend(nli.into_iter().rev());
    }
    (local_idxs, v![x, for x in loop_local_idxs, if !x.is_num()])
}

pub fn get_reduce_acc(op: OpType, dtype: Dtype) -> ConstNum {
    if op == Reduce::Sum {
        return if dtype.is_float() {
            ConstNum::Float(0.0)
        } else {
            ConstNum::Int(0)
        };
    };
    if op == Reduce::Max {
        return if dtype.is_float() {
            ConstNum::Float(f32::NEG_INFINITY)
        } else {
            ConstNum::Int(-i128::MAX)
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

impl Linearizer {
    pub fn uop_alu_idx(&mut self, a: UOp, b: ArcNode, op: OpType, dtype: Option<Dtype>) -> UOp {
        let b = self.render(b);
        self.uop_default(UOps::ALU, dtype, vec![a, b], vec![Arg::OpType(op)])
    }

    pub fn render(&mut self, node: ArcNode) -> UOp {
        let ret = match format!("{:?}", node.0).split(" ").next().unwrap() {
            "MulNode" => {
                let a = self.render(node.a().unwrap());
                self.uop_alu_idx(a, node.b().unwrap(), OpType::Binary(Binary::Mul), None)
            }
            "DivNode" => {
                let a = self.render(node.a().unwrap());
                self.uop_alu_idx(a, node.b().unwrap(), OpType::Binary(Binary::Div), None)
            }
            "ModNode" => {
                let a = self.render(node.a().unwrap());
                self.uop_alu_idx(a, node.b().unwrap(), OpType::Binary(Binary::Mod), None)
            }
            "LtNode" => {
                let a = self.render(node.a().unwrap());
                self.uop_alu_idx(a, node.b().unwrap(), OpType::Binary(Binary::Cmplt), None)
            }
            "Variable" => self.loop_uops[node.expr().unwrap()].clone(),
            "NumNode" => self.const_idx(node.num_val().unwrap().to_string(), None),
            "SumNode" => {
                let nodes = node.nodes();
                let mut uop = self.render(nodes[0].clone());
                for n in nodes[1..].iter() {
                    uop = self.uop_alu_idx(uop, n.clone(), OpType::Binary(Binary::Add), None);
                }
                uop
            }
            "AndNode" => {
                let nodes = node.nodes();
                let mut uop = self.render(nodes[0].clone());
                for n in nodes[1..].iter() {
                    uop =
                        self.uop_alu_idx(uop, n.clone(), OpType::Binary(Binary::Mul), Some(_bool));
                }
                uop
            }
            t => panic!("you forgot this {t}"),
        };
        return ret;
    }
}

fn get_recursive_parents<'a>(
    x: &'a UOp,
    acc_scope: &mut HashMap<&'a UOp, Vec<&'a UOp>>,
    with_phi: bool,
) -> Vec<&'a UOp> {
    vec![
        x.vin.iter().collect::<Vec<&UOp>>(),
        v![get_recursive_parents(p, acc_scope, with_phi),for p in x.vin.iter()].concat(),
        if with_phi && acc_scope.get(x).is_some() {
            acc_scope[x].clone()
        } else {
            vec![]
        },
    ]
    .concat()
}

pub fn cartesian_product<T: Clone>(lists: Vec<Vec<T>>) -> Vec<Vec<T>> {
    match lists.split_first() {
        Some((first, rest)) => {
            let init: Vec<Vec<T>> = first.iter().cloned().map(|n| vec![n]).collect();

            rest.iter()
                .cloned()
                .fold(init, |vec, list| partial_cartesian(vec, &list))
        }
        None => {
            vec![]
        }
    }
}
pub fn partial_cartesian<T: Clone>(a: Vec<Vec<T>>, b: &[T]) -> Vec<Vec<T>> {
    a.into_iter()
        .flat_map(|xs| {
            b.iter()
                .cloned()
                .map(|y| {
                    let mut vec = xs.clone();
                    vec.push(y);
                    vec
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
