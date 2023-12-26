use itertools::Itertools;

use crate::dtype::{_bool, float16, float32, int32};
use crate::prelude::*;
use crate::shape::shapetracker::strides_for_shape;
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use std::sync::Arc;

use crate::arg::Arg;
use crate::codegen::kernel::{Buffers, LocalBuffer, KERNEL_CNT};
use crate::ops::{Binary, LazyOp, LazyOpSrc, Reduce, Ternary, Unary};
use crate::shape::symbolic::{iter_idxs, num, var, ArcNode, NodeOp};
use crate::shape::ShapeTracker;
use crate::tensor::shape::Shape;
use crate::{dtype, lazy::LazyBuffer, ops::OpType};

use super::kernel::{ConstNum, Kernel};

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
    pub saved_exprs: HashMap<(UOps, Option<Dtype>, Vec<UOp>, Vec<Arg>), UOp>,
    pub global_size: Option<Vec<usize>>,
    pub local_size: Option<Vec<usize>>,
    pub load_cache: HashMap<String, UOp>,
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
            load_cache: HashMap::new(),
        }
    }

    pub fn linearize(&mut self) {
        // # no new opts and we already ran? skip relinearizing if self.applied_opts == self.applied_opts_cache: return self
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
            self.local_size = Some(
                v![(x.max().unwrap() + 1 ) as usize, for x in loop_local_idxs.iter().rev().rev()],
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
            self.loop_ops.extend(extend_loop_uops);

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
            self.loop_ops.extend(extend_loop_uops);
        } else {
            self.render_loop(&vec![loop_global_idxs.clone(), loop_local_idxs.clone()].concat());
        }

        let mut loaded_buffers: HashMap<Buffers, Vec<UOp>> = HashMap::new();
        let mut acc: Vec<UOp> = vec![];
        self.load_cache = HashMap::new();

        let mut fake_reduce_idxs: Vec<ArcNode> = vec![];

        if let Some(reduceop) = &self.kernel.reduceop {
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
            //else if lcaols_to_store

            // Load early bufs
            //loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[i]) if i in self.local_alias else i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs})  # noqa: E501
            let iter_ = v![(i, b.clone()), for (i, b) in  self.kernel.bufs.iter().enumerate().skip(1), if self.kernel.earlybufs.contains(b)];

            //lb_ex = {b:self.global_load(i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs}
            let lb_ex = v![(b, self.global_load(i, vec![global_idx.clone(), local_idxs.clone(), reduce_idxs.clone(), full_upcast_idxs.clone()].concat(), None, None)), for (i, b) in iter_];
            //panic!();
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
                //TODO:
            }
        }

        // load late bufs
        let iter_ = v![(i, b.clone()), for (i, b) in  self.kernel.bufs.iter().enumerate(), if !self.kernel.earlybufs.contains(b) && i != 0 && !matches!(b, Buffers::LocalBuffer(_))];
        loaded_buffers.extend(v![(b, self.global_load(i, vec![global_idx.clone(), local_idxs.clone(), fake_reduce_idxs.clone(), upcast_idxs.clone()].concat(), None, None)), for (i, b) in iter_]);
        let val = self.ast_parse(
            self.kernel.ast.src[0].lo().clone(),
            &mut acc,
            None,
            &loaded_buffers,
            false,
            None,
        );
        self.global_store(
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

        let mut acc_scope: HashMap<UOp, Vec<UOp>> = HashMap::new();
        for u in self.uops.iter() {
            if u.uop == UOps::PHI {
                acc_scope
                    .entry(u.vin[0].clone())
                    .or_default()
                    .extend(u.vin[2..].iter().map(|n| n.clone()).collect::<Vec<UOp>>())
            }
        }

        let mut loop_stack = vec![vec![]];
        for u in self.uops.iter() {
            if loop_stack[loop_stack.len() - 1].is_empty() {
                loop_stack.last_mut().unwrap().push(u.clone())
            } else if u.uop == UOps::LOOP {
                loop_stack.push(vec![u.clone()])
            } else if !matches!(u.uop, UOps::CONST | UOps::ALU | UOps::CAST | UOps::LOAD) {
                loop_stack.last_mut().unwrap().push(u.clone())
            } else {
                let parents = get_recursive_parents(u.clone(), &acc_scope, true);
                for i in (0..loop_stack.len()).rev() {
                    if v![0, for x in loop_stack[i].iter(), if parents.contains(x)].len() > 0
                        || i == 0
                    {
                        loop_stack[i].push(u.clone());
                        break;
                    }
                }
            }
        }
        self.uops = loop_stack.concat();

        // # uops optimization
        // changed_something = True
        // while changed_something:
        //   changed_something = False
        //   for u in self.uops:
        //     if u.uop == UOps.PHI and len(u.vin) == 3:
        //       # if the parents of the PHI node don't have the LOOP in their parents, it can be folded
        //       # TODO: ADD becomes a MUL, MAX can just become nothing
        //       if all(x.uop != UOps.LOOP for x in get_recursive_parents(UOp(u.uop, u.dtype, u.vin[0:2], u.arg))) and u.vin[1].arg == BinaryOps.ADD:
        //         if DEBUG >= 4: print(f"removing PHI node {u}")
        //         del self.saved_exprs[(u.uop, u.dtype, u.vin, u.arg)]
        //         # NOTE: assuming u.vin[2].vin[1] and u.vin[2].vin[0] have the same dtype
        //         loop_len = self.uop(UOps.ALU, u.vin[2].vin[1].dtype, (u.vin[2].vin[1], u.vin[2].vin[0]), BinaryOps.SUB, insert_before=self.uops.index(u))
        //         if loop_len.dtype != u.dtype: loop_len = self.uop(UOps.CAST, u.dtype, (loop_len,), insert_before=self.uops.index(u))
        //         replace_op(u, self.uop(UOps.ALU, u.dtype, (u.vin[1], loop_len,), BinaryOps.MUL, insert_before=self.uops.index(u)))
        //         changed_something = True

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

        // add uops.end
        for i in 0..self.uops.len() {
            let u = &self.uops[i];
            if u.uop == UOps::LOOP {
                //self.uop(UOps.END, None, (u,), cachable=False, insert_before=self.uops.index(sorted(list(get_recursive_children(u)), key=self.uops.index)[-1])+1)  # noqa: E501
                let inb = self
                    .uops
                    .iter()
                    .position(|uu| {
                        uu == self
                            .get_recursive_children(u.clone())
                            .iter()
                            .sorted_by_key(|x| self.uops.iter().position(|p| &p == x).unwrap())
                            .last()
                            .unwrap()
                    })
                    .unwrap()
                    + 1;
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

        // for u in self.uops.iter() {
        //     println!("{:?} {:?} {:?}", u.uop, if u.vin.first().is_some() { Some(u.vin[0].uop.clone()) } else { None}, u.args.first());
        // }
        self.kernel.sts = sts_backup;
        self.kernel.group_for_reduce = gfr_backup;
        self.kernel.upcasted = upc_backup;
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
                x.expr().unwrap().to_string(),
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
        //println!("trying to insert uop {:?} {:?} {:?}", uop, if vin.len() > 0 { Some(&vin[0].uop) } else { None }, arg.first());
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
        let key = &(uop, dtype, vin, arg);
        let insert_before = if insert_before.is_some() {
            insert_before.unwrap()
        } else {
            self.uops.len() as isize
        };
        if let Some(expr) = self.saved_exprs.get(key) {
            if cachable
                && self
                    .uops
                    .iter()
                    .position(|e| *e == *expr)
                    .is_some_and(|i| i as isize <= insert_before)
            {
                return expr.to_owned();
            }
        };
        self.uops.insert(insert_before as usize, ret.clone());
        if cachable {
            self.saved_exprs.insert(key.clone(), ret.clone());
        }
        ret
    }

    fn global_load(
        &mut self,
        i: usize,
        idxs: Vec<ArcNode>,
        acc: Option<ConstNum>,
        mut barrier: Option<UOp>,
    ) -> Vec<UOp> {
        let buf = &self.kernel.bufs[i];
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
        let fake_idxs = v![idx.substitute(&HashMap::from([(idx.expand_idx(), ev.clone())])), for (idx, ev) in izip!(idxs.iter(), expand_vars.iter())];
        let (mut g_idx, mut g_valid) = if let Some(d) = dim {
            let d = d as usize;
            let (mut gidx, mut gvalid) = self.kernel.sts[i].expr_idxs(Some(
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
                (gidx, gvalid) = self.kernel.sts[i].expr_idxs(Some(fake_idxs.clone()));
                amt = 1;
                dim = None;
            }
            (gidx, gvalid)
        } else {
            self.kernel.sts[i].expr_idxs(Some(fake_idxs.clone()))
        };
        //println!("{}", g_idx.render_default());
        if amt > 1 {
            //TODO: localtype.vectorize()
        }
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
                "{:?}{localtype}{:?}{}{}",
                acc.as_ref(),
                this_const,
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
                } else if this_const.is_some() {
                    let tmp = self._const(this_const.unwrap().to_string(), localtype.clone(), None);
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
                    assert!(self.buf_ops[i].is_some(), "buffer {i} wasn't UOped");
                    let buf_uop = self.buf_ops[i].clone().unwrap();
                    // WARN: This seem to be always empty
                    // valid_tuple = (valid.render(self.render_ops, self), self.const(invalid_value, localtype)) if valid.min == 0 else tuple()
                    // panic!("{:?}", idx.render_default());
                    let rendered_idx = self.render(idx);
                    //panic!();
                    let mut vin = vec![buf_uop, rendered_idx];
                    if let Some(bb) = barrier.take() {
                        vin.push(bb)
                    }
                    let tmp = self.uop_default(
                        UOps::LOAD,
                        Some(localtype.clone()),
                        vin,
                        vec![Arg::OpType(OpType::Ternary(Ternary::Where))],
                    );
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
        let i = if i < 0 {
            (self.kernel.bufs.len() as isize - i) as usize
        } else {
            i as usize
        };
        let buf = &self.kernel.bufs[i];
        let buf_uop = self.buf_ops[i].clone();
        assert!(buf_uop.is_some(), "buffer {i} wasn't UOped");
        let expanded_node = v![idx.expand(None), for idx in idxs.iter()];
        let _idxs = v![x.iter().rev().map(|n| n.clone()).collect::<Vec<ArcNode>>(), for x in cartesian_product(expanded_node.clone().into_iter().rev().collect())];
        let store_offset = v![(i, s), for (i, s) in izip!(_idxs, store)];
        let upcast_dim = self.get_upcast_dim(i);
        if upcast_dim.len() == 1 && matches!(expanded_node[upcast_dim[0] as usize].len(), 2 | 4) {
            //TODO: float4
        }
        let mut stores = vec![];
        for (idx, var) in store_offset.iter() {
            let (idx, valid) = self.kernel.sts[i].expr_idxs(Some(idx.to_vec()));
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

    fn get_upcast_dim(&self, i: usize) -> Vec<isize> {
        let should_upcast = self.kernel.opts.support_float4
            && matches!(self.kernel.bufs[i].dtype(), float32 | float16);
        v![x, for x in self.kernel.sts[i].unit_stride_axes(false), if should_upcast && x >= self.kernel.shape_len()-self.kernel.upcasted && Shape::from(self.kernel.sts[i].shape())[x] > 1]
    }

    fn ast_parse(
        &mut self,
        mut x: LazyOp,
        acc: &mut [UOp],
        offs: Option<&[isize]>,
        loaded_buffers: &HashMap<Buffers, Vec<UOp>>,
        do_reduce: bool,
        loop_ctx: Option<&[UOp]>,
    ) -> Vec<UOp> {
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
                        if matches!(x.src[0], LazyOpSrc::LazyOp(_))
                            && x.src[0].optype() == Binary::Mul
                        {
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
            _ => (),
        }
        // for v in x.src.iter() {
        //     assert!(
        //         self.ast_parse(
        //             v.lo().clone(),
        //             acc,
        //             offs,
        //             loaded_buffers,
        //             do_reduce,
        //             loop_ctx
        //         )
        //         .len()
        //             > 0
        //     );
        // }
        let values = v![self.ast_parse(v.lo().clone(), acc, offs, loaded_buffers, do_reduce, loop_ctx), for v in x.src.iter()];
        let ops = HashMap::from([
            (OpType::Reduce(Reduce::Sum), OpType::Binary(Binary::Add)),
            (OpType::Reduce(Reduce::Max), OpType::Binary(Binary::Max)),
            (
                OpType::Ternary(Ternary::Mulacc),
                OpType::Ternary(Ternary::Mulacc),
            ),
        ]);
        let mut ret = vec![];
        // let values_transpose = (0..values.len())
        //     .map(|i| {
        //         values
        //             .iter()
        //             .map(|row| row[i].clone())
        //             .collect::<Vec<UOp>>()
        //     })
        //     .collect::<Vec<Vec<UOp>>>();
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
        ret
    }

    fn get_recursive_children(&self, x: UOp) -> HashSet<UOp> {
        let mut deps = HashSet::from([x.clone()]);
        let mut ssize = 0;
        while ssize != deps.len() {
            ssize = deps.len();
            for u in self.uops.iter() {
                if deps
                    .intersection(&HashSet::from_iter(
                        v![x, for x in u.vin.iter(), if x.uop != UOps::PHI],
                    ))
                    .count()
                    > 0
                {
                    deps.insert(u.clone());
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

        //acc_strides = [x*(1-upcasted_i[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in upcasted_i[::-1])))]
        let r_upcasted_i = upcasted_i
            .iter()
            .rev()
            .map(|c| c.clone())
            .collect::<Vec<Vec<isize>>>();
        let acc_strides = v![x*(1-r_upcasted_i[i][2]), for (i, x) in strides_for_shape(&v![if ui[2] > 0 { 1 } else { ui[1] }, for ui in upcasted_i.iter()]).into_iter().enumerate()];
        //return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(upcasted_i[::-1])])]
        v![t.iter().sum::<isize>(), for t in cartesian_product(v![v![y*acc_strides[i], for y in 0..x[0]], for (i, x) in r_upcasted_i.iter().enumerate()])]
    }
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
            ConstNum::Float(-f32::INFINITY)
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

// #[allow(unused_variables)]
// pub trait NodeOp: 'static + core::fmt::Debug {
//     fn variable(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         // Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}]" if ctx == "DEBUG" else f"{self.expr}",
//         if ctx.is_some_and(|f| f == "DEBUG") {
//             return format!(
//                 "{}[{}-{}]",
//                 s.expr().unwrap(),
//                 s.min().unwrap(),
//                 s.max().unwrap()
//             );
//         }
//         s.expr().unwrap().to_string()
//     }
//
//     fn num(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         // NumNode: lambda self,ops,ctx: f"{self.b}",
//         s.b().unwrap().to_string()
//     }
//
//     fn mul(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         // MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{sym_render(self.b,ops,ctx)})",
//         format!(
//             "({}*{})",
//             s.a().unwrap().render(self.to_arc(), ctx, false),
//             s.b().unwrap().render(self.to_arc(), ctx, false), // <-- Everything should be a Node here,
//                                                               // so no need to "sym_render()"
//         )
//     }
//
//     fn div(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         // DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}/{self.b})",
//         format!(
//             "({}/{})",
//             s.a().unwrap().render(self.to_arc(), ctx, false),
//             s.b().unwrap()
//         )
//     }
//
//     fn _mod(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         // ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
//         format!(
//             "({}%{})",
//             s.a().unwrap().render(self.to_arc(), ctx, false),
//             s.b().unwrap()
//         )
//     }
//
//     fn lt(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         //LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
//         format!(
//             "({}<{})",
//             s.a().unwrap().render(self.to_arc(), ctx, false),
//             s.b().unwrap().render(self.to_arc(), ctx, false),
//         )
//     }
//
//     fn sum(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         let mut renders = vec![];
//         for n in s.nodes() {
//             renders.push(n.render(self.to_arc(), ctx, false));
//         }
//         renders.sort();
//         format!("({})", renders.join("+"))
//     }
//
//     fn and(&self, s: ArcNode, ctx: Option<&str>) -> String {
//         let mut renders = vec![];
//         for n in s.nodes() {
//             renders.push(n.render(self.to_arc(), ctx, false));
//         }
//         renders.sort();
//         format!("({})", renders.join("&&"))
//     }
//
//     fn to_arc(&self) -> Arc<dyn NodeOp>;
// }

impl Linearizer {
    pub fn uop_alu_idx(&mut self, a: UOp, b: ArcNode, op: OpType, dtype: Option<Dtype>) -> UOp {
        let b = self.render(b);
        self.uop_default(UOps::ALU, dtype, vec![a, b], vec![Arg::OpType(op)])
    }

    pub fn render(&mut self, node: ArcNode) -> UOp {
        return match format!("{:?}", node.0).split(" ").next().unwrap() {
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
            "Variable" => self.loop_ops[node.expr().unwrap()].clone(),
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
    }
}

fn get_recursive_parents(
    x: UOp,
    acc_scope: &HashMap<UOp, Vec<UOp>>,
    with_phi: bool,
) -> HashSet<UOp> {
    let mut ret = HashSet::from_iter(x.vin.clone());
    for p in x.vin.iter() {
        ret = ret
            .union(&get_recursive_parents(p.clone(), acc_scope, with_phi))
            .into_iter()
            .map(|s| s.clone())
            .collect::<HashSet<UOp>>();
    }
    if with_phi && acc_scope.get(&x).is_some() {
        ret = ret
            .union(&HashSet::from_iter(acc_scope[&x].clone()))
            .map(|s| s.clone())
            .collect::<HashSet<UOp>>();
    }
    ret
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
