use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::prelude::*;
use crate::{
    arg::Arg,
    ops::{LazyOp, LazyOpSrc, Load, Movement, OpType},
    shape::{shapetracker::ShapeTracker, symbolic::gcd},
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct LazyBufferId(pub(crate) usize);

pub(crate) fn lb_id() -> LazyBufferId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    LazyBufferId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
unsafe impl Send for LazyBuffer {}
unsafe impl Sync for LazyBuffer {}

#[derive(Clone, Debug)]
pub struct LOArc(Arc<LazyOp>);

impl From<LazyOp> for LOArc {
    fn from(value: LazyOp) -> Self {
        Self(Arc::new(value))
    }
}

impl Deref for LOArc {
    type Target = LazyOp;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LOArc {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Arc::get_mut_unchecked(&mut self.0) }
    }
}

#[derive(Clone, Debug)]
pub struct STArc(Arc<ShapeTracker>);

impl From<ShapeTracker> for STArc {
    fn from(value: ShapeTracker) -> Self {
        Self(Arc::new(value))
    }
}

impl Deref for STArc {
    type Target = ShapeTracker;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for STArc {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Arc::get_mut_unchecked(&mut self.0) }
    }
}

#[derive(Clone)]
pub struct LazyBuffer {
    pub lazyop: LOArc,
    pub st: STArc,
    pub device_buffer: Arc<Option<Arc<dyn Buffer>>>, // TODO: Does it have to be shared before realized?
    pub _base: Option<Arc<LazyBuffer>>,
    pub shape: Vec<isize>,
    pub children: HashSet<LazyBuffer>,
    pub views: HashSet<LazyBuffer>,
    pub id: LazyBufferId,
    pub dtype: Dtype,
    pub device: String,
}

impl PartialEq for LazyBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for LazyBuffer {}
impl std::hash::Hash for LazyBuffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl core::fmt::Debug for LazyBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<LB {:?} id={:?} op={:?} st={:?} device_buffer={:?} _base={:?}>",
            self.shape, self.id, self.lazyop.optype, self.st.views, self.device_buffer, self._base
        )
    }
}

impl LazyBuffer {
    pub fn new(
        device: &str,
        st: ShapeTracker,
        optype: OpType,
        op: LazyOp,
        dtype: Dtype,
        src: Option<Arc<dyn Buffer>>,
        base: Option<Arc<LazyBuffer>>,
    ) -> Self {
        let mut ret = Self {
            device: device.into(),
            lazyop: op.into(),
            shape: st.shape(),
            st: st.into(),
            device_buffer: Arc::new(src),
            _base: base,
            children: HashSet::new(),
            views: HashSet::new(),
            id: lb_id(),
            dtype,
        };
        let rc = ret.clone();
        for x in ret.lazyop.buffers.iter_mut() {
            x.children.insert(rc.clone());
        }
        if ret._base.is_some() {
            unsafe {
                Arc::get_mut_unchecked(&mut ret._base.as_mut().unwrap())
                    .views
                    .insert(rc.clone())
            };
        } else {
            assert!(ret.st.contiguous(), "{:?}", ret.st);
        }
        ret
    }

    pub fn base(&self) -> Self {
        if self._base.is_some() {
            return self._base.as_ref().unwrap().as_ref().clone();
        }
        self.clone()
    }

    pub fn is_realized(&self) -> bool {
        self.base().device_buffer.is_some()
    }

    pub fn map_buffers(&self, real_srcs: &HashMap<&LazyBuffer, Option<LazyOpSrc>>) -> Self {
        if let Some(s) = real_srcs.get(self) {
            if let Some(ss) = s {
                return ss.lb().clone();
            }
        }
        self.clone()
    }

    pub fn loadop(
        optype: OpType,
        shape: &[isize],
        dtype: Dtype,
        device: &str,
        args: Option<Vec<Arg>>,
        src: Option<LazyBuffer>,
    ) -> Self {
        let mut ss = vec![];
        if let Some(src) = src {
            ss.push(src);
        };
        create_lazybuffer(
            device,
            ShapeTracker::new(shape, None),
            optype.clone(),
            LazyOp::new(optype, ss, args),
            dtype,
            None,
        )
    }

    pub fn _const(&self, val: impl num_traits::ToBytes + num_traits::ToPrimitive) -> Self {
        Self::loadop(
            OpType::Load(Load::Const),
            &vec![],
            self.dtype.clone(),
            &self.device,
            Some(vec![Arg::Num(val.to_le_bytes().as_ref().to_vec())]),
            None,
        )
        .reshape(&vec![1; self.shape.len()])
        .expand(&self.shape)
    }

    pub fn from_cpu<T>(x: Vec<T>) -> Self {
        todo!()
    }

    pub fn copy_to_device(&self, device: &str) -> Self {
        if !self.is_realized()
            && matches!(self.lazyop.optype, OpType::Load(_))
            && self.lazyop.optype != Load::Const
        {
            return self.clone();
        }
        Self::loadop(
            OpType::Load(Load::From),
            &self.shape,
            self.dtype.clone(),
            &self.device,
            None,
            Some(self.contiguous()),
        )
    }

    pub fn contiguous(&self) -> Self {
        if !self.is_realized()
            && matches!(self.lazyop.optype, OpType::Load(_))
            && self.lazyop.optype != Load::Const
        {
            return self.clone();
        }
        if self.st.contiguous()
            && self.base().st.size() == self.st.size()
            && !self.is_unrealized_const()
        {
            todo!()
        }
        Self::loadop(
            OpType::Load(Load::Contiguous),
            &self.shape,
            self.dtype.clone(),
            &self.device,
            None,
            Some(self.clone()),
        )
    }

    pub fn is_unrealized_const(&self) -> bool {
        !self.is_realized() && self.base().lazyop.optype == Load::Const
    }

    // pub fn schedule(&self, seen: Option<&mut HashSet<&LazyBuffer>>) -> Vec<ScheduleItem> {
    //     let mut new_seen = HashSet::new();
    //     let mut seen = if let Some(s) = seen { s } else { &mut new_seen };
    //     let mut seen: HashSet<&LazyBuffer> = HashSet::new();
    //     if seen.contains(&self) || self.is_realized() || self.is_unrealized_const() {
    //         return vec![];
    //     }
    //     seen.insert(&self);
    //     if self.base.is_none() || self.base.as_ref().unwrap().id != self.id {
    //         return self.base.as_ref().unwrap().schedule(Some(&mut seen));
    //     }
    //     let mut lazyop = if self.lazyop.optype != Load::Contiguous {
    //         self.lazyop.clone()
    //     } else {
    //         LazyOp::new(
    //             OpType::Unary(Unary::Noop),
    //             self.lazyop.src.clone(),
    //             Some(self.lazyop.args.clone()),
    //         )
    //     };
    //     if matches!(self.lazyop.optype, OpType::Binary(_)) {
    //         lazyop = _ast_binaryops(&lazyop, &self.shape)
    //     } else if matches!(self.lazyop.optype, OpType::Reduce(_)) {
    //         lazyop = _ast_reduceops(&lazyop)
    //     }
    //     let mut ret = vec![];
    //     for x in self.lazyop.buffers.iter() {
    //         ret.extend(x.schedule(Some(&mut seen)));
    //     }
    //     // WARN: - this var_vals is produced in this order: here <- Shaptracker <- Views <- Symbolic.
    //     //         This means the min/max field in Nodes can be Node/Int, which I dont think is
    //     //         needed, it also makes everything really really ugly and complicated.
    //     //         And im not sure if that is correct since when creating some of the node
    //     //         types, you need ordering, and you cant really compare string to int and decide
    //     //         what is greater/lesser, espeically when that min/max node is a var.
    //     //       - Check FIXMEs on Mul/Div/Mod node in symbolic file.
    //     //
    //     // var_vals = dict(sorted(merge_dicts([self.st.var_vals] + [buf.st.var_vals for buf in op.buffers]).items(), key=lambda kv:cast(Variable,kv[0]).key))
    //     todo!()
    // }

    pub fn realize(&self) -> Self {
        let mut ret = self.clone();
        if ret.is_realized() {
            return ret;
        }
        //println!("1 ret {:?}", ret);
        match &self.lazyop.optype {
            OpType::Binary(_) => ret.lazyop = _ast_binaryops(&ret.lazyop, &ret.shape).into(),
            OpType::Reduce(_) => ret.lazyop = _ast_reduceops(&ret.lazyop).into(),
            OpType::Load(load) => match load {
                Load::Empty => _realize_empty(self),
                Load::Rand => todo!(),
                Load::Const => _realize_const(self),
                Load::From => todo!(),
                Load::Contiguous => todo!(),
                Load::Custom => todo!(),
            },
            _ => (),
        }
        //println!("2 ret {:?}", ret);
        if !ret.is_realized() {
            for x in ret.lazyop.buffers.iter_mut() {
                *x = x.realize();
            }
        }
        //println!("3 ret {:?}", ret);
        ret
    }

    pub fn e<O: Into<OpType>>(&self, optype: O, srcs: &[Self], arg: Option<Vec<Arg>>) -> Self {
        let optype = optype.into();
        let srcs = vec![&[self.clone()], srcs].concat();
        let out_device = srcs[0].device.clone();
        let out_shape = srcs[0].shape.clone();
        let out_dtype = if srcs[0].dtype.size > srcs[1].dtype.size {
            srcs[0].dtype.clone()
        } else {
            srcs[1].dtype.clone()
        };
        // # if we are separated from other binary ops by movement ops, we push those movement ops above those binaryops
        //     if SHUFFLE_MOVEMENT_OPS: srcs = _push_movement_ops(srcs)
        //
        //     # get outputs now
        //     out_device, out_shape, out_dtype = srcs[0].device, srcs[0].shape, max([x.dtype for x in srcs]) if op != UnaryOps.CAST else cast(Tuple[Dtype, bool], arg)[0]
        //
        //     # push all contiguous to the end of BinaryOps. kernels 198 -> 196
        //     if PUSH_CONTIGUOUS and any(not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1 for x in srcs):
        //       new_srcs: List[LazyBuffer] = []
        //       for x in srcs:
        //         if not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1:
        //           x.op.src[0].children.discard(x)
        //           new_srcs.append(cast(LazyBuffer, x.op.src[0]))
        //         else:
        //           new_srcs.append(x)
        //       return new_srcs[0].e(op, *new_srcs[1:], arg=arg).contiguous()
        //
        //     if MERGE_ELEMENTWISE_OPS:
        //       # remove the buffers from any (childless) BinaryOps that feed into this
        //       srcs = tuple([x.op if x.optype == BinaryOps and not x.children and not x.realized else x for x in srcs])  # type: ignore
        create_lazybuffer(
            &out_device,
            ShapeTracker::new(&out_shape, None),
            optype.clone(),
            LazyOp::new(optype, srcs, None),
            out_dtype,
            None,
        )
    }

    pub fn _reduce_op(&self, optype: OpType, new_shape: &[isize]) -> Self {
        if self.shape == new_shape {
            return self.clone();
        }
        let srcs = _push_movement_ops(&vec![&*self]);
        let unbound_new_shape = new_shape;
        create_lazybuffer(
            &self.device,
            ShapeTracker::new(new_shape, None),
            optype.clone(),
            LazyOp::new(
                optype,
                srcs,
                Some(
                    unbound_new_shape
                        .iter()
                        .map(|i| Arg::Num(i.to_le_bytes().to_vec()))
                        .collect::<Vec<Arg>>(),
                ),
            ),
            self.dtype.clone(),
            None,
        )
    }

    pub fn r<O: Into<OpType>>(&self, optype: O, new_shape: &[isize]) -> Self {
        let optype = optype.into();
        if self.shape.iter().product::<isize>() / new_shape.iter().product::<isize>() < 32768 {
            return self._reduce_op(optype, new_shape);
        }
        let mut t = vec![];
        for (i, (&old, (&new, &stride))) in self
            .shape
            .iter()
            .zip(new_shape.iter().zip(self.st.strides().iter()))
            .enumerate()
        {
            if old == new {
                continue;
            }
            let divisor = gcd(256, old);
            let heuristic: f32 = if stride <= 0 {
                divisor as f32 / stride as f32
            } else {
                0.0
            };
            let dim_to_split = i;
            t.push((heuristic, divisor, dim_to_split));
        }
        let &(heuristic, divisor, dim_to_split) =
            t.iter().max_by(|a, b| f32::total_cmp(&a.0, &b.0)).unwrap();
        if divisor < 16 && heuristic < 0.1 {
            return self._reduce_op(optype, new_shape);
        }

        let splitted_shape = |dim_aft_div: Vec<isize>| -> Vec<isize> {
            let dim_to_split = dim_to_split as usize;
            vec![
                self.shape[..dim_to_split].to_vec(),
                vec![self.shape[dim_to_split] / divisor],
                dim_aft_div,
                self.shape[dim_to_split + 1..].to_vec(),
            ]
            .concat()
        };
        let sh1 = splitted_shape(vec![divisor]);
        let sh2 = splitted_shape(vec![1]);
        let sh3 = splitted_shape(vec![]);
        self.reshape(&sh1)
            ._reduce_op(optype.clone(), &sh2)
            .reshape(&sh3)
            ._reduce_op(optype, new_shape)
    }

    pub fn _movement_op(&self, st: ShapeTracker, optype: OpType, arg: &[isize]) -> Self {
        if matches!(self.lazyop.optype, OpType::Binary(_))
            && !self.is_realized()
            && (matches!(
                optype,
                OpType::Movement(Movement::Shrink)
                    | OpType::Movement(Movement::Stride)
                    | OpType::Movement(Movement::Permute)
            ) || (matches!(optype, OpType::Movement(Movement::Reshape))
                && matches!(self.lazyop.optype, OpType::Unary(_))))
            && self.children.is_empty()
        {
            return match optype {
                OpType::Movement(m) => match m {
                    Movement::Reshape => self.reshape(arg),
                    Movement::Expand => self.expand(arg),
                    _ => todo!(),
                },
                _ => unreachable!(),
            };
        }
        if !self.is_realized() && st.contiguous() {
            let root = get_movementroot(&*self, false);
            if root.st.contiguous()
                && root != self
                && st.shape().iter().product::<isize>() == root.shape.iter().product::<isize>()
            {
                return root.clone().reshape(&st.shape());
            }
        }
        create_lazybuffer(
            &self.device,
            st,
            optype.clone(),
            LazyOp::new(
                optype,
                vec![self.clone()],
                Some(
                    arg.iter()
                        .map(|i| Arg::Num(i.to_le_bytes().to_vec()))
                        .collect::<Vec<Arg>>(),
                ),
            ),
            self.dtype.clone(),
            Some(Arc::new(self.base())),
        )
    }

    pub fn reshape(&self, arg: &[isize]) -> Self {
        if self.shape == arg {
            return self.clone();
        }
        if !self.is_realized() && self.lazyop.optype == Movement::Reshape {
            let s_clone = self.clone();
            let mut ret = self.lazyop.src[0].clone();
            ret.lb_mut().children.remove(&s_clone);
            return ret.lb_mut().reshape(arg);
        }
        self._movement_op(
            self.st.reshape(arg),
            OpType::Movement(Movement::Reshape),
            arg,
        )
    }

    pub fn pad(&self, arg: &[isize]) -> Self {
        if arg.iter().all(|v| *v == 0) {
            return self.clone();
        }
        if !self.is_realized() && self.lazyop.optype == Movement::Pad {
            let op_arg = self
                .lazyop
                .args
                .iter()
                .map(|v| v.to_num::<isize>())
                .collect::<Vec<isize>>();
            return self.lazyop.src[0].clone().lb_mut().pad(&op_arg);
        }

        let mut aarg = vec![];
        for a in arg.windows(2).step_by(2) {
            aarg.push((a[0], a[1]))
        }
        self._movement_op(self.st.pad(&aarg), OpType::Movement(Movement::Pad), arg)
    }

    pub fn expand(&self, arg: &[isize]) -> Self {
        if arg == [1, 4, 3, 64, 3, 64] {
            panic!()
        }
        if &self.shape == arg {
            return self.clone();
        }
        if !self.is_realized() && self.lazyop.optype == Movement::Expand {
            return self.lazyop.src[0].lb().expand(arg);
        }
        return self._movement_op(self.st.expand(arg), OpType::Movement(Movement::Expand), arg);
    }

    pub fn permute(&self, arg: &[isize]) -> Self {
        if arg == &(0..arg.len()).map(|v| v as isize).collect::<Vec<isize>>() {
            return self.clone();
        }
        if !self.is_realized() {
            match &self.lazyop.optype {
                OpType::Movement(m) => match m {
                    Movement::Permute => {
                        return self.lazyop.src[0].clone().lb_mut().permute(
                            &self
                                .lazyop
                                .args
                                .iter()
                                .map(|v| v.to_num::<isize>())
                                .collect::<Vec<isize>>(),
                        )
                    }
                    Movement::Expand => {
                        return self.lazyop.src[0].clone().lb_mut().permute(arg).expand(
                            &arg.iter()
                                .map(|i| self.lazyop.args[*i as usize].to_num::<isize>())
                                .collect::<Vec<isize>>(),
                        )
                    }
                    // Movement::Reshape if matches!(self.lazyop.src[0], LazyOpSrc::LazyBuffer(_)) => {
                    //     if let Some(shape_idx_groups) =
                    //         get_contraction(&self.lazyop.src[0].lb().shape, &self.shape)
                    //     {
                    //         for i in 0..self.lazyop.src.len() {
                    //             if matches!(self.lazyop.src[i], LazyOpSrc::LazyBuffer(_))
                    //                 && self.lazyop.src[0].lb().id == self.id
                    //             {
                    //                 //self.lazyop.src.remove(i);
                    //             }
                    //         }
                    //         return self.lazyop.src[0]
                    //             .lb()
                    //             .permute(
                    //                 &arg.iter()
                    //                     .map(|&i| shape_idx_groups[i as usize].clone())
                    //                     .collect::<Vec<Vec<isize>>>()
                    //                     .concat(),
                    //             )
                    //             .reshape(&self.st.permute(arg).shape());
                    //     }
                    // }
                    _ => (),
                },
                OpType::Reduce(_) => {
                    let narg = arg
                        .iter()
                        .map(|i| self.lazyop.args[*i as usize].clone())
                        .collect::<Vec<Arg>>();
                    let s_clone = self.clone();
                    let mut src = self.lazyop.src[0].clone();
                    let optype = &self.lazyop.optype;
                    src.lb_mut().children.remove(&s_clone);
                    //return src.permute(arg).r(cast(ReduceOps, rop), narg)
                    todo!()
                }
                t => (),
            };
        }
        self._movement_op(
            self.st.permute(arg),
            OpType::Movement(Movement::Permute),
            arg,
        )
    }

    pub fn shrink(&self, arg: &[isize]) -> Self {
        if self
            .shape
            .iter()
            .zip(arg.windows(2).step_by(2))
            .all(|(sh, ab)| ab[1] - ab[0] == *sh)
        {
            return self.clone();
        }
        if !self.is_realized() && self.lazyop.optype == Movement::Shrink {
            let mut aarg = vec![];
            for (be1, be2) in self
                .lazyop
                .args
                .windows(2)
                .step_by(2)
                .zip(arg.windows(2).step_by(2))
            {
                aarg.push(be1[0].to_num::<isize>() + be2[0]);
                aarg.push(be1[0].to_num::<isize>() + be2[1]);
            }
            return self.lazyop.src[0].clone().lb_mut().shrink(&aarg);
        }
        let st = self.st.shrink(
            &arg.windows(2)
                .step_by(2)
                .map(|a| (a[0], a[1]))
                .collect::<Vec<(isize, isize)>>(),
        );
        self._movement_op(st, OpType::Movement(Movement::Shrink), arg)
    }

    pub fn stride(&self, arg: &[isize]) -> Self {
        if arg.iter().all(|i| *i == 1) {
            return self.clone();
        }
        if !self.is_realized() && self.lazyop.optype == Movement::Stride {
            return self.lazyop.src[0].clone().lb_mut().stride(
                &arg.iter()
                    .zip(self.lazyop.args.iter())
                    .map(|(a, aa)| a * aa.to_num::<isize>())
                    .collect::<Vec<isize>>(),
            );
        }
        self._movement_op(self.st.stride(arg), OpType::Movement(Movement::Stride), arg)
    }
}

pub fn create_lazybuffer(
    device: &str,
    st: ShapeTracker,
    optype: OpType,
    op: LazyOp,
    dtype: Dtype,
    base: Option<Arc<LazyBuffer>>,
) -> LazyBuffer {
    if matches!(
        optype,
        OpType::Load(Load::Empty) | OpType::Load(Load::Rand) | OpType::Load(Load::Const)
    ) {
        return LazyBuffer::new(device, st, optype, op, dtype, None, base);
    }
    // # wop is the deduping key. i feel this used to compare more deeply
    // wop = (device, dtype, optype, ref(op), ref(base) if base else None)
    // if wop in lazycache:
    //   for x in op.buffers: x.children.add(lazycache[wop])
    //   return lazycache[wop]
    //
    // lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype, base=base)
    let ret = LazyBuffer::new(device, st, optype, op, dtype, None, base);
    println!("{:?}", ret);
    ret
}

fn _ast_reduceops(op: &LazyOp) -> LazyOp {
    let src = op.src[0].lb();
    let mut ret = op.clone();
    if src.is_realized() {
        return ret;
    }
    let src = if matches!(src.lazyop.optype, OpType::Binary(_)) && src.children.len() <= 1 {
        LazyOpSrc::LazyOp(src.lazyop.clone())
    } else {
        LazyOpSrc::LazyBuffer(src.clone())
    };
    LazyOp::new(op.optype.clone(), vec![src], Some(op.args.clone()))
}

fn _ast_binaryops(op: &LazyOp, shape: &[isize]) -> LazyOp {
    let mut real_srcs: HashMap<&LazyBuffer, Option<LazyOpSrc>> = {
        let mut m = HashMap::new();
        for x in &op.buffers {
            m.insert(x, None);
        }
        m
    };
    //[(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype == ReduceOps and not x.realized and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
    let mut psrcs: Vec<(&LazyBuffer, &LazyBuffer)> = real_srcs
        .keys()
        .into_iter()
        .map(|&v| (v, get_movementroot_contiguous(v)))
        .into_iter()
        .filter(|(k, x)| {
            matches!(x.lazyop.optype, OpType::Reduce(_))
                && !x.is_realized()
                && k.shape.iter().product::<isize>() == x.shape.iter().product::<isize>()
                && x.children.len() <= 1
                && k.children.len() <= 1
        })
        .collect();
    let mut intermediate_shape = shape;
    let mut top: Option<LazyOp> = None;
    if !psrcs.is_empty() {
        let psrc = psrcs[0];
        if matches!(psrc.1.lazyop.optype, OpType::Reduce(_)) {
            top = Some(_ast_reduceops(&psrc.1.lazyop));
        }
        real_srcs.insert(
            psrc.0,
            if top.is_none() {
                None
            } else {
                Some(LazyOpSrc::LazyOp(top.clone().unwrap().into()))
            },
        );
        if top.is_some() {
            for x in top.as_ref().unwrap().buffers.iter() {
                real_srcs.insert(x, Some(LazyOpSrc::LazyBuffer(x.clone())));
            }
        };
        if psrc.0.shape != psrc.1.shape {
            intermediate_shape = shape;
        }
    }
    for (k, v) in real_srcs.iter_mut() {
        if v.is_none() {
            *v = Some(LazyOpSrc::LazyBuffer(k.reshape(intermediate_shape)));
        }
    }
    let ast = op.map_buffers(&real_srcs);
    LazyOp::new(OpType::Movement(Movement::Reshape), vec![ast], None)
}

fn get_single_root(root: &LazyBuffer) -> &LazyBuffer {
    if root.lazyop.src.len() == 1 && matches!(root.lazyop.src[0], LazyOpSrc::LazyBuffer(_)) {
        return get_single_root(root.lazyop.src[0].lb());
    };
    root
}

fn get_movementroot(root: &LazyBuffer, allow_contiguous: bool) -> &LazyBuffer {
    if !root.is_realized()
        && (matches!(root.lazyop.optype, OpType::Movement(_))
            || (matches!(root.lazyop.optype, OpType::Load(Load::Contiguous))
                && allow_contiguous
                && root.lazyop.src[0].lb().st.contiguous()))
    {
        return get_movementroot(root.lazyop.src[0].lb(), allow_contiguous);
    }
    root
}

fn get_movementroot_contiguous(x: &LazyBuffer) -> &LazyBuffer {
    if !x.is_realized() && matches!(x.lazyop.optype, OpType::Load(Load::Contiguous)) {
        return get_movementroot_contiguous(x.lazyop.src[0].lb());
    }
    if matches!(x.lazyop.optype, OpType::Movement(_)) && x.st.contiguous() {
        return get_movementroot(x, true);
    }
    x
}

fn _push_movement_ops(srcs: &[&LazyBuffer]) -> Vec<LazyBuffer> {
    let mut new_srcs = vec![];
    for &x in srcs {
        let mut mops = vec![];
        let mut bx = x;
        while !bx.is_realized()
            && matches!(bx.lazyop.optype, OpType::Movement(_))
            && bx.lazyop.optype != Movement::Expand
            && bx.children.len() <= 1
        {
            mops.push((bx.lazyop.optype.clone(), bx.lazyop.args.clone()));
            assert!(matches!(bx.lazyop.src[0], LazyOpSrc::LazyBuffer(_)));
            bx = bx.lazyop.src[0].lb();
        }
        if mops.len() > 0
            && !bx.is_realized()
            && matches!(bx.lazyop.optype, OpType::Binary(_))
            && bx.children.len() <= 1
            && mops.iter().all(|m| m.0 != Movement::Pad)
        {
            todo!()
        } else {
            new_srcs.push((*x).clone());
        }
    }
    new_srcs
}

fn get_contraction(old_shape: &[isize], new_shape: &[isize]) -> Option<Vec<Vec<isize>>> {
    let mut a = 1;
    let acc_shape = old_shape
        .iter()
        .map(|n| {
            a = a * n;
            a
        })
        .collect::<Vec<isize>>();
    let mut a = 1;
    let acc_new = new_shape
        .iter()
        .map(|n| {
            a = a * n;
            a
        })
        .collect::<Vec<isize>>();
    let mut split = Vec::new();
    for acc in acc_new {
        if acc != 1 {
            let n = {
                if let Some(n) = new_shape.iter().position(|&i| i == acc) {
                    n
                } else {
                    return None;
                }
            };
            split.push(n)
        } else {
            split.push(0)
        }
    }
    let l = split.len();
    let mut ret = Vec::new();
    let s: Vec<usize> = vec![&[0], &split[..l - 1]].concat();
    let e: Vec<usize> = vec![&split[..l - 1], &[old_shape.len()]].concat();
    for (st, ed) in izip!(s, e) {
        ret.push((st..ed).map(|n| n as isize).collect())
    }
    Some(ret)
}

fn _realize_from(buffer: &LazyBuffer) {
    let realized_buffer = buffer.lazyop.src[0].lb().realize();
    assert!(realized_buffer.is_realized());
    todo!()
}

fn _realize_empty(buffer: &LazyBuffer) {
    let mut buffer = buffer.clone(); // LazyBuffer is cheap to clone, underlying device buffer is shared through Arc
    unsafe {
        let x = Arc::get_mut_unchecked(&mut buffer.device_buffer);
        x.replace(DEVICE.alloc(
            buffer.shape.iter().product::<isize>() as usize,
            buffer.dtype,
        ));
    }
}

fn _realize_const(buffer: &LazyBuffer) {
    let mut buffer = buffer.clone(); // LazyBuffer is cheap to clone, underlying device buffer is shared through Arc
    unsafe {
        if let Arg::Num(bytes) = &buffer.lazyop.args[0] {
            let mut b = DEVICE.alloc(1, buffer.dtype);
            Arc::get_mut_unchecked(&mut b).from_cpu(bytes);
            Arc::get_mut_unchecked(&mut buffer.device_buffer).replace(b);
        } else {
        }
    }
}
