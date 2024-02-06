use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use half::f16;
use itertools::Itertools;
use rand::Rng;

use crate::codegen::kernel::Buffers;
use crate::codegen::kernel::{ConstBuffer, MemBuffer};
use crate::dtype::{least_upper_dtype, NumType};
use crate::ops::{self, ScheduleItem};
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

impl core::fmt::Display for LazyBufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
    pub st: ShapeTracker,
    pub device_buffer: Arc<Option<Arc<dyn Buffer>>>,
    pub _base: Option<Arc<LazyBuffer>>,
    pub shape: Vec<isize>,
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
            "<LB {:?} dtype={:?} op={:?} st={:?}>",
            self.shape, self.dtype.type_name, self.lazyop.optype, self.st.views
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
        base: Option<Arc<LazyBuffer>>,
    ) -> Self {
        let mut ret = Self {
            device: device.into(),
            lazyop: op.into(),
            shape: st.shape_vec(),
            st: st.into(),
            // children: HashSet::new(),
            // views: HashSet::new(),
            id: lb_id(),
            dtype,
            device_buffer: if base.is_some() {
                base.as_ref().unwrap().device_buffer.clone()
            } else {
                Arc::new(None)
            },
            _base: base,
        };
        let rc = ret.clone();
        for x in ret.lazyop.buffers.iter_mut() {
            // unsafe {
            //     Arc::get_mut_unchecked(x).children.insert(rc.clone());
            // }
        }
        if ret._base.is_some() {
            // unsafe {
            //     Arc::get_mut_unchecked(&mut ret._base.as_mut().unwrap())
            //         .views
            //         .insert(rc.clone())
            // };
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

    pub fn map_buffers(&self, real_srcs: &HashMap<LazyBuffer, LazyOpSrc>) -> LazyOpSrc {
        if let Some(s) = real_srcs.get(self) {
            return s.clone();
        }
        self.clone().into()
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
            ss.push(src.into());
        };
        create_lazybuffer(
            device,
            ShapeTracker::new(shape, None),
            LazyOp::new(optype, ss, args),
            dtype,
            None,
        )
    }

    pub fn _const(val: impl Display, dtype: Dtype, device: &str) -> Self {
        Self::loadop(
            OpType::Load(Load::Const),
            &vec![1],
            dtype,
            device,
            Some(vec![Arg::Str(val.to_string())]),
            None,
        )
    }

    pub fn const_like(&self, val: impl Display) -> Self {
        Self::loadop(
            OpType::Load(Load::Const),
            &vec![1],
            self.dtype.clone(),
            &self.device,
            Some(vec![Arg::Str(val.to_string())]),
            None,
        )
        .reshape(&vec![1; self.shape.len()])
        .expand(&self.shape)
    }

    pub fn from_bytes(x: &[u8]) -> Self {
        let bytes = x;
        let mut buf = DEVICE.alloc(x.len(), dtype::type_to_dtype::<u8>());
        DEVICE.copyin(bytes.to_vec(), &*buf);
        Self {
            lazyop: LazyOp::new(Load::From.into(), vec![], None).into(),
            st: ShapeTracker::from_shape(&[x.len() as isize]).into(),
            device_buffer: Arc::new(Some(buf)),
            _base: None,
            shape: vec![x.len() as isize],
            // children: HashSet::new(),
            // views: HashSet::new(),
            id: lb_id(),
            dtype: dtype::type_to_dtype::<u8>(),
            device: "GPU".into(),
        }
    }

    pub fn from_cpu<T: NumType>(x: Vec<T>) -> Self {
        let bytes = x
            .iter()
            .map(|x| x._to_le_bytes())
            .collect::<Vec<Vec<u8>>>()
            .concat();
        let mut buf = DEVICE.alloc(x.len(), dtype::type_to_dtype::<T>());
        DEVICE.copyin(bytes, &*buf);
        Self {
            lazyop: LazyOp::new(Load::From.into(), vec![], None).into(),
            st: ShapeTracker::from_shape(&[x.len() as isize]).into(),
            device_buffer: Arc::new(Some(buf)),
            _base: None,
            shape: vec![x.len() as isize],
            // children: HashSet::new(),
            // views: HashSet::new(),
            id: lb_id(),
            dtype: dtype::type_to_dtype::<T>(),
            device: "GPU".into(),
        }
    }

    // pub fn copy_to_device(&self, device: &str) -> Self {
    //     if !self.is_realized()
    //         && matches!(self.lazyop.optype, OpType::Load(_))
    //         && self.lazyop.optype != Load::Const
    //     {
    //         return self.clone();
    //     }
    //     Self::loadop(
    //         OpType::Load(Load::From),
    //         &self.shape,
    //         self.dtype.clone(),
    //         &self.device,
    //         None,
    //         Some(self.contiguous()),
    //     )
    // }

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
            return create_lazybuffer(
                &self.device,
                ShapeTracker::from_shape(&self.shape),
                LazyOp::new(
                    OpType::Load(Load::Contiguous),
                    vec![LazyOpSrc::LazyBuffer(Arc::new(self.clone()))],
                    None,
                ),
                self.dtype.clone(),
                self._base.clone(),
            );
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

    pub fn schedule(&self, mut seen: &mut HashSet<Self>) -> VecDeque<ScheduleItem> {
        if seen.contains(self) || self.is_realized() || self.is_unrealized_const() {
            return VecDeque::new();
        }
        seen.insert(self.clone());
        if self.base() != *self {
            return self.base().schedule(seen);
        }

        let mut op = (*self.lazyop).clone();
        if matches!(self.lazyop.optype, OpType::Binary(_)) {
            op = _ast_binaryops(&op, &self.shape);
        } else if matches!(self.lazyop.optype, OpType::Reduce(_)) {
            op = _ast_reduceops(&op);
        }

        let mut ret = VecDeque::new();
        for x in op.buffers.iter() {
            ret.extend(x.schedule(seen))
        }

        let (mut op, base_bufs) = _replace_bufferops(op);
        //println!("{} {:?}", base_bufs.len(), base_bufs);
        if !matches!(op.optype, OpType::Load(_)) {
            let info = get_lazyop_info(&op.clone().into());
            op = LazyOp::new(
                OpType::Buffer(ops::Buffer::Store),
                vec![op.clone().into()],
                Some(vec![Arg::Buffer(
                    MemBuffer {
                        idx: 0,
                        dtype: self.dtype.clone(),
                        st: ShapeTracker::from_shape(&info.shape),
                    }
                    .into(),
                )]),
            );
        }
        ret.push_back(ScheduleItem {
            ast: op,
            out: self.clone(),
            inputs: base_bufs,
        });
        ret
    }

    pub fn e<O: Into<OpType>>(&self, optype: O, srcs: &[Self], arg: Option<Vec<Arg>>) -> Self {
        let optype = optype.into();
        let mut srcs: Vec<LazyBuffer> = vec![&[self.clone()], srcs].concat();
        srcs = _push_movement_ops(&srcs.iter().map(|s| s).collect::<Vec<&Self>>());
        let out_device = srcs[0].device.clone();
        let out_shape = srcs[0].shape.clone();
        let out_dtype = v![s, for s in srcs.iter()]
            .iter()
            .max_by(|x, y| x.dtype.size.cmp(&y.dtype.size))
            .unwrap()
            .dtype
            .clone();
        let srcs: Vec<LazyOpSrc> = srcs
            .into_iter()
            .map(|x| {
                // if matches!(x.lazyop.optype, OpType::Binary(_))
                //     && !x.is_realized()
                // {
                //     LazyOpSrc::LazyOp(x.lazyop)
                // } else {
                x.into()
                //}
            })
            .collect();
        create_lazybuffer(
            &out_device,
            ShapeTracker::new(&out_shape, None),
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
            LazyOp::new(
                optype,
                srcs.into_iter().map(|s| s.into()).collect(),
                Some(vec![Arg::Shape(unbound_new_shape.to_vec())]),
            ),
            self.dtype.clone(),
            None,
        )
    }

    pub fn r<O: Into<OpType>>(&self, optype: O, new_shape: &[isize]) -> Self {
        let optype = optype.into();
        self._reduce_op(optype, new_shape)
        // if self.shape.iter().product::<isize>() / new_shape.iter().product::<isize>() < 32768 {
        //     return self._reduce_op(optype, new_shape);
        // }
        // let mut t = vec![];
        // for (i, (&old, (&new, &stride))) in self
        //     .shape
        //     .iter()
        //     .zip(new_shape.iter().zip(self.st.strides().iter()))
        //     .enumerate()
        // {
        //     if old == new {
        //         continue;
        //     }
        //     let divisor = gcd(256, old);
        //     let heuristic: f32 = if stride <= 0 {
        //         divisor as f32 / stride as f32
        //     } else {
        //         0.0
        //     };
        //     let dim_to_split = i;
        //     t.push((heuristic, divisor, dim_to_split));
        // }
        // let &(heuristic, divisor, dim_to_split) =
        //     t.iter().max_by(|a, b| f32::total_cmp(&a.0, &b.0)).unwrap();
        // if divisor < 16 && heuristic < 0.1 {
        //     return self._reduce_op(optype, new_shape);
        // }
        //
        // let splitted_shape = |dim_aft_div: Vec<isize>| -> Vec<isize> {
        //     let dim_to_split = dim_to_split as usize;
        //     vec![
        //         self.shape[..dim_to_split].to_vec(),
        //         vec![self.shape[dim_to_split] / divisor],
        //         dim_aft_div,
        //         self.shape[dim_to_split + 1..].to_vec(),
        //     ]
        //     .concat()
        // };
        // let sh1 = splitted_shape(vec![divisor]);
        // let sh2 = splitted_shape(vec![1]);
        // let sh3 = splitted_shape(vec![]);
        // self.reshape(&sh1)
        //     ._reduce_op(optype.clone(), &sh2)
        //     .reshape(&sh3)
        //     ._reduce_op(optype, new_shape)
    }

    pub fn _movement_op(&self, st: ShapeTracker, optype: OpType, arg: &[isize]) -> Self {
        // if !self.is_realized()
        //     && matches!(self.lazyop.optype, OpType::Binary(_))
        //     && self.children.len() == 0
        // {
        //     if matches!(
        //         optype,
        //         OpType::Movement(Movement::Shrink)
        //             | OpType::Movement(Movement::Stride)
        //             | OpType::Movement(Movement::Permute)
        //     ) || (matches!(optype, OpType::Movement(Movement::Reshape))
        //         && matches!(self.lazyop.optype, OpType::Unary(_)))
        //     {
        //         return replace_with_movement_ops(
        //             &(*self.lazyop.0).clone().into(),
        //             &[(self.lazyop.optype.clone(), arg.to_vec())],
        //         );
        //     }
        // }
        // if matches!(self.lazyop.optype, OpType::Binary(_))
        //     && !self.is_realized()
        //     && (matches!(
        //         optype,
        //         OpType::Movement(Movement::Shrink)
        //             | OpType::Movement(Movement::Stride)
        //             | OpType::Movement(Movement::Permute)
        //     && self.children.is_empty()
        // {
        //     return replace_with_movement_ops(
        //         &(*self.lazyop.0).clone().into(),
        //         &[(self.lazyop.optype.clone(), arg.to_vec())],
        //     );
        //     // return match optype {
        //     //     OpType::Movement(m) => match m {
        //     //         Movement::Reshape => self.reshape(arg),
        //     //         Movement::Expand => self.expand(arg),
        //     //         _ => unreachable!(),
        //     //     },
        //     //     _ => unreachable!(),
        //     // };
        // }
        assert!(!st.shape_vec().is_empty());
        if !self.is_realized() && st.contiguous() {
            let root = get_movementroot(&*self, false);
            if root.st.contiguous()
                && root != self
                && st.shape_vec().iter().product::<isize>() == root.shape.iter().product::<isize>()
            {
                return root.reshape(&st.shape_vec());
            }
        }
        create_lazybuffer(
            &self.device,
            st,
            LazyOp::new(
                optype,
                vec![self.clone().into()],
                Some(arg.iter().map(|i| Arg::Idx(*i)).collect::<Vec<Arg>>()),
            ),
            self.dtype.clone(),
            Some(Arc::new(self.base())),
        )
    }

    pub fn reshape(&self, arg: &[isize]) -> Self {
        //assert!(!arg.is_empty());
        if self.shape == arg {
            return self.clone();
        }
        if !self.is_realized() && self.lazyop.optype == Movement::Reshape {
            let s_clone = self.clone();
            let mut ret = self.lazyop.src[0].clone();
            //ret.lb_mut().children.remove(&s_clone);
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
                .map(|v| v.to_idx())
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
        // if !self.is_realized() {
        //     match &self.lazyop.optype {
        //         OpType::Movement(m) => match m {
        //             Movement::Permute => {
        //                 return self.lazyop.src[0].clone().lb_mut().permute(
        //                     &self
        //                         .lazyop
        //                         .args
        //                         .iter()
        //                         .map(|v| v.to_idx())
        //                         .collect::<Vec<isize>>(),
        //                 )
        //             }
        //             Movement::Expand => {
        //                 return self.lazyop.src[0].lb().permute(arg).expand(
        //                     &arg.iter()
        //                         .map(|i| self.lazyop.args[*i as usize].to_idx())
        //                         .collect::<Vec<isize>>(),
        //                 );
        //             }
        //             Movement::Reshape if matches!(self.lazyop.src[0], LazyOpSrc::LazyBuffer(_)) => {
        //                 if let Some(shape_idx_groups) =
        //                     get_contraction(&self.lazyop.src[0].lb().shape, &self.shape)
        //                 {
        //                     //self.lazyop.clone().src[0].lb_mut().children.remove(self);
        //                     return self.lazyop.src[0]
        //                         .lb()
        //                         .permute(
        //                             &arg.iter()
        //                                 .map(|&i| shape_idx_groups[i as usize].clone())
        //                                 .collect::<Vec<Vec<isize>>>()
        //                                 .concat(),
        //                         )
        //                         .reshape(&self.st.permute(arg).shape());
        //                 }
        //             }
        //             _ => (),
        //         },
        //         OpType::Reduce(_) => {
        //             let arg_shape = self.lazyop.args[0].to_shape();
        //             let narg = arg
        //                 .iter()
        //                 .map(|i| arg_shape[*i as usize])
        //                 .collect::<Vec<isize>>();
        //             let mut src = self.lazyop.src[0].clone();
        //             let optype = &self.lazyop.optype;
        //             //src.lb_mut().children.remove(self);
        //             return src.lb().permute(arg).r(optype.clone(), &narg);
        //         }
        //         t => (),
        //     };
        // }
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
                aarg.push(be1[0].to_idx() + be2[0]);
                aarg.push(be1[0].to_idx() + be2[1]);
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
                    .map(|(a, aa)| a * aa.to_idx())
                    .collect::<Vec<isize>>(),
            );
        }
        self._movement_op(self.st.stride(arg), OpType::Movement(Movement::Stride), arg)
    }
}

pub fn create_lazybuffer(
    device: &str,
    st: ShapeTracker,
    op: LazyOp,
    dtype: Dtype,
    base: Option<Arc<LazyBuffer>>,
) -> LazyBuffer {
    let optype = op.optype.clone();
    if matches!(
        optype,
        OpType::Load(Load::Empty) | OpType::Load(Load::Rand) | OpType::Load(Load::Const)
    ) {
        let ret = LazyBuffer::new(device, st, optype, op, dtype, base);
        if DEBUG.0.contains("LB") {
            println!("{} {:?}", ret.id, ret);
        }
        return ret;
    }
    // # wop is the deduping key. i feel this used to compare more deeply
    // wop = (device, dtype, optype, ref(op), ref(base) if base else None)
    // if wop in lazycache:
    //   for x in op.buffers: x.children.add(lazycache[wop])
    //   return lazycache[wop]
    //
    // lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype, base=base)
    let ret = LazyBuffer::new(device, st, optype, op, dtype, base);
    if DEBUG.0.contains("LB") {
        println!("{} {:?}", ret.id, ret);
    }
    ret
}

fn _ast_reduceops(op: &LazyOp) -> LazyOp {
    let src = op.src[0].lb();
    let mut ret = op.clone();
    if src.is_realized() {
        return ret;
    }
    let src = if matches!(src.lazyop.optype, OpType::Binary(_)) {
        LazyOpSrc::LazyOp(src.lazyop.clone())
    } else {
        LazyOpSrc::LazyBuffer(Arc::new(src.clone()))
    };
    LazyOp::new(op.optype.clone(), vec![src], Some(op.args.clone()))
}

fn _ast_binaryops(op: &LazyOp, shape: &[isize]) -> LazyOp {
    let mut real_srcs: HashMap<&LazyBuffer, Option<LazyOpSrc>> = {
        let mut m = HashMap::new();
        for x in &op.buffers {
            m.insert(x.as_ref(), None);
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
            intermediate_shape = &psrc.1.shape;
        }
    }
    for (k, v) in real_srcs.iter_mut() {
        if v.is_none() {
            *v = Some(LazyOpSrc::LazyBuffer(Arc::new(
                k.reshape(intermediate_shape),
            )));
        }
    }
    let mut tmp = HashMap::new();
    for (lb, op) in real_srcs {
        if op.is_none() {
            continue;
        }
        tmp.insert(lb.clone(), op.unwrap());
    }
    let ast = op.map_buffers(&tmp);
    LazyOp::new(OpType::Movement(Movement::Reshape), vec![ast.into()], None)
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
        {
            mops.push((bx.lazyop.optype.clone(), bx.lazyop.args.clone()));
            assert!(matches!(bx.lazyop.src[0], LazyOpSrc::LazyBuffer(_)));
            bx = bx.lazyop.src[0].lb();
        }
        if mops.len() > 0
            && !bx.is_realized()
            && matches!(bx.lazyop.optype, OpType::Binary(_))
            && mops.iter().all(|m| m.0 != Movement::Pad)
        {
            new_srcs.push((*x).clone());
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
    todo!()
}

fn _realize_empty(buffer: &LazyBuffer) {
    let mut buffer = buffer.clone();
    unsafe {
        let x = Arc::get_mut_unchecked(&mut buffer.device_buffer);
        x.replace(DEVICE.alloc(
            buffer.shape.iter().product::<isize>() as usize,
            buffer.dtype.clone(),
        ));
    }
}

fn _realize_rand(buffer: &LazyBuffer) {
    let mut buffer = buffer.clone();
    let numel = buffer.shape.iter().product::<isize>() as usize;
    let mut on_cpu = gen_rand_num_bytes(numel, &buffer.dtype);
    unsafe {
        let mut b = DEVICE.alloc(numel, buffer.dtype.clone());
        Arc::get_mut_unchecked(&mut b).from_cpu(on_cpu);
        Arc::get_mut_unchecked(&mut buffer.device_buffer).replace(b);
    }
}

// fn _realize_const(buffer: &LazyBuffer) {
//     let mut buffer = buffer.clone();
//     unsafe {
//         if let Arg::Num(bytes) = &buffer.lazyop.args[0] {
//             let mut b = DEVICE.alloc(1, buffer.dtype);
//             Arc::get_mut_unchecked(&mut b).from_cpu(bytes.to_vec());
//             Arc::get_mut_unchecked(&mut buffer.device_buffer).replace(b);
//         } else {
//         }
//     }
// }

fn _realize_contiguous(buffer: &LazyBuffer) {
    todo!();
}

// Have to do this because the lack of num trait in Rust.
// num_traits's traits are not object safe.
#[rustfmt::skip]
fn gen_rand_num_bytes(size: usize, dtype: &Dtype) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let ptr = match dtype.type_name {
        "f16" => { let mut ret = (0..size).map(|_| rng.gen::<f16>()).collect::<Vec<f16>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "f32" => { let mut ret = (0..size).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "f64" => { let mut ret = (0..size).map(|_| rng.gen::<f64>()).collect::<Vec<f64>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "u8"  => { let mut ret = (0..size).map(|_| rng.gen::< u8>()).collect::<Vec< u8>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "u16" => { let mut ret = (0..size).map(|_| rng.gen::<u16>()).collect::<Vec<u16>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "u32" => { let mut ret = (0..size).map(|_| rng.gen::<u32>()).collect::<Vec<u32>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "u64" => { let mut ret = (0..size).map(|_| rng.gen::<u64>()).collect::<Vec<u64>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "i8"  => { let mut ret = (0..size).map(|_| rng.gen::< i8>()).collect::<Vec< i8>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "i16" => { let mut ret = (0..size).map(|_| rng.gen::<i16>()).collect::<Vec<i16>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "i32" => { let mut ret = (0..size).map(|_| rng.gen::<i32>()).collect::<Vec<i32>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        "i64" => { let mut ret = (0..size).map(|_| rng.gen::<i64>()).collect::<Vec<i64>>(); let ret_ptr = ret.as_mut_ptr() as *mut u8; std::mem::forget(ret); ret_ptr},
        t => panic!("unable gen type t={t}"),
    };
    unsafe { Vec::<u8>::from_raw_parts(ptr, size * dtype.size, size * dtype.size)}
}

pub fn _replace_bufferops(op: LazyOp) -> (LazyOp, Vec<LazyBuffer>) {
    let mut replacements: HashMap<LazyBuffer, LazyOp> = HashMap::new();
    let mut base_bufs: Vec<LazyBuffer> =
        v![x.base(), for x in op.buffers.iter(), if !x.is_unrealized_const()]
            .into_iter()
            .unique()
            .collect();
    for x in op.buffers.iter() {
        let st = x.st.simplify();
        if base_bufs.contains(&x.base()) {
            replacements.insert(
                (**x).clone(),
                LazyOp::new(
                    OpType::Buffer(ops::Buffer::Load),
                    vec![],
                    Some(vec![Arg::Buffer(
                        MemBuffer {
                            idx: base_bufs.iter().position(|b| b == &x.base()).unwrap() + 1,
                            dtype: x.dtype.clone(),
                            st,
                        }
                        .into(),
                    )]),
                ),
            );
        } else if !x.is_realized() && matches!(x.base().lazyop.optype, OpType::Load(Load::Const)) {
            replacements.insert(
                (**x).clone(),
                LazyOp::new(
                    OpType::Buffer(ops::Buffer::Const),
                    vec![],
                    Some(vec![Arg::Buffer(
                        ConstBuffer {
                            val: x.base().lazyop.args[0].to_str(),
                            dtype: x.dtype.clone(),
                            st,
                        }
                        .into(),
                    )]),
                ),
            );
        } else {
            panic!("not handled {x:?}");
        }
    }
    let mut tmp = HashMap::new();
    for (k, v) in replacements {
        tmp.insert(k, v.into());
    }
    let ret_op = if matches!(
        op.optype,
        OpType::Movement(Movement::Reshape) | OpType::Load(Load::Contiguous)
    ) {
        op.src[0].map_buffers(&tmp)
    } else {
        op.map_buffers(&tmp)
    };

    (ret_op.lo().clone(), base_bufs)
}

#[derive(Debug, Clone)]
pub struct KernelCache {
    prg_str: String,
    prg: Arc<dyn Program>,
    global_size: Vec<usize>,
    local_size: Vec<usize>,
}

unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

lazy_static::lazy_static! {
    pub static ref KERNEL_CACHED: Mutex<HashMap<String, KernelCache>>  = Default::default();
}

pub fn run_schedule(mut schedule: VecDeque<ScheduleItem>) {
    //TODO: Need to "copyin/out" here to avoid alloc data to new buf instead of bufs that are
    //already allocated.
    let debug_cache = DEBUG.0.contains("CACHE");
    let debug_kernel = DEBUG.0.contains("KERNEL");
    let debug_sch = DEBUG.0.contains("SCH");
    while !schedule.is_empty() {
        let mut si = schedule.pop_front().unwrap();
        if debug_sch {
            println!("{:?}", si);
        }
        if DEBUG.0.contains("OP") {
            println!("{:?}\n", si.out.lazyop.optype);
        }
        //println!("si optype {:?}", si.ast.optype);
        for x in si.inputs.iter() {
            if !x.is_realized() {
                panic!("Can't run schedule, {x:?} isnt't realized")
            }
        }
        match &si.ast.optype {
            OpType::Load(l) => {
                match l {
                    Load::Rand => _realize_rand(&si.out),
                    Load::From => _realize_from(&si.out),
                    Load::Custom => todo!(),
                    _ => (),
                }
                //println!("allocating mem for lb id:{}", si.out.id);
                continue;
            }
            _ => (),
        }
        if si.out.device_buffer.is_none() {
            _realize_empty(&si.out);
        }
        si.out.lazyop.src.clear();
        si.out.lazyop.buffers.clear();
        let mut bufs = vec![(*si.out.device_buffer).as_ref().unwrap().clone()];
        bufs.extend(v![(*b.device_buffer).as_ref().unwrap().clone(), for b in si.inputs.iter()]);
        let cached = KERNEL_CACHED
            .lock()
            .unwrap()
            .get(&format!("{:?}", si))
            .map(|v| (*v).clone());
        if let Some(kernel) = cached {
            if debug_cache {
                println!("\ncached hit");
            }
            if DEBUG.0.contains("KERNEL") {
                println!("{}", kernel.prg_str);
            }
            kernel.prg.run(
                &bufs,
                kernel.global_size.as_ref(),
                Some(kernel.local_size.as_ref()),
                &[],
                &[],
            );
        } else {
            let mut lin = DEVICE.get_lin(si.ast.clone());
            lin.linearize();
            let global_size = if let Some(mut gs) = lin.global_size.clone() {
                gs.extend(vec![1; 3 - gs.len()]);
                gs
            } else {
                vec![]
            };
            let local_size = if let Some(mut ls) = lin.local_size.clone() {
                ls.extend(vec![1; 3 - ls.len()]);
                ls
            } else {
                vec![]
            };
            let (name, prg_str) = DEVICE.render(lin);
            if debug_kernel {
                println!("{prg_str}");
            }
            if debug_cache {
                println!("\nzero hit");
            }
            let prg = DEVICE.build(&name, &prg_str);
            prg.run(&bufs, &global_size, Some(&local_size), &[], &[]);
            KERNEL_CACHED.lock().unwrap().insert(
                format!("{:?}", si),
                KernelCache {
                    prg_str,
                    prg,
                    global_size: global_size.clone(),
                    local_size: local_size.clone(),
                },
            );
        }
    }
}

#[derive(Default, Debug)]
pub struct FlopCounter {
    shape: Vec<isize>,
    dtype: Dtype,
    flops: f64,
    mem: HashMap<usize, usize>,
}

impl FlopCounter {
    pub fn mem_estimate(&self) -> usize {
        self.mem.values().sum()
    }

    fn buffer_load(arg: &Buffers) -> Self {
        Self {
            shape: arg.st().shape_vec(),
            dtype: arg.dtype(),
            flops: 0.,
            mem: HashMap::from([(arg.idx(), arg.dtype().size * arg.st().size() as usize)]),
        }
    }
    fn buffer_const(arg: &Buffers) -> Self {
        Self {
            shape: arg.st().shape_vec(),
            dtype: arg.dtype(),
            flops: 0.,
            mem: HashMap::new(),
        }
    }
    fn buffer_store(self, arg: &Buffers) -> Self {
        Self {
            shape: arg.st().shape_vec(),
            dtype: arg.dtype(),
            flops: self.flops,
            mem: self.mem,
        }
    }
    fn unary_cast(self, arg: Dtype) -> Self {
        Self {
            shape: self.shape,
            dtype: arg,
            flops: self.flops,
            mem: self.mem,
        }
    }

    fn unary(self) -> Self {
        Self {
            dtype: self.dtype,
            flops: self.flops * self.shape.iter().product::<isize>() as f64,
            shape: self.shape,
            mem: self.mem,
        }
    }

    fn binary(mut self, y: Self) -> Self {
        self.mem.extend(y.mem);
        Self {
            flops: self.flops + y.flops + self.shape.iter().product::<isize>() as f64,
            shape: self.shape,
            dtype: least_upper_dtype(&[self.dtype.clone(), y.dtype.clone()]),
            mem: self.mem,
        }
    }
    fn reduce(self, new_shape: &[isize]) -> Self {
        Self {
            shape: new_shape.to_vec(),
            dtype: self.dtype,
            flops: self.flops * self.shape.iter().product::<isize>() as f64,
            mem: self.mem,
        }
    }
    fn ternary_where(mut self, y: Self, z: Self) -> Self {
        self.mem.extend(y.mem);
        self.mem.extend(z.mem);
        Self {
            flops: self.flops + y.flops + z.flops + self.shape.iter().product::<isize>() as f64,
            shape: self.shape,
            dtype: least_upper_dtype(&[y.dtype.clone(), z.dtype.clone()]),
            mem: self.mem,
        }
    }
}

// Well good luck to me debugging this if there are any. LMAO
pub fn get_lazyop_info(ast: &LazyOpSrc) -> FlopCounter {
    let srcs = vec![vec![ast.clone()], ast.src()].concat();
    for o in srcs {
        match o.optype() {
            OpType::Unary(_) => return get_lazyop_info(&o.lo().src[0]).unary(),
            OpType::Binary(b) => {
                //println!("-------------------{}", o.lo().src[0].src().len());
                //let Buffers::LazyBuffer(lb) = o.lo().src[0].lo().args[0].to_buf() else { panic!() };
                return get_lazyop_info(&o.lo().src[0]).binary(get_lazyop_info(&o.lo().src[1]));
            }
            OpType::Reduce(_) => {
                //println!("REDUCE REDUCE REDUCE\n{:?}", o);
                return get_lazyop_info(&o.lo().src[0]).reduce(&o.lo().args[0].to_shape());
            }
            OpType::Ternary(t) => match t {
                ops::Ternary::Where => {
                    return get_lazyop_info(&o.lo().src[0]).ternary_where(
                        get_lazyop_info(&o.lo().src[1]),
                        get_lazyop_info(&o.lo().src[2]),
                    )
                }
                t => println!("{t:?}"),
            },
            OpType::Buffer(b) => match b {
                ops::Buffer::Load => {
                    //println!("{:?}", o);
                    return FlopCounter::buffer_load(&o.lo().args[0].to_buf());
                }
                ops::Buffer::Store => {
                    return get_lazyop_info(&o).buffer_store(&o.lo().args[0].to_buf())
                }
                ops::Buffer::Const => {
                    //println!("CONST CONST CONST\n{:?}", o);
                    return FlopCounter::buffer_const(&o.lo().args[0].to_buf());
                }
                t => println!("{t:?}"),
            },
            t => println!("{t:?}"),
        }
    }
    unreachable!()
}

pub fn replace_with_movement_ops(src: &LazyOpSrc, ops: &[(OpType, Vec<isize>)]) -> LazyBuffer {
    match src {
        LazyOpSrc::LazyOp(src) => {
            assert!(matches!(
                src.optype,
                OpType::Unary(_) | OpType::Binary(_) | OpType::Ternary(_)
            ));
            let srcs = v![replace_with_movement_ops(z, ops), for z in src.src.iter()];
            return srcs[0].e(
                src.optype.clone(),
                &srcs[1..]
                    .iter()
                    .map(|b| b.clone())
                    .collect::<Vec<LazyBuffer>>(),
                Some(src.args.clone()),
            );
        }
        LazyOpSrc::LazyBuffer(y) => {
            let mut y = (**y).clone();
            for (op, arg) in ops {
                match op {
                    OpType::Movement(m) => match m {
                        Movement::Reshape => y = y.reshape(arg),
                        Movement::Permute => y = y.permute(arg),
                        Movement::Pad => y = y.pad(arg),
                        Movement::Expand => y = y.expand(arg),
                        Movement::Shrink => y = y.shrink(arg),
                        Movement::Stride => y = y.stride(arg),
                        Movement::AsStrided => todo!(),
                    },
                    t => (),
                }
            }
            return y;
        }
    }
}
