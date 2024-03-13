use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
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
    pub force_realize: bool,
    pub contiguous_child: Arc<Option<(Self, ShapeTracker)>>,
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
        st: ShapeTracker,
        optype: OpType,
        op: LazyOp,
        dtype: Dtype,
        base: Option<Arc<LazyBuffer>>,
    ) -> Self {
        let mut ret = Self {
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
            force_realize: false,
            contiguous_child: Arc::new(None),
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

    pub fn device(&self) -> String {
        if let Some(d) = self.device_buffer.as_ref() {
            d.device()
        } else {
            "Unrealized".into()
        }
    }

    pub fn base(&self) -> Self {
        if self._base.is_some() {
            return self._base.as_ref().unwrap().as_ref().clone();
        }
        self.clone()
    }

    pub fn base_ref(&self) -> &Self {
        if self._base.is_some() {
            return self._base.as_ref().unwrap();
        }
        self
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
        args: Option<Vec<Arg>>,
        src: Option<LazyBuffer>,
    ) -> Self {
        let mut ss = vec![];
        if let Some(src) = src {
            ss.push(src.into());
        };
        create_lazybuffer(
            ShapeTracker::new(shape, None),
            LazyOp::new(optype, ss, args),
            dtype,
            None,
        )
    }

    pub fn _const(val: impl Display, dtype: Dtype) -> Self {
        Self::loadop(
            OpType::Load(Load::Const),
            &vec![1],
            dtype,
            Some(vec![Arg::Str(val.to_string())]),
            None,
        )
    }

    pub fn const_like(&self, val: impl Display) -> Self {
        Self::loadop(
            OpType::Load(Load::Const),
            &vec![1],
            self.dtype.clone(),
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
            force_realize: false,
            contiguous_child: Arc::new(None),
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
            force_realize: false,
            contiguous_child: Arc::new(None),
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
        if !self.st.contiguous()
            || self.st.size() != self.base_ref().st.size()
            || self.is_unrealized_const()
        {
            let ret = self.e(Load::Contiguous, &[], None);
            if let Some(sti) = self.st.invert(&self.base_ref().shape) {
                unsafe {
                    let mut arc_clone = self.base().contiguous_child.clone();
                    let arc_mut = Arc::get_mut_unchecked(&mut arc_clone);
                    arc_mut.replace((ret.clone(), sti));
                }
            }
            return ret;
        }
        self.clone()
    }

    pub fn is_unrealized_const(&self) -> bool {
        !self.is_realized() && self.base().lazyop.optype == Load::Const
    }

    pub fn schedule(&self, mut seen: &mut HashSet<Self>) -> VecDeque<ScheduleItem> {
        create_schedule(vec![self], None).into()
    }

    pub fn _view(&self, op: Movement, new_st: ShapeTracker) -> Self {
        if self.st.size() == 0 {
            return Self::_const(0, self.dtype.clone()).reshape(&new_st.shape().dims);
        }
        if new_st.contiguous() && self.base_ref().shape == new_st.shape().dims {
            self.base()
        } else {
            create_lazybuffer(
                new_st,
                LazyOp::new(OpType::Movement(op), vec![], None),
                self.dtype.clone(),
                Some(Arc::new(self.base())),
            )
        }
    }

    pub fn e<O: Into<OpType>>(&self, optype: O, in_srcs: &[Self], arg: Option<Vec<Arg>>) -> Self {
        let optype = optype.into();
        let mut srcs: Vec<LazyBuffer> = vec![];
        for &s in vec![vec![self], in_srcs.iter().collect()].concat().iter() {
            if s == s.base_ref()
                && s.base_ref().contiguous_child.is_some()
                && let Some(root) = s.base_ref().contiguous_child.as_ref()
            {
                srcs.push(root.0._view(Movement::Reshape, root.1.clone()));
            } else {
                srcs.push(s.clone());
            }
        }
        // srcs = _push_movement_ops(&srcs.iter().map(|s| s).collect::<Vec<&Self>>());
        // let out_device = srcs[0].device.clone();
        // let out_shape = srcs[0].shape.clone();
        // let out_dtype = v![s, for s in srcs.iter()]
        //     .iter()
        //     .max_by(|x, y| x.dtype.size.cmp(&y.dtype.size))
        //     .unwrap()
        //     .dtype
        //     .clone();
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
        // let out_dtype = if optype != ops::Binary::Cmplt {
        //     srcs.last().unwrap().lb().dtype.clone()
        // } else {
        //     _bool
        // };
        create_lazybuffer(
            ShapeTracker::new(&self.shape, None),
            LazyOp::new(optype, srcs, None),
            self.dtype.clone(),
            None,
        )
    }

    pub fn _reduce_op(&self, optype: OpType, new_shape: &[isize]) -> Self {
        if self.shape == new_shape {
            return self.clone();
        }
        let unbound_new_shape = new_shape;
        create_lazybuffer(
            ShapeTracker::new(new_shape, None),
            LazyOp::new(
                optype,
                vec![self.clone().into()],
                Some(vec![Arg::Shape(unbound_new_shape.to_vec())]),
            ),
            self.dtype.clone(),
            None,
        )
    }

    pub fn r<O: Into<OpType>>(&self, optype: O, new_shape: &[isize]) -> Self {
        let optype = optype.into();
        self._reduce_op(optype, new_shape)
    }

    pub fn cast(&self, dtype: Dtype, bitcast: Option<bool>) -> Self {
        if dtype.size == self.dtype.size {
            return self.clone();
        }
        if dtype.size <= self.dtype.size && self != self.base_ref() {
            return self
                .base_ref()
                .cast(dtype, bitcast)
                ._view(Movement::Reshape, self.st.clone());
        }
        let bitcast = bitcast.unwrap_or(false);
        create_lazybuffer(
            ShapeTracker::from_shape(&self.shape),
            LazyOp::new(
                ops::Unary::Cast.into(),
                vec![self.clone().into()],
                Some(vec![Arg::Dtype(dtype.clone())]),
            ),
            dtype,
            None,
        )
    }

    pub fn reshape(&self, arg: &[isize]) -> Self {
        self._view(Movement::Reshape, self.st.reshape(arg))
    }

    pub fn pad(&self, arg: &[isize]) -> Self {
        if arg.iter().all(|v| *v == 0) {
            return self.clone();
        }
        let mut aarg = vec![];
        for a in arg.windows(2).step_by(2) {
            aarg.push((a[0], a[1]))
        }
        self._view(Movement::Pad, self.st.pad(&aarg))
    }

    pub fn expand(&self, arg: &[isize]) -> Self {
        if &self.shape == arg {
            return self.clone();
        }
        // if !self.is_realized() && self.lazyop.optype == Movement::Expand {
        //     return self.lazyop.src[0].lb().expand(arg);
        // }

        self._view(Movement::Expand, self.st.expand(&arg))
    }

    pub fn permute(&self, arg: &[isize]) -> Self {
        if arg == &(0..arg.len()).map(|v| v as isize).collect::<Vec<isize>>() {
            return self.clone();
        }
        self._view(Movement::Permute, self.st.permute(arg))
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
        let st = self.st.shrink(
            &arg.windows(2)
                .step_by(2)
                .map(|a| (a[0], a[1]))
                .collect::<Vec<(isize, isize)>>(),
        );
        self._view(Movement::Shrink, st)
    }

    pub fn stride(&self, arg: &[isize]) -> Self {
        if arg.iter().all(|i| *i == 1) {
            return self.clone();
        }
        self._view(Movement::Stride, self.st.stride(arg))
    }
}

pub fn create_lazybuffer(
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
        let ret = LazyBuffer::new(st, optype, op, dtype, base);
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
    let ret = LazyBuffer::new(st, optype, op, dtype, base);
    if DEBUG.0.contains("LB") {
        println!("{} {:?}", ret.id, ret);
    }
    ret
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
        DEVICE.copyin(on_cpu, b.as_ref());
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

use rand::SeedableRng;
// Have to do this because the lack of num trait in Rust.
// num_traits's traits are not object safe.
#[rustfmt::skip]
fn gen_rand_num_bytes(size: usize, dtype: &Dtype) -> Vec<u8> {
    // let chains, it is not irrefutable_let_patterns
    #[allow(irrefutable_let_patterns)]
    let mut rng = if let seed = getenv::<isize>("SEED", -1) && seed >= 0 {
        rand::rngs::StdRng::seed_from_u64(seed as u64)
    } else {
        rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap()
    };
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
    let debug_cache = DEBUG.0.contains("CACHE");
    let debug_kernel = DEBUG.0.contains("KERNEL");
    let debug_sch = DEBUG.0.contains("SCH");
    let mut k_lock = KERNEL_CACHED.lock().unwrap();
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
        let cached = k_lock.get(&format!("{:?}", si.ast));
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
            k_lock.insert(
                format!("{:?}", si.ast),
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
    pub shape: Vec<isize>,
    pub dtype: Dtype,
    pub flops: f64,
    pub mem: HashMap<usize, usize>,
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
            mem: HashMap::from([(arg.idx(), arg.dtype().size * arg.st().real_size())]),
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
        let mut mem = self.mem.clone();
        mem.insert(arg.idx(), arg.dtype().size * arg.st().real_size());
        Self {
            shape: arg.st().shape_vec(),
            dtype: arg.dtype(),
            flops: self.flops,
            mem,
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
            dtype: self.dtype,
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
            dtype: y.dtype,
            mem: self.mem,
        }
    }
}

pub fn _recursive_lb<'a>(
    buf: &'a LazyBuffer,
    realizes: &mut HashSet<&'a LazyBuffer>,
    allbufs: &mut HashMap<&'a LazyBuffer, Option<&LazyBuffer>>,
    simple_pads: &mut HashSet<&'a LazyBuffer>,
    children: &mut HashMap<LazyBuffer, HashMap<LazyBuffer, Option<LazyBuffer>>>,
    scheduled: Option<bool>,
) {
    let scheduled = scheduled.unwrap_or(false);
    //println!("\ncalled realizes {}\n{:?}\n{:?}\n", realizes.len(), buf, buf.base_ref());
    if allbufs.contains_key(buf) || buf.base_ref().is_realized() {
        return;
    }
    let buf_base = buf.base_ref();
    if buf_base != buf {
        if prod(&buf_base.st.shape().dims) < prod(&buf.st.shape().dims) {
            //println!("buf base st prod < buf st prod");
            if buf.st.views.len() == 1
                && buf.st.views[buf.st.views.len() - 1].mask.is_some()
                && prod(&buf_base.st.shape().dims)
                    >= prod(
                        &v![y-x, for (x, y) in buf.st.views[buf.st.views.len()-1].mask.as_ref().unwrap()],
                    )
            {
                simple_pads.insert(buf_base);
            } else {
                //println!("base st < buf st real inresrting");
                realizes.insert(buf_base);
            }
        }
        return _recursive_lb(buf_base, realizes, allbufs, simple_pads, children, None);
    }
    if buf.force_realize {
        realizes.insert(buf);
    }
    allbufs.insert(buf, None);
    if matches!(buf.lazyop.optype, OpType::Load(_)) {
        realizes.insert(buf.base_ref());
    }
    if matches!(buf.lazyop.optype, OpType::Load(Load::From)) {
        realizes.insert(buf.lazyop.src[0].lb().base_ref());
    }

    for x in buf.lazyop.src.iter() {
        *children
            .entry(x.lb().base())
            .or_default()
            .entry(buf.clone())
            .or_default() = None;
        _recursive_lb(x.lb(), realizes, allbufs, simple_pads, children, None);
    }
}

pub fn _is_padding_okay(buf: &LazyBuffer, realizes: &HashSet<&LazyBuffer>) -> bool {
    use ops::*;
    if buf.is_realized() || realizes.contains(buf) {
        return true;
    }
    if matches!(
        buf.lazyop.optype,
        OpType::Binary(Binary::Div)
            | OpType::Binary(Binary::Cmplt)
            | OpType::Unary(Unary::Log2)
            | OpType::Unary(Unary::Exp2)
    ) {
        return false;
    }
    any(&v![_is_padding_okay(x.lb().base_ref(), realizes), for x in buf.lazyop.src.iter()])
}
pub fn _recursive_lazyop<'a>(
    mut buf: &'a LazyBuffer,
    inputs: &mut Vec<&'a LazyBuffer>,
    mut st: ShapeTracker,
    realizes: &mut HashSet<&LazyBuffer>,
    cache: &mut HashMap<(&'a LazyBuffer, ShapeTracker), LazyOp>,
    first: Option<bool>,
) -> LazyOp {
    //println!("{:?}", buf);
    let first = first.unwrap_or(true);
    if cache.contains_key(&(buf, st.clone())) {
        return cache[&(buf, st.clone())].clone();
    }
    if buf != buf.base_ref() {
        st = buf.st.concat(&st);
        buf = buf.base_ref();
    }
    if buf.lazyop.optype == Load::Const {
        return LazyOp::new(
            OpType::Buffer(ops::Buffer::Const),
            vec![],
            Some(vec![Arg::Buffer(
                ConstBuffer {
                    val: buf.lazyop.args[0].to_str(),
                    dtype: buf.dtype.clone(),
                    st,
                }
                .into(),
            )]),
        );
    }

    if buf.is_realized() || (!first && realizes.contains(buf)) {
        if !inputs.contains(&buf) {
            inputs.push(buf)
        }
        return LazyOp::new(
            OpType::Buffer(ops::Buffer::Load),
            vec![],
            Some(vec![Arg::Buffer(
                MemBuffer {
                    idx: inputs.iter().position(|x| x.id == buf.id).unwrap() + 1,
                    dtype: buf.dtype.clone(),
                    st,
                }
                .into(),
            )]),
        );
    }

    if buf.lazyop.optype == Load::Contiguous {
        assert!(first);
        return _recursive_lazyop(
            &buf.lazyop.src[0].lb(),
            inputs,
            st,
            realizes,
            cache,
            Some(false),
        );
    }

    if matches!(buf.lazyop.optype, OpType::Reduce(_)) {
        assert!(st.contiguous());
        st = ShapeTracker::from_shape(&buf.lazyop.src[0].lb().shape);
    }
    let mut ret = LazyOp::new(
        buf.lazyop.optype.clone(),
        v![_recursive_lazyop(x.lb(), inputs, st.clone(), realizes, cache, Some(false)).into(), for x in buf.lazyop.src.iter()],
        Some(buf.lazyop.args.clone()),
    );
    cache.insert((buf, st.clone()), ret.clone());
    ret
}

pub fn _recursive_schedule<'a>(
    out: &'a LazyBuffer,
    seen: &mut HashSet<&'a LazyBuffer>,
    realizes: &mut HashSet<&LazyBuffer>,
    reduce_for_op: &mut HashMap<&LazyBuffer, &LazyBuffer>,
) -> Vec<ScheduleItem> {
    if out.is_realized() || out.lazyop.optype == Load::Const || seen.contains(out) {
        return vec![];
    }
    assert!(out.base_ref() == out);
    seen.insert(out);
    let mut inputs: Vec<&LazyBuffer> = vec![];
    // if out.op in {LoadOps.CUSTOM, LoadOps.SYNC, LoadOps.WAIT, LoadOps.COPY, LoadOps.EMPTY}:
    let op = if matches!(
        out.lazyop.optype,
        OpType::Load(Load::Rand) | OpType::Load(Load::From)
    ) {
        inputs = v![x.lb(),for x in out.lazyop.src.iter()];
        LazyOp::new(
            out.lazyop.optype.clone(),
            vec![],
            Some(out.lazyop.args.clone()),
        )
    } else {
        let output_st = ShapeTracker::from_shape(if reduce_for_op.contains_key(out) {
            &reduce_for_op[out].shape
        } else {
            &out.shape
        });
        let mut cache = HashMap::new();
        //println!("\n1111\nout {out:?}\ninputs {inputs:?}\noutput st {output_st:?}\nrealizes {realizes:?}\n");
        let op = _recursive_lazyop(
            out,
            &mut inputs,
            output_st.clone(),
            realizes,
            &mut cache,
            None,
        );
        //println!("inputs after recur {}", inputs.len());
        //println!("{:?}", op.src[0].src());
        LazyOp::new(
            OpType::Buffer(ops::Buffer::Store),
            vec![op.into()],
            Some(vec![Arg::Buffer(
                MemBuffer {
                    idx: 0,
                    dtype: out.dtype.clone(),
                    st: output_st.simplify(),
                }
                .into(),
            )]),
        )
    };
    let mut ret =
        v![_recursive_schedule(x.base_ref(), seen, realizes, reduce_for_op), for x in inputs.iter()].concat();
    let owned_input = inputs
        .iter()
        .map(|&i| i.clone())
        .collect::<Vec<LazyBuffer>>();
    ret.push(ScheduleItem {
        ast: op,
        out: out.clone(),
        inputs: owned_input,
    });
    ret
}

pub fn create_schedule<'a>(
    outs: Vec<&'a LazyBuffer>,
    seen: Option<&mut HashSet<&'a LazyBuffer>>,
) -> Vec<ScheduleItem> {
    let mut set = HashSet::new();
    let seen = seen.unwrap_or(&mut set);
    let _r = v![x.base_ref(), for x in outs.iter(), if !x.base_ref().is_realized()];
    let mut realizes: HashSet<&LazyBuffer> = HashSet::from_iter(_r.iter());
    let mut allbufs = HashMap::new();
    let mut simple_pads = HashSet::new();
    let mut children = HashMap::new();
    //println!("+++ {} {} {} {}", realizes.len(), allbufs.len(), simple_pads.len(), children.len());
    for out in outs.iter() {
        //println!("out base {:?} {:?} {}", out.base_ref(), out.base_ref().is_realized(), out.lazyop.src.len());
        _recursive_lb(
            out.base_ref(),
            &mut realizes,
            &mut allbufs,
            &mut simple_pads,
            &mut children,
            Some(true),
        );
    }
    //println!("+++ {} {} {} {}", realizes.len(), allbufs.len(), simple_pads.len(), children.len());

    for p in simple_pads {
        if !_is_padding_okay(p, &realizes) {
            realizes.insert(p);
        }
    }
    let mut reduce_for_op: HashMap<&LazyBuffer, &LazyBuffer> = HashMap::new();
    for &r in allbufs.keys() {
        if r != r.base_ref()
            || !matches!(r.lazyop.optype, OpType::Reduce(_))
            || realizes.contains(r)
        {
            continue;
        }
        let mut child_set: HashMap<&LazyBuffer, ShapeTracker> = HashMap::from([(r, r.st.clone())]);
        let mut realized_child: HashMap<&LazyBuffer, ShapeTracker> = HashMap::new();
        let mut force_realize = false;
        let mut can_chase = true;
        while !force_realize && child_set.len() > 0 {
            let mut next_child_set: HashMap<&LazyBuffer, ShapeTracker> = HashMap::new();
            for (tr, st) in child_set.iter() {
                if realizes.contains(tr) {
                    realized_child.insert(tr, st.clone());
                    if realized_child.len() > 1
                        || !st.contiguous()
                        || st.size() != r.st.size()
                        || (reduce_for_op.get(tr).is_some_and(|x| x != tr))
                    {
                        can_chase = !reduce_for_op.contains_key(tr) || reduce_for_op[tr] == r;
                        force_realize = true;
                        break;
                    }
                    continue;
                }
                for tr_next in children[tr].keys() {
                    if !tr_next.is_realized() {
                        if matches!(tr_next.lazyop.optype, OpType::Reduce(_)) {
                            force_realize = true;
                            break;
                        }

                        let st_childs = dedup(
                            v![s, for s in tr_next.lazyop.src.iter(), if s.lb().base_ref() == *tr],
                        );
                        if st_childs.len() > 1 {
                            force_realize = true;
                            break;
                        }
                        next_child_set.insert(tr_next, st.concat(&st_childs[0].lb().st));
                    }
                }
            }
            child_set = next_child_set;
        }
        if force_realize {
            let mut tr = r;
            if can_chase {
                let mut st = tr.st.clone();
                while children.contains_key(tr) && children[tr].len() == 1 {
                    let tr_next = children[tr].keys().next().unwrap();
                    let st_childs = dedup(
                        v![s.lb(), for s in tr_next.lazyop.src.iter(), if s.lb().base_ref() == tr ],
                    );
                    if st_childs.len() > 1 || st.size() != st_childs[0].st.size() {
                        break;
                    }
                    st = st.concat(&st_childs[0].st);
                    if !st.contiguous() || matches!(tr_next.lazyop.optype, OpType::Reduce(_)) {
                        break;
                    };
                    tr = tr_next;
                }
                reduce_for_op.insert(tr, r);
            }
            realizes.insert(tr);
        } else {
            assert!(realized_child.len() == 1);
            reduce_for_op.insert(realized_child.keys().next().unwrap(), r);
        }
    }
    v![_recursive_schedule(x.base_ref(), seen, &mut realizes, &mut reduce_for_op), for x in outs]
        .concat()
}

pub fn get_lazyop_info(ast: &LazyOpSrc) -> FlopCounter {
    let srcs = vec![vec![ast.clone()], ast.src()].concat();
    for o in srcs {
        match o.optype() {
            OpType::Unary(u) => {
                return match u {
                    ops::Unary::Cast => {
                        get_lazyop_info(&o.lo().src[0]).unary_cast(o.lo().args[0].to_dtype())
                    }
                    _ => get_lazyop_info(&o.lo().src[0]).unary_cast(float32),
                }
            }
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
