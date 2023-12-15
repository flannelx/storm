use crate::{
    dtype,
    lazy::LazyBuffer,
    ops::{LazyOp, OpType},
};

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
    reduceop: Option<LazyOp>
}

#[allow(unused_variables)]
impl Kenrel {
    pub fn new(ast: &LazyOp, output_buffer: &LazyBuffer, opts: LinearizerOptions) -> Self {
        // let ast = if ast.optype == Movement::Reshape {
        //     ast.src[0].clone().to_lo()
        // } else {
        //     ast.clone()
        // };

        let mut reduceops = ast
            .get_lazyops()
            .into_iter()
            .filter(|x| matches!(x.optype, OpType::Reduce(_)))
            .collect::<Vec<LazyOp>>();
        let reduceop = if reduceops.is_empty() {
            None
        } else {
            Some(reduceops.swap_remove(0))
        };

        let mut bufs = vec![output_buffer.clone()];
        let mut tmp = ast.buffers.clone();
        tmp.dedup();
        bufs.extend(tmp);
        todo!()
    }
}
