use crate::arg::Arg;
use crate::{dtype, ops::OpType, lazy::LazyBuffer};

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
