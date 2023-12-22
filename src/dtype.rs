#![allow(non_upper_case_globals)]

use std::collections::{HashMap, HashSet};

use half::{bf16, f16};

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Dtype {
    pub priority: usize,
    pub size: usize,
    pub c_name: &'static str,
    pub sz: usize,
    pub shape: Option<Vec<isize>>,
    pub ptr: bool,
    pub type_name: &'static str,
}

impl PartialOrd for Dtype {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.size.cmp(&other.size))
    }
}

impl Ord for Dtype {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.size.cmp(&other.size)
    }
}

impl core::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(sh) = &self.shape {
            return write!(f, "dtypes.{}({:?})", self.c_name, sh);
        } else if self.ptr {
            return write!(f, "ptr.{}", self.c_name);
        }
        write!(f, "dtypes.{}", self.c_name)
    }
}

impl Dtype {
    pub fn is_int(&self) -> bool {
        matches!(
            *self,
            int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(*self, float16 | float32 | float64)
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(*self, uint8 | uint16 | uint32 | uint64)
    }
}

pub const _bool: Dtype = Dtype {
    priority: 0,
    size: std::mem::size_of::<bool>(),
    c_name: "bool",
    type_name: "bool",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const float16: Dtype = Dtype {
    priority: 0,
    size: std::mem::size_of::<half::f16>(),
    c_name: "half",
    type_name: "f16",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const half: Dtype = float16;

pub const float32: Dtype = Dtype {
    priority: 4,
    size: std::mem::size_of::<f32>(),
    c_name: "float",
    type_name: "f32",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const float64: Dtype = Dtype {
    priority: 0,
    size: std::mem::size_of::<f64>(),
    c_name: "double",
    type_name: "f64",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const double: Dtype = float64;

pub const int8: Dtype = Dtype {
    priority: 0,
    size: std::mem::size_of::<i8>(),
    c_name: "char",
    type_name: "i8",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const int16: Dtype = Dtype {
    priority: 1,
    size: std::mem::size_of::<i16>(),
    c_name: "short",
    type_name: "i16",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const int32: Dtype = Dtype {
    priority: 2,
    size: std::mem::size_of::<i32>(),
    c_name: "int",
    type_name: "i32",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const int64: Dtype = Dtype {
    priority: 3,
    size: std::mem::size_of::<i64>(),
    c_name: "long",
    type_name: "i64",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint8: Dtype = Dtype {
    priority: 0,
    size: std::mem::size_of::<u8>(),
    c_name: "unsigned char",
    type_name: "u8",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint16: Dtype = Dtype {
    priority: 1,
    size: std::mem::size_of::<u16>(),
    c_name: "unsigned short",
    type_name: "u16",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint32: Dtype = Dtype {
    priority: 2,
    size: std::mem::size_of::<u32>(),
    c_name: "unsigned int",
    type_name: "u32",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint64: Dtype = Dtype {
    priority: 3,
    size: std::mem::size_of::<u64>(),
    c_name: "unsigned long",
    type_name: "u64",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const bfloat16: Dtype = Dtype {
    priority: 0,
    size: std::mem::size_of::<half::bf16>(),
    c_name: "__bf16",
    type_name: "bf16",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const _int2: Dtype = Dtype {
    priority: 2,
    size: 8,
    c_name: "int2",
    type_name: "int2",
    sz: 2,
    shape: None,
    ptr: false,
};

pub const _half4: Dtype = Dtype {
    priority: 0,
    size: 8,
    c_name: "half4",
    type_name: "half4",
    sz: 4,
    shape: None,
    ptr: false,
};

pub const _float2: Dtype = Dtype {
    priority: 4,
    size: 8,
    c_name: "float2",
    type_name: "float2",
    sz: 2,
    shape: None,
    ptr: false,
};

pub const _float4: Dtype = Dtype {
    priority: 4,
    size: 16,
    c_name: "float4",
    type_name: "float4",
    sz: 4,
    shape: None,
    ptr: false,
};

pub const _arg_int32: Dtype = Dtype {
    priority: 2,
    size: 4,
    c_name: "_arg_int32",
    type_name: "_arg_int32",
    sz: 1,
    shape: None,
    ptr: false,
};

// pub trait Num: num_traits::ToPrimitive {
//     fn is_float() -> bool;
//     fn is_uint() -> bool;
//     fn is_int() -> bool;
// }
//
// macro_rules! num_impl {
//     ($t: tt, $float: tt, $uint: tt, $int: tt) => {
//         impl Num for $t {
//             fn is_float() -> bool {
//                 $float
//             }
//             fn is_uint() -> bool {
//                 $uint
//             }
//             fn is_int() -> bool {
//                 $int
//             }
//         }
//     };
// }
//
// num_impl!(f32, true, false, false);
// num_impl!(f64, true, false, false);
// num_impl!(usize, false, true, false);
// num_impl!(u8, false, true, false);
// num_impl!(u16, false, true, false);
// num_impl!(u32, false, true, false);
// num_impl!(u64, false, true, false);
// num_impl!(isize, false, false, true);
// num_impl!(i8, false, false, true);
// num_impl!(i16, false, false, true);
// num_impl!(i32, false, false, true);
// num_impl!(i64, false, false, true);

pub fn name_to_dtype(name: &str) -> Dtype {
    match name {
        "f16" => float16,
        "f32" => float32,
        "f64" => float64,
        "u8" => uint8,
        "u16" => uint16,
        "u32" => uint32,
        "u64" => uint64,
        "usize" => match std::mem::size_of::<usize>() {
            4 => uint32,
            8 => uint64,
            _ => unreachable!(),
        },
        "i8" => int8,
        "i16" => int16,
        "i32" => int32,
        "i64" => int64,
        "isize" => match std::mem::size_of::<isize>() {
            4 => int32,
            8 => int64,
            _ => unreachable!(),
        },
        t => panic!("{t} is not Rust's primitives"),
    }
}

pub fn type_to_dtype<T>() -> Dtype {
    let name = std::any::type_name::<T>().split("::").last().unwrap();
    match name {
        "f16" => float16,
        "f32" => float32,
        "f64" => float64,
        "u8" => uint8,
        "u16" => uint16,
        "u32" => uint32,
        "u64" => uint64,
        "usize" => match std::mem::size_of::<usize>() {
            4 => uint32,
            8 => uint64,
            _ => unreachable!(),
        },
        "i8" => int8,
        "i16" => int16,
        "i32" => int32,
        "i64" => int64,
        "isize" => match std::mem::size_of::<isize>() {
            4 => int32,
            8 => int64,
            _ => unreachable!(),
        },
        t => panic!("{t} is not Rust's primitives"),
    }
}

pub trait NumType:
    'static
    + core::fmt::Debug
    + core::fmt::Display
    + Default
    + Copy
    + Send
    + Sync
    + num_traits::FromPrimitive
    + num_traits::ToPrimitive
    + num_traits::Num
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + core::ops::DivAssign
    + core::ops::Add<Self, Output = Self>
    + core::ops::Sub<Self, Output = Self>
    + core::ops::Mul<Self, Output = Self>
    + core::ops::Div<Self, Output = Self>
    + PartialEq
    + PartialOrd
{
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn _to_le_bytes(&self) -> Vec<u8>;
}

macro_rules! NumTypeImpl {
    ($t:tt) => {
        impl NumType for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                $t::from_le_bytes(bytes.try_into().expect(&format!(
                    "Unable to get '{}' from {bytes:?}",
                    std::any::type_name::<Self>()
                )))
            }
            fn _to_le_bytes(&self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
        }
    };
}

NumTypeImpl!(u8);
NumTypeImpl!(u16);
NumTypeImpl!(u32);
NumTypeImpl!(u64);
NumTypeImpl!(usize);
NumTypeImpl!(i8);
NumTypeImpl!(i16);
NumTypeImpl!(i32);
NumTypeImpl!(i64);
NumTypeImpl!(isize);
NumTypeImpl!(f32);
NumTypeImpl!(f64);
NumTypeImpl!(f16);
NumTypeImpl!(bf16);

#[rustfmt::skip]
lazy_static::lazy_static! {
    pub static ref type_rules: HashMap<Dtype, HashSet<Dtype>> = HashMap::from([
    (_bool,    HashSet::from([_bool,     uint8,     uint16,    uint32,    uint64,    int8,      int16,     int32,     int64,     float16,   float32,   bfloat16])),
    (uint8,    HashSet::from([uint8,     uint8,     uint16,    uint32,    uint64,    int16,     int16,     int32,     int64,     float16,   float32,   bfloat16])),
    (uint16,   HashSet::from([uint16,    uint16,    uint16,    uint32,    uint64,    int32,     int32,     int32,     int64,     float16,   float32,   bfloat16])),
    (uint32,   HashSet::from([uint32,    uint32,    uint32,    uint32,    uint64,    int64,     int64,     int64,     int64,     float16,   float32,   bfloat16])),
    (uint64,   HashSet::from([uint64,    uint64,    uint64,    uint64,    uint64,    float32,   float32,   float32,   float32,   float16,   float32,   bfloat16])),
    (int8,     HashSet::from([int8,      int16,     int32,     int64,     float32,   int8,      int16,     int32,     int64,     float16,   float32,   bfloat16])),
    (int16,    HashSet::from([int16,     int16,     int32,     int64,     float32,   int16,     int16,     int32,     int64,     float16,   float32,   bfloat16])),
    (int32,    HashSet::from([int32,     int32,     int32,     int64,     float32,   int32,     int32,     int32,     int64,     float16,   float32,   bfloat16])),
    (int64,    HashSet::from([int64,     int64,     int64,     int64,     float32,   int64,     int64,     int64,     int64,     float16,   float32,   bfloat16])),
    (float16,  HashSet::from([float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float32,   float32])),
    (float32,  HashSet::from([float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32])),
    (bfloat16, HashSet::from([bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  float32,   float32,   bfloat16])),
    ]);
}
pub fn least_upper_dtype(dtypes: &[Dtype]) -> Dtype {
    let mut rets = HashSet::new();
    for d in dtypes {
        if rets.is_empty() {
            rets = type_rules[d].clone();
        }
        rets = rets.intersection(&type_rules[d]).cloned().collect();
    }
    rets.iter().min().unwrap().to_owned()
}
