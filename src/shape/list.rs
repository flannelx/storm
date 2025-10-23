use core::ops::Index;
use num_traits::{PrimInt, ToPrimitive};
use std::ops::RangeBounds;

use crate::shape::{Shape, view::View};

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct ShapeVec(pub Vec<isize>);

impl IntoIterator for ShapeVec {
    type Item = isize;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl ShapeVec {
    pub fn to_shape(&self) -> Shape {
        Shape {
            views: vec![View {
                shape: vec![self.clone()],
                stride: vec![self.stride()],
                mask: None,
            }],
        }
    }

    pub fn new<I: ToPrimitive, Dims: Into<Vec<I>>>(dims: Dims) -> Self {
        let dims = dims
            .into()
            .iter()
            .map(|i| i.to_isize().unwrap())
            .collect::<Vec<isize>>();
        Self(dims)
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product::<isize>() as usize
    }

    pub fn stride(&self) -> ShapeVec {
        let mut dims = vec![1; self.0.len()];
        let mut stride = 1;
        dims.iter_mut()
            .zip(self.0.iter())
            .rev()
            .for_each(|(st, sh)| {
                *st = stride;
                stride *= *sh
            });
        ShapeVec(dims)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

// macro_rules! from_vec {
//     ($ty: ty) => {
//         impl<const D:usize> From<[$ty; D]> for ShapeVec {
//             fn from(value: [$ty; D]) -> Self {
//                 Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
//             }
//
//         }
//         impl From<&[$ty]> for ShapeVec {
//             fn from(value: &[$ty]) -> Self {
//                 Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
//             }
//         }
//         impl From<Vec<$ty>> for ShapeVec {
//             fn from(value: Vec<$ty>) -> Self {
//                 Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
//             }
//         }
//     };
// }
//
// from_vec!(i32);
// from_vec!(i64);
// from_vec!(isize);
// from_vec!(u32);
// from_vec!(u64);
// from_vec!(usize);

impl<const D: usize, I: PrimInt + ToPrimitive> From<[I; D]> for ShapeVec {
    fn from(value: [I; D]) -> Self {
        Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
    }
}

impl<const D: usize, I: PrimInt + ToPrimitive> From<&[I; D]> for ShapeVec {
    fn from(value: &[I; D]) -> Self {
        Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
    }
}

impl<I: PrimInt + ToPrimitive> From<Vec<I>> for ShapeVec {
    fn from(value: Vec<I>) -> Self {
        Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
    }
}

impl<I: PrimInt + ToPrimitive> From<&Vec<I>> for ShapeVec {
    fn from(value: &Vec<I>) -> Self {
        Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
    }
}

impl<I: PrimInt + ToPrimitive> From<&[I]> for ShapeVec {
    fn from(value: &[I]) -> Self {
        Self(value.iter().map(|e| e.to_isize().unwrap()).collect())
    }
}

impl Index<isize> for ShapeVec {
    type Output = isize;
    fn index(&self, index: isize) -> &Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.0[index]
    }
}

impl core::ops::IndexMut<isize> for ShapeVec {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &mut self.0[index]
    }
}

impl Index<i32> for ShapeVec {
    type Output = isize;
    fn index(&self, index: i32) -> &Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.0[index]
    }
}

impl core::ops::IndexMut<i32> for ShapeVec {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &mut self.0[index]
    }
}

impl Index<usize> for ShapeVec {
    type Output = isize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<usize> for ShapeVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Index<std::ops::Range<usize>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<std::ops::Range<usize>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Index<std::ops::Range<isize>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::Range<isize>) -> &Self::Output {
        let len = self.0.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.0[start..end]
    }
}

impl core::ops::IndexMut<std::ops::Range<isize>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::Range<isize>) -> &mut Self::Output {
        let len = self.0.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.0[start..end]
    }
}

impl Index<std::ops::RangeTo<isize>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeTo<isize>) -> &Self::Output {
        let len = self.0.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.0[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeTo<isize>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeTo<isize>) -> &mut Self::Output {
        let len = self.0.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.0[start..end]
    }
}

impl Index<std::ops::RangeFrom<isize>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFrom<isize>) -> &Self::Output {
        let len = self.0.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i as isize + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i as isize + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i as isize + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.0[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeFrom<isize>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeFrom<isize>) -> &mut Self::Output {
        let len = self.0.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i as isize + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i as isize + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i as isize + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.0[start..end]
    }
}

impl Index<std::ops::Range<i32>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::Range<i32>) -> &Self::Output {
        let len = self.0.len() as i32;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.0[start..end]
    }
}

impl core::ops::IndexMut<std::ops::Range<i32>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::Range<i32>) -> &mut Self::Output {
        let len = self.0.len() as i32;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.0[start..end]
    }
}

impl Index<std::ops::RangeTo<i32>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeTo<i32>) -> &Self::Output {
        let len = self.0.len() as i32;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.0[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeTo<i32>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeTo<i32>) -> &mut Self::Output {
        let len = self.0.len() as i32;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.0[start..end]
    }
}

impl Index<std::ops::RangeFrom<i32>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFrom<i32>) -> &Self::Output {
        let len = self.0.len() as i32;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.0[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeFrom<i32>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeFrom<i32>) -> &mut Self::Output {
        let len = self.0.len() as i32;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => {
                if i < 0 {
                    i + len + 1
                } else {
                    i + 1
                }
            }
            std::ops::Bound::Excluded(&i) => {
                if i < 0 {
                    i + len
                } else {
                    i
                }
            }
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.0[start..end]
    }
}

impl Index<std::ops::RangeTo<usize>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<std::ops::RangeTo<usize>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeTo<usize>) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Index<std::ops::RangeFull> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFull) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<std::ops::RangeFull> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeFull) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Index<std::ops::RangeFrom<usize>> for ShapeVec {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<std::ops::RangeFrom<usize>> for ShapeVec {
    fn index_mut(&mut self, index: std::ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl core::fmt::Display for ShapeVec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl core::fmt::Debug for ShapeVec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
