use core::ops::Index;
use num_traits::{PrimInt, ToPrimitive};
use std::ops::RangeBounds;

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims: Vec<isize>,
}

impl IntoIterator for Shape {
    type Item = isize;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl Shape {
    pub fn new<I: ToPrimitive, Dims: Into<Vec<I>>>(dims: Dims) -> Self {
        let dims = dims
            .into()
            .iter()
            .map(|i| i.to_isize().unwrap())
            .collect::<Vec<isize>>();
        Self { dims }
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product::<isize>() as usize
    }

    pub fn strides(&self) -> Shape {
        let mut dims = vec![1; self.dims.len()];
        let mut stride = 1;
        dims.iter_mut()
            .zip(self.dims.iter())
            .rev()
            .for_each(|(st, sh)| {
                *st = stride;
                stride *= *sh
            });
        Shape { dims }
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }
}

impl<const D: usize, I: PrimInt + ToPrimitive> From<[I; D]> for Shape {
    fn from(value: [I; D]) -> Self {
        Self {
            dims: value.iter().map(|e| e.to_isize().unwrap()).collect(),
        }
    }
}

impl<I: PrimInt + ToPrimitive> From<Vec<I>> for Shape {
    fn from(value: Vec<I>) -> Self {
        Self {
            dims: value.iter().map(|e| e.to_isize().unwrap()).collect(),
        }
    }
}

impl<I: PrimInt + ToPrimitive> From<&[I]> for Shape {
    fn from(value: &[I]) -> Self {
        Self {
            dims: value.iter().map(|e| e.to_isize().unwrap()).collect(),
        }
    }
}

impl Index<isize> for Shape {
    type Output = isize;
    fn index(&self, index: isize) -> &Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.dims[index]
    }
}

impl core::ops::IndexMut<isize> for Shape {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &mut self.dims[index]
    }
}

impl Index<i32> for Shape {
    type Output = isize;
    fn index(&self, index: i32) -> &Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.dims[index]
    }
}

impl core::ops::IndexMut<i32> for Shape {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &mut self.dims[index]
    }
}

impl Index<usize> for Shape {
    type Output = isize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl core::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl Index<std::ops::Range<usize>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl core::ops::IndexMut<std::ops::Range<usize>> for Shape {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl Index<std::ops::Range<isize>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::Range<isize>) -> &Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.dims[start..end]
    }
}

impl core::ops::IndexMut<std::ops::Range<isize>> for Shape {
    fn index_mut(&mut self, index: std::ops::Range<isize>) -> &mut Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.dims[start..end]
    }
}

impl Index<std::ops::RangeTo<isize>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeTo<isize>) -> &Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.dims[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeTo<isize>> for Shape {
    fn index_mut(&mut self, index: std::ops::RangeTo<isize>) -> &mut Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.dims[start..end]
    }
}

impl Index<std::ops::RangeFrom<isize>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFrom<isize>) -> &Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.dims[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeFrom<isize>> for Shape {
    fn index_mut(&mut self, index: std::ops::RangeFrom<isize>) -> &mut Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.dims[start..end]
    }
}


impl Index<std::ops::Range<i32>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::Range<i32>) -> &Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.dims[start..end]
    }
}

impl core::ops::IndexMut<std::ops::Range<i32>> for Shape {
    fn index_mut(&mut self, index: std::ops::Range<i32>) -> &mut Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.dims[start..end]
    }
}

impl Index<std::ops::RangeTo<i32>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeTo<i32>) -> &Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.dims[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeTo<i32>> for Shape {
    fn index_mut(&mut self, index: std::ops::RangeTo<i32>) -> &mut Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.dims[start..end]
    }
}

impl Index<std::ops::RangeFrom<i32>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFrom<i32>) -> &Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &self.dims[start..end]
    }
}

impl core::ops::IndexMut<std::ops::RangeFrom<i32>> for Shape {
    fn index_mut(&mut self, index: std::ops::RangeFrom<i32>) -> &mut Self::Output {
        let len = self.dims.len() as isize;
        let start = match index.start_bound() {
            std::ops::Bound::Included(&i) => panic!(),
            std::ops::Bound::Excluded(&i) => i as isize + len,
            std::ops::Bound::Unbounded => 0,
        } as usize;
        let end = match index.end_bound() {
            std::ops::Bound::Included(&i) => i as isize + len,
            std::ops::Bound::Excluded(&i) => i as isize + len + 1,
            std::ops::Bound::Unbounded => len,
        } as usize;
        &mut self.dims[start..end]
    }
}


impl Index<std::ops::RangeTo<usize>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl core::ops::IndexMut<std::ops::RangeTo<usize>> for Shape {
    fn index_mut(&mut self, index: std::ops::RangeTo<usize>) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl Index<std::ops::RangeFrom<usize>> for Shape {
    type Output = [isize];
    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl core::ops::IndexMut<std::ops::RangeFrom<usize>> for Shape {
    fn index_mut(&mut self, index: std::ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

impl core::fmt::Debug for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}
