use core::ops::Index;
use num_traits::{PrimInt, ToPrimitive};
use std::ops::RangeBounds;

#[derive(Clone, Eq, PartialEq)]
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

impl<T: ToPrimitive + core::fmt::Debug> Index<T> for Shape {
    type Output = isize;

    fn index(&self, index: T) -> &Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.dims[index]
    }
}

impl<T: ToPrimitive + core::fmt::Debug> core::ops::IndexMut<T> for Shape {
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        let index = index.to_isize().unwrap();
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
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
