pub mod util;
pub mod view;

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::prelude::*;
use crate::view;

pub use util::*;
use view::View;

use crate::shape::symbolic::{num, var};

use super::symbolic::ArcNode;

#[derive(Clone, Debug, Default)]
pub struct ArcViews(Arc<Vec<View>>); // Switch to this later avoid copys of views. however,
                                     // deep clone is needed for some op in shapetracker

impl From<Vec<View>> for ArcViews {
    fn from(value: Vec<View>) -> Self {
        Self(Arc::new(value))
    }
}

impl From<&[View]> for ArcViews {
    fn from(value: &[View]) -> Self {
        Self(Arc::new(value.to_vec()))
    }
}

impl Deref for ArcViews {
    type Target = Vec<View>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ArcViews {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Arc::get_mut_unchecked(&mut self.0) }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShapeTracker {
    pub views: Vec<View>,
}

impl ShapeTracker {
    pub fn new(shape: &[isize], views: Option<Vec<View>>) -> Self {
        let views = if let Some(v) = views {
            v
        } else {
            vec![View::new(shape, None, None, None)]
        };

        Self { views }
    }

    pub fn invert(&self, out_shape: &[isize]) -> Option<Self> {
        let shapes = v![s, for (v, s) in izip!(self.views.iter().rev(), vec![v![x.shape.clone(), for x in self.views.iter().rev().skip(1)], vec![out_shape.to_vec()]].concat())];
        let ret =
            v![v.invert(&s), for (v, s) in izip!(self.views.iter().rev(), shapes.into_iter())];
        if all(&v![x.is_some(), for x in ret.iter()]) {
            Some(
                Self {
                    views: v![x.unwrap(), for x in ret.into_iter()],
                }
                .reshape(out_shape),
            )
        } else {
            None
        }
    }

    pub fn concat(&self, st: &Self) -> Self {
        let mut ret = self.clone();
        for v in st.views.iter() {
            ret = ShapeTracker {
                views: vec![ret.views.clone(), vec![v.clone()]].concat(),
            }
            .simplify();
        }
        if ret.shape().dims.contains(&0) {
            panic!();
        }
        ret
    }

    pub fn from_shape(shape: &[isize]) -> Self {
        let views = vec![View::new(shape, None, None, None)];
        Self { views }
    }

    pub fn contiguous(&self) -> bool {
        self.views.len() == 1 && self.views[0].contiguous
    }

    pub fn shape_vec(&self) -> Vec<isize> {
        self.views.last().unwrap().shape.clone()
    }

    pub fn shape(&self) -> crate::tensor::shape::Shape {
        self.views.last().unwrap().shape.clone().into()
    }

    pub fn strides(&self) -> Vec<isize> {
        self.views.last().unwrap().strides.clone()
    }

    pub fn key(&self) -> Vec<View> {
        self.views.clone()
    }

    pub fn size(&self) -> isize {
        let v = self.views.last().unwrap();
        v.shape
            .iter()
            .zip(v.strides.iter())
            .filter(|(_, &st)| st != 0)
            .map(|(sh, _)| *sh)
            .product()
    }

    // pub fn real_offset(&self) -> isize {
    //     let (real_offset, _) = self.expr_node(Some(var("zero", 0, 0)));
    //     assert!(real_offset.is_num());
    //     real_offset.num_val().unwrap()
    // }

    pub fn real_strides(&self, ignore_valid: bool) -> Vec<Option<isize>> {
        let last_view = self.views.last().unwrap();
        if self.views.len() == 1 && last_view.mask.is_none() {
            return last_view.strides.iter().map(|st| Some(*st)).collect();
        };
        let mut ret = vec![None; last_view.shape.len()];
        let idxs: Vec<ArcNode> = self
            .shape_vec()
            .iter()
            .enumerate()
            .map(|(i, sh)| var(&format!("idx{}", i), 0, sh - 1))
            .collect();
        let (idx, valid) = self.expr_idxs(Some(idxs.clone()));
        for this_dim in if idx.is_sum() {
            idx.nodes()
        } else {
            vec![idx.clone()]
        } {
            // println!("\n----\nidxs: {:?}\n\nidx: {}", idxs.iter().map(|n| n.key()).collect::<Vec<String>>(), this_dim.a().unwrap());
            if this_dim.is_mul()
                && this_dim.a().unwrap().is_var()
                && idxs.contains(&this_dim.a().unwrap())
            {
                ret[idxs
                    .iter()
                    .position(|n| n == &this_dim.a().unwrap())
                    .unwrap()] = Some(this_dim.b().unwrap().num_val().unwrap());
            } else if this_dim.is_var() {
                ret[idxs.iter().position(|n| n == &this_dim).unwrap()] = Some(1);
            }
        }
        let (idx_vars, valid_vars) = (idx.vars(), valid.vars());
        for (i, tidx) in idxs.iter().enumerate() {
            if valid_vars.contains(tidx) && !ignore_valid {
                ret[i] = None;
            } else if !idx_vars.contains(tidx) {
                ret[i] = Some(0);
            }
        }
        ret
    }

    pub fn simplify(&self) -> Self {
        if self.views.len() >= 2  && let Some(new_view) = merge_views(&self.views[self.views.len() - 2], &self.views[self.views.len() - 1]) {
            return Self {
                views: vec![self.views[..self.views.len()-2].to_vec(), vec![new_view]].concat(),
            }.simplify();
        }
        self.clone()
    }

    pub fn _expr_idx(&self, mut idx: ArcNode, mut valid: ArcNode) -> (ArcNode, ArcNode) {
        for v in self.views[0..self.views.len() - 1].iter().rev() {
            if valid.max().unwrap() == 0 {
                return (num(-1), valid);
            }
            valid = v.expr_node_mask(idx.clone(), Some(valid));
            idx = v.expr_node(Some(idx));
        }
        return (idx, valid);
    }

    pub fn expr_idxs(&self, idxs: Option<Vec<ArcNode>>) -> (ArcNode, ArcNode) {
        let idxs = if let Some(i) = idxs {
            i
        } else {
            self.shape_vec()
                .iter()
                .enumerate()
                .map(|(i, sh)| var(&format!("idx{}", i), 0, sh - 1))
                .collect()
        };
        let idx = self.views[self.views.len() - 1].expr_idxs(&idxs);
        let valid = self.views[self.views.len() - 1].expr_node_mask(
            idxs_to_idx(&self.views[self.views.len() - 1].shape, &idxs),
            None,
        );
        self._expr_idx(idx, valid)
    }

    pub fn axis_is_masked(&self, axis: isize) -> bool {
        let (_, valid) = self.expr_idxs(None);
        valid
            .vars()
            .iter()
            .any(|n| n.expr().is_some() && n.expr().unwrap() == &format!("idx{axis}"))
    }

    pub fn pad(&self, arg: &[(isize, isize)]) -> Self {
        let mut views = self.views.clone();
        let p = views.pop().unwrap();
        views.push(p.pad(arg));
        ShapeTracker { views }
    }

    pub fn shrink(&self, arg: &[(isize, isize)]) -> Self {
        let mut views = self.views.clone();
        let p = views.pop().unwrap();
        views.push(p.shrink(arg));
        ShapeTracker { views }
    }

    pub fn expand(&self, new_shape: &[isize]) -> Self {
        let mut views = self.views.clone();
        let p = views.pop().unwrap();
        views.push(p.expand(new_shape));
        ShapeTracker { views }
    }

    pub fn reshape(&self, new_shape: &[isize]) -> Self {
        let mut views = self.views.clone();
        if getenv("MERGE_VIEW", 1) > 0 {
            let new_view = self.views[self.views.len() - 1].reshape(new_shape);
            if let Some(nv) = new_view {
                views.pop();
                views.push(nv);
                return ShapeTracker { views };
            }
        }
        views.push(view!(new_shape));
        ShapeTracker { views }
    }

    pub fn permute(&self, axis: &[isize]) -> Self {
        let mut views = self.views.clone();
        let p = views.pop().unwrap();
        views.push(p.permute(axis));
        ShapeTracker { views }
    }

    pub fn stride(&self, mul: &[isize]) -> Self {
        let mut views = self.views.clone();
        let p = views.pop().unwrap();
        views.push(p.stride(mul));
        ShapeTracker { views }
    }

    pub fn unit_stride_axes(&self, ignore_valid: bool) -> Vec<isize> {
        crate::v![i as isize, for (i, st) in self.real_strides(ignore_valid).iter().enumerate(), if st.is_some_and(|s| s == 1)]
    }

    pub fn real_size(&self) -> usize {
        if self.shape().dims.contains(&0) {
            return 0;
        }
        let mut ret = self.expr_idxs(None).0.max().unwrap();
        (ret + 1) as usize
    }
}

fn _expr_view(view: &View, idxs: &[ArcNode], valid: Option<ArcNode>) -> (ArcNode, ArcNode) {
    assert!(idxs.len() == view.shape.len());
    let mut iexpr = vec![num(view.offset)];
    let mut vexpr = if valid.is_some() {
        vec![valid.unwrap()]
    } else {
        vec![]
    };
    for (idx, sh, st, m) in crate::izip!(
        idxs,
        view.shape.clone(),
        view.strides.clone(),
        if view.mask.is_some() {
            view.mask
                .as_ref()
                .unwrap()
                .iter()
                .map(|s| Some(s))
                .collect()
        } else {
            vec![None; view.shape.len()]
        }
    ) {
        if sh != 1 && st != 0 {
            iexpr.push(idx * st);
        }
        if let Some(mm) = m {
            vexpr.extend([idx.ge(num(mm.0)), idx.lt(num(mm.1))]);
        }
    }
    (
        crate::shape::symbolic::sum(&iexpr),
        crate::shape::symbolic::ands(&vexpr),
    )
}
