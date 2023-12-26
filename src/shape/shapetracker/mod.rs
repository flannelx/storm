pub mod util;
pub mod view;

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

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

    pub fn from_shape(shape: &[isize]) -> Self {
        let views = vec![View::new(shape, None, None, None)];
        Self { views }
    }

    pub fn contiguous(&self) -> bool {
        self.views.len() == 1 && self.views[0].contiguous
    }

    pub fn shape(&self) -> Vec<isize> {
        self.views.last().unwrap().shape.clone()
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

    pub fn real_offset(&self) -> isize {
        let (real_offset, _) = self.expr_node(Some(var("zero", 0, 0)));
        assert!(real_offset.is_num());
        real_offset.num_val().unwrap()
    }

    pub fn real_strides(&self, ignore_valid: bool) -> Vec<Option<isize>> {
        let last_view = self.views.last().unwrap();
        if self.views.len() == 1 && last_view.mask.is_none() {
            return last_view.strides.iter().map(|st| Some(*st)).collect();
        };
        let mut ret = vec![None; last_view.shape.len()];
        let idxs: Vec<ArcNode> = self
            .shape()
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
        if self.views.len() < 2 {
            return self.clone();
        }
        let l = self.views.len();
        let mut ret = self.clone();
        if let Some(new_view) = merge_view(&self.views[l - 2], &self.views[l - 1]) {
            ret.views.pop();
            ret.views.pop();
            ret.views.push(new_view);
            ret = ret.simplify();
        }
        ret
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
            self.shape()
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

    pub fn expr_node(&self, idx: Option<ArcNode>) -> (ArcNode, ArcNode) {
        let idx = if let Some(i) = idx {
            i
        } else {
            var(
                "idx",
                0,
                self.views.last().unwrap().shape.iter().product::<isize>() - 1,
            )
        };
        self._expr_idx(
            self.views[self.views.len() - 1].expr_node(Some(idx.clone())),
            self.views[self.views.len() - 1].expr_node_mask(idx, None),
        )
    }

    pub fn expr_node_str(&self, _idx: &str) -> (ArcNode, ArcNode) {
        let idx = var("idx", 0, self.shape().iter().product());
        self._expr_idx(
            self.views[self.views.len() - 1].expr_node(Some(idx.clone())),
            self.views[self.views.len() - 1].expr_node_mask(idx, None),
        )
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
        let new_view = self.views[self.views.len() - 1].reshape(new_shape);
        let mut views = self.views.clone();
        if let Some(nv) = new_view {
            views.pop();
            views.push(nv);
            return ShapeTracker { views };
        }
        views.push(view!(new_shape));
        // if new_view.is_none() {
        //     let extra = View::new(new_shape, None, None, None);
        //     if let Some(merged_view) = merge_view(self.views.last().unwrap(), &extra) {
        //         views.pop();
        //         views.push(merged_view);
        //     } else {
        //         views.push(extra);
        //     }
        // } else {
        //     views.pop();
        //     views.push(new_view.unwrap());
        // }
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
}
