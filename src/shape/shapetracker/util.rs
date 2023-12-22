use crate::{
    izip,
    shape::{
        shapetracker::ShapeTracker,
        symbolic::{sum, ArcNode},
    },
    v,
};

use super::view::View;

pub fn to_shape_strides(shape: &[isize], strides: &[isize]) -> Vec<(isize, isize)> {
    assert!(shape.len() == strides.len());
    let mut ret = vec![(shape[0], strides[0])];
    for i in 1..shape.len() {
        let ret_last = ret.last_mut().unwrap();
        if ret_last.1 == shape[i] * strides[i] || ret_last.0 == 1 {
            *ret_last = (ret_last.0 * shape[i], strides[i]);
        } else if shape[i] == 1 {
            continue;
        } else {
            ret.push((shape[i], strides[i]));
        }
    }
    ret
}

pub fn _merge_dims(
    shape: &[isize],
    strides: &[isize],
    mask: Option<Vec<(isize, isize)>>,
) -> Vec<(isize, isize, isize)> {
    if shape.len() == 0 {
        return vec![];
    }
    assert!(shape.len() == strides.len());
    let mut ret = vec![(
        shape[0],
        strides[0],
        if strides[0] > 0 { shape[0] } else { 0 },
    )];
    let mut state = if mask.is_some()
        && strides[0] == 0
        && shape[0] != 1
        && mask.as_ref().unwrap()[0].1 - mask.as_ref().unwrap()[0].0 == 1
    {
        1
    } else {
        0
    };
    let start_i = 1;
    for (i, (&sh, &st)) in izip!(shape[1..].iter(), strides[1..].iter()).enumerate() {
        let i = start_i + i;
        if sh == 1 {
            continue;
        }
        if state == 1 || ret[ret.len() - 1].1 == sh * st {
            *ret.last_mut().unwrap() = (
                ret[ret.len() - 1].0 * sh,
                st,
                if state == 1 {
                    sh
                } else {
                    if st > 0 {
                        ret[ret.len() - 1].2 * sh
                    } else {
                        0
                    }
                },
            );
        }
        else {
            ret.push((sh, st, if st > 0 { sh } else { 0 }));
        }
        state = if let Some(ref m) = mask {
            if st == 0 && m[i].1 - m[i].0 == 1 {
                1
            } else {
                0
            }
        } else {
            if state !=0 {
                2
            } else {
                0
            }
        };
    }
    ret
}

pub fn strides_for_shape(shape: &[isize]) -> Vec<isize> {
    let mut strides = vec![1; shape.len()];
    let mut tmp = 1;
    strides
        .iter_mut()
        .zip(shape.iter())
        .rev()
        .for_each(|(st, sh)| {
            *st = tmp;
            tmp *= *sh
        });
    filter_strides(shape, &strides)
}

pub fn filter_strides(shape: &[isize], strides: &[isize]) -> Vec<isize> {
    shape
        .into_iter()
        .zip(strides.into_iter())
        .map(|(sh, st)| if *sh != 1 { *st } else { 0 })
        .collect()
}

pub fn idxs_to_idx(shape: &[isize], idxs: &[ArcNode]) -> ArcNode {
    assert!(shape.len() == idxs.len());
    let mut acc = 1;
    let mut ret = vec![];
    for (tidx, d) in idxs.iter().zip(shape.iter()).rev() {
        ret.push(tidx * acc);
        acc *= d;
    }
    sum(&ret)
}

pub fn merge_view(vm2: &View, vm1: &View) -> Option<View> {
    if vm1.contiguous && vm1.shape == vm2.shape {
        return Some(vm2.clone());
    }
    if vm2.contiguous {
        return Some(vm1.clone());
    }
    if vm2.mask.is_some() || vm1.offset != 0 {
        return None;
    }
    let strides = ShapeTracker {
        views: vec![vm2.clone(), vm1.clone()],
    }
    .real_strides(false);
    if strides.contains(&None) {
        return None;
    }
    let strides = v![s.unwrap(), for s in strides];
    return Some(View::new(
        &vm1.shape,
        Some(strides),
        Some(vm2.offset),
        vm1.mask.clone(),
    ));
    // if vm2.mask.is_some() {
    //     return None;
    // }
    // let mst = ShapeTracker::new(&vm1.shape, Some([vm2.clone(), vm1.clone()].to_vec()));
    // let strides = mst.real_strides(false);
    // //println!("vm1 st real strides {:?}", strides);
    // if strides.iter().any(|n| n.is_none()) {
    //     return None;
    // }
    // let strides = strides.iter().map(|s| s.unwrap()).collect();
    // Some(View::new(
    //     &vm1.shape,
    //     Some(strides),
    //     Some(mst.real_offset()),
    //     vm1.mask.clone(),
    // ))
}

pub fn _reshape(view: &View, new_shape: &[isize]) -> (View, bool) {
    let (shape, mask, strides, offset) = (&view.shape, &view.mask, &view.strides, view.offset);
    if shape
        .iter()
        .filter(|&&x| x != 1)
        .eq(new_shape.iter().filter(|&&x| x != 1))
    {
        let mut new_strides: Vec<isize> = shape
            .iter()
            .zip(strides.iter())
            .filter(|(&x, _)| x != 1)
            .rev()
            .map(|(_, &y)| y)
            .collect();
        let new_strides_tuple: Vec<isize> = new_shape
            .iter()
            .map(|&x| {
                if x == 1 {
                    0
                } else {
                    new_strides.pop().unwrap()
                }
            })
            .collect();
        let mut new_mask_tuple = None;
        if let Some(m) = mask {
            for (&x, &y) in shape.iter().zip(m.iter()) {
                if x == 1 && y != (0, 1) {
                    new_mask_tuple = Some(vec![(0, 0); new_shape.len()]);
                    break;
                }
            }
            if new_mask_tuple.is_none() {
                let mut new_mask: Vec<(isize, isize)> = shape
                    .iter()
                    .zip(m.iter())
                    .filter(|(&sh, _)| sh != 1)
                    .map(|(_, &mm)| mm)
                    .rev()
                    .collect();
                new_mask_tuple = Some(
                    new_shape
                        .iter()
                        .map(|&x| {
                            if x == 1 {
                                (0, 1)
                            } else {
                                new_mask.pop().unwrap()
                            }
                        })
                        .collect::<_>(),
                );
            }
        };
        return (
            View::new(
                new_shape,
                Some(new_strides_tuple),
                Some(offset),
                new_mask_tuple,
            ),
            false,
        );
    }
    let new_view = View::new(new_shape, None, None, None);
    if view.contiguous {
        return (new_view, false);
    }
    if let Some(merged_view) = merge_view(view, &new_view) {
        return (merged_view, false);
    }
    return (new_view, true);
}

pub fn get_pad_args(
    shape: &[isize],
    arg: &[(isize, isize)],
) -> (Vec<(isize, isize)>, Vec<(isize, isize)>) {
    let mut ret = (vec![], vec![]);
    for (&s, &(b, e)) in shape.iter().zip(arg.iter()) {
        ret.0.push((-b, s + e))
    }
    for (&s, &(b, _)) in shape.iter().zip(arg.iter()) {
        ret.1.push((b, s + b))
    }
    ret
}

pub fn get_unsafe_resize_offset(strides: &[isize], arg: &[(isize, isize)]) -> isize {
    strides.iter().zip(arg.iter()).map(|(&s, &x)| s * x.0).sum()
}
