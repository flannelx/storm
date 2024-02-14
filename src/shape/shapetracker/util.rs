use crate::{
    izip,
    shape::{
        shapetracker::ShapeTracker,
        symbolic::{num, sum, var, ArcNode},
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
    for (i, (&sh, &st)) in izip!(shape[1..].iter(), strides[1..].iter()).enumerate() {
        let i = i;
        if sh == 1 {
            continue;
        }
        if state == 1 || ret[ret.len() - 1].1 == sh * st {
            *ret.last_mut().unwrap() = (
                ret[ret.len() - 1].0 * sh,
                st,
                if st > 0 {
                    if state == 1 {
                        sh
                    } else {
                        ret[ret.len() - 1].2 * sh
                    }
                } else {
                    0
                },
            );
        } else {
            ret.push((sh, st, if st > 0 { sh } else { 0 }));
        }
        state = if let Some(ref m) = mask {
            if st == 0 && m[i].1 - m[i].0 == 1 {
                1
            } else {
                0
            }
        } else {
            if state != 0 {
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

pub fn merge_views(vm2: &View, vm1: &View) -> Option<View> {
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

use crate::prelude::*;

fn un1d(shape: &[isize], mut offset: isize) -> Vec<isize> {
    let strides = strides_for_shape(shape);
    let mut result = vec![];
    for stride in strides {
        let here = if stride > 0 { offset / stride } else { 0 };
        result.push(here);
        offset -= here * stride;
    }
    result
}

// pub fn merge_views(vm2: &View, vm1: &View) -> Option<View> {
//     if vm1.contiguous && vm1.shape == vm2.shape {
//         return Some(vm2.clone());
//     }
//     if vm2.contiguous {
//         return Some(vm1.clone());
//     }
//     let rstrides = ShapeTracker {
//         views: vec![vm2.clone(), vm1.clone()],
//     }
//     .real_strides(false);
//     if vm2.mask.is_none() && vm1.offset == 0 && !rstrides.contains(&None) {
//         return Some(view!(
//             vm1.shape,
//             rstrides
//                 .into_iter()
//                 .map(|s| s.unwrap())
//                 .collect::<Vec<isize>>(),
//             vm2.offset,
//             vm1.mask.clone()
//         ));
//     }
//
//     if let Some(vm1_mask) = &vm1.mask {
//         for &(b, e) in vm1_mask {
//             if !(b < e) {
//                 return Some(view!(
//                     vm1.shape,
//                     vec![0; vm1.shape.len()],
//                     0,
//                     Some(vec![(0, 0); vm1.shape.len()])
//                 ));
//             }
//         }
//         println!("{:?}", vm1);
//         let merged = merge_views(vm2, &vm1.shrink(&vm1_mask));
//         if let Some(m) = merged {
//             return Some(m.pad(&v![(b, s-e), for (&(b, e), s) in izip!(vm1_mask, vm1.shape.clone())]));
//         } else {
//             return None;
//         }
//     }
//
//     let origin = un1d(&vm2.shape, vm1.offset);
//     let mut terms = vec![vec![]; origin.len()];
//     let mut strides = vec![0; vm1.shape.len()];
//     for (d1, &st) in vm1.strides.iter().enumerate() {
//         if st == 0 {
//             continue;
//         }
//         for (d2, (o, s1)) in izip!(origin.iter(), un1d(&vm2.shape, vm1.offset + st)).enumerate() {
//             let s1 = s1 - o;
//             if s1 == 0 {
//                 continue;
//             }
//             terms[d2].push((d1, s1));
//             strides[d1] += s1 * vm2.strides[d2];
//         }
//     }
//     let mut idxs = v![var(&format!("idx{i}"),0, s-1), for (i, s) in vm1.shape.iter().enumerate()];
//     let mut merged_size = 1;
//     let mut merged_term = num(0);
//     let mut extends = vec![];
//     for (term, s, o) in izip!(
//         terms.iter().rev(),
//         vm2.shape.iter().rev(),
//         origin.iter().rev()
//     ) {
//         merged_term = &merged_term
//             + sum(&v![&idxs[d1] * (num(s1) * &merged_term),for &(d1, s1) in term.iter()])
//             + num(o * merged_size);
//         merged_size *= s;
//         if merged_term.ge(num(merged_size)).is_num() && merged_term.lt(num(0)).is_num() {
//             extends.push((merged_size, merged_term));
//             merged_size = 1;
//             merged_term = num(0);
//         }
//     }
//     if merged_term.is_var() || (merged_term.is_num() && merged_term.num_val().unwrap() > 0) {
//         return None;
//     }
//     let vm2_shape = v![*s, for (s, _) in extends.iter().rev()];
//     if vm2_shape != vm2.shape {
//         let reshape_vm2 = vm2.reshape(&vm2_shape);
//         if let Some(new_shape) = reshape_vm2 {
//             return merge_views(&new_shape, vm1);
//         } else {
//             return None;
//         }
//     }
//
//     if let Some(vm2_mask) = &vm2.mask {
//         let mut newb = vec![0; vm1.shape.len()];
//         let mut newe = vm1.shape.clone();
//         let mut bad = false;
//         //x: ((&(isize, isize), &isize), &(isize, ArcNode)) // size = 24 (0x18), align = 0x8
//         for (d2, ((&(b, e), &o), (_, t))) in vm2_mask
//             .iter()
//             .zip(origin.iter())
//             .zip(extends.iter().rev())
//             .enumerate()
//         {
//             if t.min().is_some_and(|m| m < b) || t.max().is_some_and(|m| m >= e) {
//                 continue;
//             }
//             let term = &terms[d2];
//             if term.len() != 1 {
//                 if term.len() > 0 && newe.len() > 0 {
//                     newe[0] = 0;
//                 }
//                 else {
//                     bad = true;
//                 }
//                 continue;
//             }
//             let (d1, s1) = term[0];
//             newb[d1] = newb[d1].max(((( if s1 > 0  { b - o } else {e - o - 1}) as f32 / s1 as f32).ceil() as isize));
//             newe[d1] = newe[d1].min((if s1 < 0 { b - o } else  { e - o - 1}) / s1 + 1);
//         }
//
//         for (&b, &e, &s) in izip!(newb.iter(), newe.iter(), vm1.shape.iter()) {
//             if b != 0 || e != s {
//                 return merge_views(vm2, &view!(vm1.shape,vm1.strides, vm1.offset, Some(v![x, for x in izip!(newb, newe)])));
//             }
//         }
//         if bad {
//             return None;
//         }
//     }
//     Some(view!(vm1.shape, strides, crate::utils::sum(&v![o * s, for (o, s) in izip!(origin, vm2.strides.clone())]) + vm2.offset))
// }

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
    if let Some(merged_view) = merge_views(view, &new_view) {
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
