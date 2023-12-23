use storm::prelude::*;
use storm::shape::shapetracker::{view::*, ShapeTracker};

#[test]
fn test_reshape_doesnt_multiview() {
    let mut st = ShapeTracker::new(
        &[256, 256, 2, 2, 2, 2, 2, 256, 8, 2],
        Some(vec![view!(
            [256, 256, 2, 2, 2, 2, 2, 256, 8, 2],
            [0, 8, 0, 4, 0, 0, 2, 16384, 2048, 1],
            0
        )]),
    );
    st.reshape(&[128, 2, 256, 2, 2, 2, 2, 2, 256, 8, 2]);
    assert!(st.views.len() == 1)
}

#[test]
fn test_real_doesnt_simplify_1() {
    let mut st = ShapeTracker::new(
        &[8, 6, 11],
        Some(vec![
            view!([8, 3, 1, 2, 11, 1], [33, 11, 0, 0, 1, 0]),
            view!([8, 6, 11], [66, 11, 1]),
        ]),
    );

    assert!(
        st.real_strides(false) == vec![Some(33), None, Some(1)],
        // "{:?} != {:?}",
        // st.real_strides(false),
        // vec![Some(33), None, Some(1)]
    );
}

#[test]
fn test_real_doesnt_simplify_2() {
    let mut st = ShapeTracker::new(
        &[4, 4, 3, 3],
        Some(vec![
            view!([2, 2, 4, 3, 3], [72, 9, 18, -3, -1], 8),
            view!([4, 4, 3, 3], [36, 9, 3, 1]),
        ]),
    );

    assert!(st.real_strides(false) == vec![None, Some(18), Some(-3), Some(-1)],);
}

#[test]
fn test_realstrides() {
    let mut st = ShapeTracker::new(
        &[16, 32, 4],
        Some(vec![
            view!([2048], [1], 0, [(0, 512)]),
            view!([16, 32, 4], [128, 4, 1]),
        ]),
    );
    let rs = st.real_strides(false);
    assert!(rs == vec![None, Some(4), Some(1)]);
}

#[test]
fn test_real_simplifies_1() {
    let mut st = ShapeTracker::new(
        &[1, 3, 2, 11, 26, 1, 1, 3],
        Some(vec![
            view!([1, 3, 2, 11, 4, 28], [0, 308, 0, 28, 0, 1]),
            view!([1, 3, 2, 11, 26, 1, 1, 3], [0, 2464, 0, 112, 1, 0, 0, 29]),
        ]),
    );
    let rst: Vec<isize> = st.real_strides(false).iter().map(|s| s.unwrap()).collect();
    st = st.simplify();
    assert!(st.views.len() == 1);
    assert!(st.views.last().unwrap().strides == rst);
}

#[test]
fn test_real_simplifies_2() {
    let mut st = ShapeTracker::new(
        &[8, 1, 6, 10, 28, 3, 2, 1],
        Some(vec![
            view!([8, 3, 3, 11, 2, 28], [924, 308, 0, 28, 0, 1]),
            view!(
                [8, 1, 6, 10, 28, 3, 2, 1],
                [5544, 0, 0, 56, 1, 1848, 672, 0]
            ),
        ]),
    );
    let rst: Vec<isize> = st.real_strides(false).iter().map(|s| s.unwrap()).collect();
    st = st.simplify();
    assert!(st.views.len() == 1);
    assert!(st.views.last().unwrap().strides == rst);
}

mod test_index_expr_2d {
    use std::sync::Arc;

    use crate::ShapeTracker;
    use itertools::iproduct;
    use storm::shape::shapetracker::view::View;
    use storm::shape::symbolic::var;
    use storm::shape::symbolic::ArcNode;
    const SHAPES: [[isize; 2]; 5] = [[30, 5], [15, 10], [15, 1], [5, 10], [5, 1]];
    const OFFSETS: [isize; 5] = [0, 1, 15, 28, 10000];

    struct Setup {
        shape: Vec<Vec<isize>>,
        offsets: Vec<isize>,
        sts: Vec<ShapeTracker>,
        node_exprs: Vec<Arc<dyn Fn(ArcNode) -> ArcNode>>,
        idxs_exprs: Vec<Arc<dyn Fn(Vec<ArcNode>) -> ArcNode>>,
    }

    impl Default for Setup {
        fn default() -> Self {
            let mut sts = vec![];
            let mut osts = vec![];
            let mut shs = vec![];
            for base_shape in SHAPES.iter() {
                for offset in OFFSETS.iter() {
                    sts.push(ShapeTracker::new(
                        base_shape,
                        Some(vec![View::new(base_shape, None, Some(*offset), None)]),
                    ));
                    osts.push(*offset);
                    shs.push(base_shape.to_vec());
                }
            }
            Self {
                shape: shs,
                offsets: osts,
                sts,
                node_exprs: vec![],
                idxs_exprs: vec![],
            }
        }
    }

    impl Setup {
        fn teardown(self) {
            for ((((st, offset), shape), node_expr), idxs_expr) in self
                .sts
                .into_iter()
                .zip(self.offsets.into_iter())
                .zip(self.shape.into_iter())
                .zip(self.node_exprs.into_iter())
                .zip(self.idxs_exprs.into_iter())
            {
                let numel = shape.iter().product::<isize>();
                // println!(
                //     "{} == {} {:?} {:?} offset: {}",
                //     node_expr(Self::default_idx(&st.shape())),
                //     st.expr_node(None).0,
                //     st.views.last().unwrap().shape,
                //     &st.shape(),
                //     &st.views.last().unwrap().offset,
                // );
                assert!(node_expr(Self::default_idx(&st.shape())) == st.expr_node(None).0);
                // assert node_expr(self.default_idx(st.shape)) == st.expr_node(None)[0]
                // assert node_expr(self.default_idx(st.shape)) == st.expr_node('idx')[0]
                Self::check_bounds(node_expr(Self::default_idx(&st.shape())), offset, numel);
                for (i1, i2) in [
                    (0, numel - 1),
                    (7, 203),
                    (2, 5),
                    (0, 0),
                    (numel, numel),
                    (0, numel),
                    (0, numel + 1),
                    (numel + 100, numel + 100),
                ] {
                    let idx = var("idx", i1, i2);
                    assert!(node_expr(idx.clone()) == st.expr_node(Some(idx.clone())).0);
                    Self::check_bounds(node_expr(idx), offset, numel);
                }
                assert!(node_expr(Self::default_idx(&st.shape())) == st.expr_node(None).0);
                Self::check_bounds(idxs_expr(Self::default_idxs(&st.shape())), offset, numel);
                let idx0s = [
                    (0, 0),
                    (0, (st.shape()[0] - 1).min(1)),
                    (0, st.shape()[0] - 1),
                    ((st.shape()[0] - 1).min(3), (st.shape()[0] - 1).min(6)),
                    (st.shape()[0] - 1, st.shape()[0] - 1),
                ];
                let idx1s = [
                    (0, 0),
                    (0, (st.shape()[1] - 1).min(1)),
                    (0, st.shape()[1] - 1),
                    ((st.shape()[1] - 1).min(3), (st.shape()[1] - 1).min(6)),
                    (st.shape()[1] - 1, st.shape()[1] - 1),
                ];
                let idx2s: Vec<Option<(isize, isize)>> = if st.shape().len() == 3 {
                    [
                        (0, 0),
                        (0, (st.shape()[2] - 1).min(1)),
                        (0, st.shape()[2] - 1),
                        ((st.shape()[2] - 1).min(3), (st.shape()[2] - 1).min(6)),
                        (st.shape()[2] - 1, st.shape()[2] - 1),
                    ]
                    .iter()
                    .map(|x| Some(*x))
                    .collect()
                } else {
                    vec![None; idx0s.len()]
                };

                for idx0 in idx0s.iter() {
                    for idx1 in idx1s.iter() {
                        for idx2 in idx2s.iter() {
                            let it = if let Some(i2) = idx2 {
                                vec![idx0, idx1, i2]
                            } else {
                                vec![idx0, idx1]
                            };
                            let mut idxs = vec![];
                            for (i, &&idx) in it.iter().enumerate() {
                                idxs.push(var(&format!("idx{}", i), idx.0, idx.1));
                            }
                            assert!(idxs_expr(idxs.clone()) == st.expr_idxs(Some(idxs.clone())).0);
                            Self::check_bounds(idxs_expr(idxs), offset, numel);
                        }
                    }
                }
            }
        }

        fn default_idx(shape: &Vec<isize>) -> ArcNode {
            var("idx", 0, shape.iter().product::<isize>() - 1)
        }

        fn default_idxs(shape: &Vec<isize>) -> Vec<ArcNode> {
            shape
                .iter()
                .enumerate()
                .map(|(i, d)| var(&format!("idx{}", i), 0, d - 1))
                .collect()
        }

        fn check_bounds(expr: ArcNode, offset: isize, numel: isize) {
            assert!(expr.min().unwrap() >= offset);
            assert!(expr.max().unwrap() <= offset + numel - 1);
        }
    }

    #[test]
    fn test_noop() {
        let mut s = Setup::default();
        for ((st, base_shape), &offset) in s.sts.iter().zip(s.shape.iter()).zip(s.offsets.iter()) {
            let pd = base_shape.iter().product::<isize>();
            let b1 = base_shape[1];
            s.node_exprs.push(Arc::new(move |idx| idx % pd + offset));
            s.idxs_exprs
                .push(Arc::new(move |idxs| &idxs[0] * b1 + &idxs[1] + offset));
        }
        s.teardown();
    }

    #[test]
    fn test_permute() {
        let mut s = Setup::default();
        for ((st, base_shape), &offset) in
            s.sts.iter_mut().zip(s.shape.iter()).zip(s.offsets.iter())
        {
            *st = st.permute(&[1, 0]);
            let pd = base_shape.iter().product::<isize>();
            let b0 = base_shape[0];
            let b1 = base_shape[1];
            s.node_exprs.push(Arc::new(move |idx| {
                &idx % b0 * b1 + &idx / b0 % b1 + offset
            }));
            s.idxs_exprs
                .push(Arc::new(move |idxs| &idxs[0] + &idxs[1] * b1 + offset));
        }
        s.teardown();
    }

    #[test]
    fn test_reshape() {
        let mut s = Setup::default();
        for ((st, base_shape), &offset) in
            s.sts.iter_mut().zip(s.shape.iter()).zip(s.offsets.iter())
        {
            let pd = base_shape.iter().product::<isize>();
            let b0 = base_shape[0];
            let b1 = base_shape[1];
            *st = st.reshape(&[b0, 1, b1]);

            s.node_exprs.push(Arc::new(move |idx| idx % pd + offset));
            s.idxs_exprs
                .push(Arc::new(move |idxs| &idxs[0] * b1 + &idxs[2] + offset));
        }
        s.teardown();
    }

    #[test]
    fn test_reshape_expand() {
        let mut s = Setup::default();
        for ((st, base_shape), &offset) in
            s.sts.iter_mut().zip(s.shape.iter()).zip(s.offsets.iter())
        {
            let b0 = base_shape[0];
            let b1 = base_shape[1];
            *st = st.reshape(&[b0, 1, b1]).expand(&[b0, b1, b1]);
            s.node_exprs.push(Arc::new(move |idx| {
                &idx / (b1 * b1) % b0 * b1 + &idx % b1 + offset
            }));
            s.idxs_exprs
                .push(Arc::new(move |idxs| &idxs[0] * b1 + &idxs[2] + offset));
        }
        s.teardown();
    }

    #[test]
    fn test_permute_expand_1() {
        let mut s = Setup::default();
        for ((st, base_shape), &offset) in
            s.sts.iter_mut().zip(s.shape.iter()).zip(s.offsets.iter())
        {
            let pd = base_shape.iter().product::<isize>();
            let b0 = base_shape[0];
            let b1 = base_shape[1];
            *st = st.permute(&[1, 0]).reshape(&[b0 / 5, 1, b1 * 5]);
            s.node_exprs.push(Arc::new(move |idx| {
                &idx % pd % b0 * b1 + &idx / b0 % b1 + offset
            }));
            s.idxs_exprs.push(Arc::new(move |idxs| {
                (&idxs[0] * (b1 * 5) + &idxs[2]) % b0 * b1
                    + (&idxs[0] * (b1 * 5) + &idxs[2]) / b0
                    + offset
            }));
        }
        s.teardown();
    }

    #[test]
    fn test_permute_expand_2() {
        let mut s = Setup::default();
        for ((st, base_shape), &offset) in
            s.sts.iter_mut().zip(s.shape.iter()).zip(s.offsets.iter())
        {
            let pd = base_shape.iter().product::<isize>();
            let b0 = base_shape[0];
            let b1 = base_shape[1];
            *st = st.permute(&[1, 0]).reshape(&[1, b0 / 5, b1 * 5]);
            s.node_exprs.push(Arc::new(move |idx| {
                &idx % pd % b0 * b1 + &idx / b0 % b1 + offset
            }));
            s.idxs_exprs.push(Arc::new(move |idxs| {
                (&idxs[1] * (b1 * 5) + &idxs[2]) % b0 * b1
                    + (&idxs[1] * (b1 * 5) + &idxs[2]) / b0
                    + offset
            }));
        }
        s.teardown();
    }
}
