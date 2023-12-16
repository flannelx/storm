use std::sync::Arc;

use storm::shape::symbolic::*;

#[rustfmt::skip]
fn helper_test_variable(v: ArcNode, n:isize, m:isize, s: &str) {
    assert!(format!("{}", v.render(Arc::new(Python), None, false)) == s, "render {} != {}", v.render(Arc::new(Python), None, false), s);
    assert!(v.min().unwrap() == n, "min    {} != {}", v.min().unwrap(), n);
    assert!(v.max().unwrap() == m, "max    {} != {}", v.max().unwrap(), m);
}

#[test]
fn test_ge() {
    helper_test_variable(var("a", 3, 8).ge(num(77)), 0, 0, "0");
    helper_test_variable(var("a", 3, 8).ge(num(9)), 0, 0, "0");
    helper_test_variable(var("a", 3, 8).ge(num(8)), 0, 1, "((a*-1)<-7)");
    helper_test_variable(var("a", 3, 8).ge(num(4)), 0, 1, "((a*-1)<-3)");
    helper_test_variable(var("a", 3, 8).ge(num(3)), 1, 1, "1");
    helper_test_variable(var("a", 3, 8).ge(num(2)), 1, 1, "1");
}

#[test]
fn test_lt() {
    helper_test_variable(var("a", 3, 8).lt(num(77)), 1, 1, "1");
    helper_test_variable(var("a", 3, 8).lt(num(9)), 1, 1, "1");
    helper_test_variable(var("a", 3, 8).lt(num(8)), 0, 1, "(a<8)");
    helper_test_variable(var("a", 3, 8).lt(num(4)), 0, 1, "(a<4)");
    helper_test_variable(var("a", 3, 8).lt(num(3)), 0, 0, "0");
    helper_test_variable(var("a", 3, 8).lt(num(2)), 0, 0, "0");
}

#[test]
fn test_ge_divides() {
    let expr = (var("idx", 0, 511) * 4 + var("FLOAT4_INDEX", 0, 3)).lt(num(512));
    helper_test_variable(expr.clone(), 0, 1, "((idx*4)<512)");
    helper_test_variable(expr / 4, 0, 1, "(idx<128)");
}

#[test]
fn test_ge_divides_and() {
    let expr = ands(&[
        (var("idx1", 0, 511) * 4 + var("FLOAT4_INDEX", 0, 3)).lt(num(512)),
        (var("idx2", 0, 511) * 4 + var("FLOAT4_INDEX", 0, 3)).lt(num(512)),
    ]);
    helper_test_variable(expr / 4, 0, 1, "((idx1<128) and (idx2<128))");
    let expr = ands(&[
        (var("idx1", 0, 511) * 4 + var("FLOAT4_INDEX", 0, 3)).lt(num(512)),
        (var("idx2", 0, 511) * 4 + var("FLOAT8_INDEX", 0, 7)).lt(num(512)),
    ]);
    helper_test_variable(
        expr / 4,
        0,
        1,
        "((((FLOAT8_INDEX//4)+idx2)<128) and (idx1<128))",
    );
}

#[test]
fn test_lt_factors() {
    let expr = ands(&[(var("idx1", 0, 511) * 4 + var("FLOAT4_INDEX", 0, 256)).lt(num(512))]);
    helper_test_variable(expr, 0, 1, "(((idx1*4)+FLOAT4_INDEX)<512)");
}

#[test]
fn test_div_becomes_num() {
    assert!((var("a", 2, 3) / 2).is_num())
}

#[test]
fn test_var_becomes_num() {
    assert!(var("a", 2, 2).is_num())
}

#[test]
fn test_equality() {
    let idx1 = var("idx1", 0, 3);
    let idx2 = var("idx2", 0, 3);
    assert!(idx1 == idx1);
    assert!(idx1 != idx2);
    assert!(&idx1 * 4 == &idx1 * 4);
    assert!(&idx1 * 4 != &idx1 * 3);
    assert!(&idx1 * 4 != &idx1 + 4);
    assert!(&idx1 * 4 != &idx2 * 4);
    assert!(&idx1 + &idx2 == &idx1 + &idx2);
    assert!(&idx1 + &idx2 == &idx2 + &idx1);
    assert!(&idx1 + &idx2 != idx2);
}

#[test]
fn test_factorize() {
    let a = var("a", 0, 8);
    helper_test_variable(&a * 2 + &a * 3, 0, 8 * 5, "(a*5)")
}

#[test]
fn test_factorize_no_mul() {
    let a = var("a", 0, 8);
    helper_test_variable(&a + &a * 3, 0, 8 * 4, "(a*4)")
}

#[test]
fn test_neg() {
    helper_test_variable(-var("a", 0, 8), -8, 0, "(a*-1)")
}

#[test]
fn test_add_1() {
    helper_test_variable(var("a", 0, 8) + 1, 1, 9, "(1+a)");
}

#[test]
fn test_add_num_1() {
    helper_test_variable(var("a", 0, 8) + num(1), 1, 9, "(1+a)");
}

#[test]
fn test_sub_1() {
    helper_test_variable(var("a", 0, 8) - 1, -1, 7, "(-1+a)");
}

#[test]
fn test_sub_num_1() {
    helper_test_variable(var("a", 0, 8) - num(1), -1, 7, "(-1+a)");
}

#[test]
fn test_mul_0() {
    helper_test_variable(var("a", 0, 8) * 0, 0, 0, "0");
}

#[test]
fn test_mul_1() {
    helper_test_variable(var("a", 0, 8) * 1, 0, 8, "a");
}

#[test]
fn test_mul_neg_1() {
    helper_test_variable((var("a", 0, 2) * -1) / 3, -1, 0, "((((a*-1)+3)//3)+-1)");
}

#[test]
fn test_mul_2() {
    helper_test_variable(var("a", 0, 8) * 2, 0, 16, "(a*2)");
}

#[test]
fn test_div_1() {
    helper_test_variable(var("a", 0, 8) / 1, 0, 8, "a");
}

#[test]
fn test_mod_1() {
    helper_test_variable(var("a", 0, 8) % 1, 0, 0, "0");
}

#[test]
fn test_add_min_max() {
    helper_test_variable(var("a", 0, 8) * 2 + 12, 12, 16 + 12, "((a*2)+12)");
}

#[test]
fn test_div_min_max() {
    helper_test_variable(var("a", 0, 7) / 2, 0, 3, "(a//2)");
}

#[test]
fn test_div_neg_min_max() {
    helper_test_variable(var("a", 0, 7) / -2, -3, 0, "((a//2)*-1)");
}

#[test]
fn test_sum_div_min_max() {
    helper_test_variable(
        sum(&[var("a", 0, 7), var("b", 0, 3)]) / 2,
        0,
        5,
        "((a+b)//2)",
    );
}

#[test]
fn test_sum_div_factor() {
    helper_test_variable(
        sum(&[var("a", 0, 7) * 4, var("b", 0, 3) * 4]) / 2,
        0,
        20,
        "((a*2)+(b*2))",
    );
}

#[test]
fn test_sum_div_some_factor() {
    helper_test_variable(
        sum(&[var("a", 0, 7) * 5, var("b", 0, 3) * 4]) / 2,
        0,
        23,
        "(((a*5)//2)+(b*2))",
    );
}

#[test]
fn test_sum_div_some_partial_factor() {
    helper_test_variable(
        sum(&[var("a", 0, 7) * 6, var("b", 0, 7) * 6]) / 16,
        0,
        5,
        "(((a*3)+(b*3))//8)",
    );
    helper_test_variable(
        sum(&[num(16), var("a", 0, 7) * 6, var("b", 0, 7) * 6]) / 16,
        1,
        6,
        "((((a*3)+(b*3))//8)+1)",
    );
}

#[test]
fn test_sum_div_no_factor() {
    helper_test_variable(
        sum(&[var("a", 0, 7) * 5, var("b", 0, 3) * 5]) / 2,
        0,
        25,
        "(((a*5)+(b*5))//2)",
    );
}

#[test]
fn test_mod_factor() {
    // NOTE: even though the mod max is 50, it can't know this without knowing about the mul
    helper_test_variable(
        sum(&[var("a", 0, 7) * 100, var("b", 0, 3) * 50]) % 100,
        0,
        99,
        "((b*50)%100)",
    );
}

#[test]
fn test_sum_div_const() {
    helper_test_variable(sum(&[var("a", 0, 7) * 4, num(3)]) / 4, 0, 7, "a");
}

#[test]
fn test_sum_div_const_big() {
    helper_test_variable(sum(&[var("a", 0, 7) * 4, num(3)]) / 16, 0, 1, "(a//4)");
}

#[test]
fn test_mod_mul() {
    helper_test_variable((var("a", 0, 5) * 10) % 9, 0, 5, "a");
}

#[test]
fn test_mul_mul() {
    helper_test_variable((var("a", 0, 5) * 10) * 9, 0, 5 * 10 * 9, "(a*90)");
}

#[test]
fn test_div_div() {
    helper_test_variable((var("a", 0, 1800) / 10) / 9, 0, 20, "(a//90)");
}

#[test]
fn test_distribute_mul() {
    helper_test_variable(
        sum(&[var("a", 0, 3), var("b", 0, 5)]) * 3,
        0,
        24,
        "((a*3)+(b*3))",
    );
}

#[test]
fn test_mod_mul_sum() {
    helper_test_variable(
        sum(&[var("b", 0, 2), var("a", 0, 5) * 10]) % 9,
        0,
        7,
        "(a+b)",
    );
}

#[test]
fn test_sum_0() {
    helper_test_variable(sum(&[var("a", 0, 7)]), 0, 7, "a");
}

#[test]
fn test_mod_remove() {
    helper_test_variable(var("a", 0, 6) % 100, 0, 6, "a");
}

#[test]
fn test_big_mod() {
    // NOTE: we no longer support negative variables
    //helper_test_variable(var("a", -20, 20) % 10, -9, 9, "(a%10)");
    //helper_test_variable(var("a", -20, 0) % 10, -9, 0, "(a%10)");
    //helper_test_variable(var("a", -20, 1) % 10, -9, 1, "(a%10)");
    helper_test_variable(var("a", 0, 20) % 10, 0, 9, "(a%10)");
    //helper_test_variable(var("a", -1, 20) % 10, -1, 9, "(a%10)");
}

#[test]
fn test_gt_remove() {
    helper_test_variable(var("a", 0, 6).ge(num(25)), 0, 0, "0");
}

#[test]
fn test_lt_remove() {
    helper_test_variable(var("a", 0, 6).lt(num(-3)), 0, 0, "0");
    helper_test_variable(var("a", 0, 6).lt(num(3)), 0, 1, "(a<3)");
    helper_test_variable(var("a", 0, 6).lt(num(8)), 1, 1, "1");
}

#[test]
fn test_lt_sum_remove() {
    helper_test_variable((var("a", 0, 6) + 2).lt(num(3)), 0, 1, "(a<1)");
}

#[test]
fn test_and_fold() {
    helper_test_variable(ands(&[num(0), var("a", 0, 1)]), 0, 0, "0");
}

#[test]
fn test_and_remove() {
    helper_test_variable(ands(&[num(1), var("a", 0, 1)]), 0, 1, "a");
}

#[test]
fn test_mod_factor_negative() {
    helper_test_variable(
        sum(&[num(-29), var("a", 0, 10), var("b", 0, 10) * 28]) % 28,
        0,
        27,
        "((27+a)%28)",
    );
    helper_test_variable(
        sum(&[num(-29), var("a", 0, 100), var("b", 0, 10) * 28]) % 28,
        0,
        27,
        "((27+a)%28)",
    );
}

#[test]
fn test_sum_combine_num() {
    helper_test_variable(sum(&[num(29), var("a", 0, 10), num(-23)]), 6, 16, "(6+a)");
}

#[test]
fn test_sum_num_hoisted_and_factors_cancel_out() {
    helper_test_variable(
        sum(&[var("a", 0, 1) * -4 + 1, var("a", 0, 1) * 4]),
        1,
        1,
        "1",
    );
}

#[test]
fn test_div_factor() {
    helper_test_variable(
        sum(&[num(-40), var("a", 0, 10) * 2, var("b", 0, 10) * 40]) / 40,
        -1,
        9,
        "(-1+b)",
    );
}

#[test]
fn test_mul_div() {
    helper_test_variable((var("a", 0, 10) * 4) / 4, 0, 10, "a");
}

#[test]
fn test_mul_div_factor_mul() {
    helper_test_variable((var("a", 0, 10) * 8) / 4, 0, 20, "(a*2)");
}

#[test]
fn test_mul_div_factor_div() {
    helper_test_variable((var("a", 0, 10) * 4) / 8, 0, 5, "(a//2)"); //
}

#[test]
fn test_div_remove() {
    helper_test_variable(
        sum(&[var("idx0", 0, 127) * 4, var("idx2", 0, 3)]) / 4,
        0,
        127,
        "idx0",
    );
}

#[test]
fn test_div_numerator_negative() {
    helper_test_variable(
        (var("idx", 0, 9) * -10) / 11,
        -9,
        0,
        "((((idx*-10)+99)//11)+-9)",
    );
}

#[test]
fn test_div_into_mod() {
    helper_test_variable((var("idx", 0, 16) * 4) % 8 / 4, 0, 1, "(idx%2)");
}

fn helper_test_numeric<F>(f: F)
where
    F: Fn(ArcNode) -> ArcNode,
{
    let (MIN, MAX) = (0, 10);
    for i in MIN..MAX {
        let v = f(num(i));
        assert!(v.min().unwrap() == v.max().unwrap());
        assert!(v.min().unwrap() == f(num(i)).num_val().unwrap());
    }
    for kmin in MIN..MAX {
        for kmax in MIN..MAX {
            if kmin > kmax {
                continue;
            }
            let v = f(var("tmp", kmin, kmax));
            let values: Vec<isize> = (kmin..kmax + 1)
                .map(|i| f(num(i)).num_val().unwrap())
                .collect();
            assert!(v.min().unwrap() <= *values.iter().min().unwrap());
            assert!(v.max().unwrap() <= *values.iter().max().unwrap());
        }
    }
}

#[test]
fn test_mod_4() {
    helper_test_numeric(|x| x % 4);
}

#[test]
fn test_div_4() {
    helper_test_numeric(|x| x / 4);
}

#[test]
fn test_plus_1_div_2() {
    helper_test_numeric(|x| (x + 1) / 2);
}

#[test]
fn test_plus_1_mod_2() {
    helper_test_numeric(|x| (x + 1) % 2);
}

#[test]
fn test_times_2() {
    helper_test_numeric(|x| x * 2);
}

#[test]
fn test_times_2_plus_3() {
    helper_test_numeric(|x| x * 2 + 3);
}

#[test]
fn test_times_2_plus_3_mod_4() {
    helper_test_numeric(|x| (x * 2 + 3) % 4);
}

#[test]
fn test_times_2_plus_3_div_4() {
    helper_test_numeric(|x| (x * 2 + 3) / 4);
}

#[test]
fn test_times_2_plus_3_div_4_mod_4() {
    helper_test_numeric(|x| ((x * 2 + 3) / 4) % 4);
}

#[test]
fn test_vars_simple() {
    let z = num(0);
    let a = var("a", 0, 10);
    let b = var("b", 0, 10);
    let c = var("c", 0, 10);
    assert!(
        z.vars() == z.vars() && z.vars() == vec![],
        "{:?} != []",
        z.vars()
    );
    assert!(
        a.vars() == a.vars() && a.vars() == vec![a.clone()],
        "{:?} != [a]",
        a.vars()
    );
    let m = &a * 3;
    assert!(m.vars() == vec![a.clone()]);
    let s = sum(&[a.clone(), b.clone(), c.clone()]);
    assert!(s.vars() == vec![a.clone(), b.clone(), c.clone()]);
}

#[test]
fn test_vars_compound() {
    let a = var("a", 0, 10);
    let b = var("b", 0, 10);
    let c = var("c", 0, 10);
    let tmp = &a + &b * &c;
    assert!(
        tmp.vars() == vec![a.clone(), b.clone(), c.clone()],
        "\n{:?}\n!=\n{:?}",
        tmp.vars(),
        vec![a.clone(), b.clone(), c.clone()]
    );
}
