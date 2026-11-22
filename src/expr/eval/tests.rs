/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use super::*;
use crate::atoms::{index, indexc, select, slice, slicec};
use crate::expr::{constant, variable};
use crate::prelude::*;
use nalgebra::DMatrix;
use std::collections::HashMap;

/// Simple test context using a HashMap of variable values.
struct TestCtx(HashMap<ExprId, Array>);

impl Evaluable for TestCtx {
    fn get_variable_value(&self, id: ExprId) -> Option<&Array> {
        self.0.get(&id)
    }
}

fn make_var_scalar(val: f64) -> (Expr, TestCtx) {
    let x = variable(());
    let id = x.variable_id().unwrap();
    let mut map = HashMap::new();
    map.insert(id, Array::Scalar(val));
    (x, TestCtx(map))
}

fn make_var_vec(vals: Vec<f64>) -> (Expr, TestCtx) {
    let n = vals.len();
    let x = variable(n);
    let id = x.variable_id().unwrap();
    let mut map = HashMap::new();
    map.insert(id, Array::from_vec(vals));
    (x, TestCtx(map))
}

#[test]
fn test_eval_variable() {
    let (x, ctx) = make_var_scalar(3.0);
    assert_eq!(x.value(&ctx)[(0, 0)], 3.0);
}

#[test]
fn test_eval_constant() {
    let c = constant(5.0);
    let (_, ctx) = make_var_scalar(0.0);
    assert_eq!(c.value(&ctx)[(0, 0)], 5.0);
}

#[test]
fn test_eval_add_scalars() {
    let (x, ctx) = make_var_scalar(3.0);
    let expr = x.clone() + constant(2.0);
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 5.0).abs() < 1e-10);
}

#[test]
fn test_eval_neg() {
    let (x, ctx) = make_var_scalar(3.0);
    let expr = -&x;
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v + 3.0).abs() < 1e-10);
}

#[test]
fn test_eval_mul_scalar() {
    let (x, ctx) = make_var_scalar(4.0);
    let expr = &x * 2.5;
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 10.0).abs() < 1e-10);
}

#[test]
fn test_eval_promote_scalar_to_matrix() {
    let (x, ctx) = make_var_scalar(2.0);
    let value = promote(&x, (2, 3)).value(&ctx);

    if let Array::Dense(m) = value {
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert!(m.iter().all(|v| (*v - 2.0).abs() < 1e-10));
    } else {
        panic!("expected dense promoted value");
    }
}

#[test]
fn test_eval_scalar_like_broadcast_add_sub_mul() {
    let matrix = constant_dmatrix(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
    let row_scalar = constant_vec(vec![10.0]);
    let matrix_scalar = constant_dmatrix(DMatrix::from_row_slice(1, 1, &[2.0]));
    let (_, ctx) = make_var_scalar(0.0);

    let add_value = (&row_scalar + &matrix).value(&ctx);
    let sub_value = (&matrix - &row_scalar).value(&ctx);
    let mul_value = (&matrix_scalar * &matrix).value(&ctx);

    if let (Array::Dense(add), Array::Dense(sub), Array::Dense(mul)) =
        (add_value, sub_value, mul_value)
    {
        assert_eq!(add[(0, 0)], 11.0);
        assert_eq!(add[(1, 1)], 14.0);
        assert_eq!(sub[(0, 0)], -9.0);
        assert_eq!(sub[(1, 1)], -6.0);
        assert_eq!(mul[(0, 0)], 2.0);
        assert_eq!(mul[(1, 1)], 8.0);
    } else {
        panic!("expected dense broadcast values");
    }
}

#[test]
fn test_eval_row_broadcast_add_sub_mul() {
    let matrix = constant_dmatrix(DMatrix::from_row_slice(
        2,
        3,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let row = constant_dmatrix(DMatrix::from_row_slice(1, 3, &[10.0, 20.0, 30.0]));
    let (_, ctx) = make_var_scalar(0.0);

    let add_value = (&row + &matrix).value(&ctx);
    let sub_value = (&matrix - &row).value(&ctx);
    let mul_value = (&row * &matrix).value(&ctx);

    if let (Array::Dense(add), Array::Dense(sub), Array::Dense(mul)) =
        (add_value, sub_value, mul_value)
    {
        assert_eq!(add[(0, 0)], 11.0);
        assert_eq!(add[(1, 0)], 14.0);
        assert_eq!(add[(0, 2)], 33.0);
        assert_eq!(sub[(0, 1)], -18.0);
        assert_eq!(sub[(1, 2)], -24.0);
        assert_eq!(mul[(0, 0)], 10.0);
        assert_eq!(mul[(1, 1)], 100.0);
        assert_eq!(mul[(0, 2)], 90.0);
    } else {
        panic!("expected dense row broadcast values");
    }
}

#[test]
fn test_eval_column_broadcast_add_sub_mul() {
    let matrix = constant_dmatrix(DMatrix::from_row_slice(
        2,
        3,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let col = constant_dmatrix(DMatrix::from_row_slice(2, 1, &[10.0, 20.0]));
    let (_, ctx) = make_var_scalar(0.0);

    let add_value = (&matrix + &col).value(&ctx);
    let sub_value = (&matrix - &col).value(&ctx);
    let mul_value = (&col * &matrix).value(&ctx);

    if let (Array::Dense(add), Array::Dense(sub), Array::Dense(mul)) =
        (add_value, sub_value, mul_value)
    {
        assert_eq!(add[(0, 0)], 11.0);
        assert_eq!(add[(1, 0)], 24.0);
        assert_eq!(add[(0, 2)], 13.0);
        assert_eq!(sub[(0, 1)], -8.0);
        assert_eq!(sub[(1, 2)], -14.0);
        assert_eq!(mul[(0, 0)], 10.0);
        assert_eq!(mul[(1, 1)], 100.0);
        assert_eq!(mul[(1, 2)], 120.0);
    } else {
        panic!("expected dense column broadcast values");
    }
}

#[test]
#[should_panic(expected = "cannot broadcast shapes (2, 3) and (3, 2)")]
fn test_eval_incompatible_broadcast_panics_at_construction() {
    let a = constant_dmatrix(DMatrix::zeros(2, 3));
    let b = constant_dmatrix(DMatrix::zeros(3, 2));

    let _ = &a + &b;
}

#[test]
fn test_eval_norm2() {
    let (x, ctx) = make_var_vec(vec![3.0, 4.0]);
    let expr = norm2(&x);
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 5.0).abs() < 1e-10);
}

#[test]
fn test_eval_norm1() {
    let (x, ctx) = make_var_vec(vec![-1.0, 2.0, -3.0]);
    let expr = norm1(&x);
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 6.0).abs() < 1e-10);
}

#[test]
fn test_eval_sum() {
    let (x, ctx) = make_var_vec(vec![1.0, 2.0, 3.0]);
    let expr = sum(&x);
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 6.0).abs() < 1e-10);
}

#[test]
fn test_eval_matrix_first_axis_slice() {
    let x = constant_dmatrix(DMatrix::from_row_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let (_, ctx) = make_var_scalar(0.0);
    let value = slice(&x, 0, 2).value(&ctx);

    if let Array::Dense(m) = value {
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 4);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 5.0);
        assert_eq!(m[(0, 3)], 4.0);
        assert_eq!(m[(1, 3)], 8.0);
    } else {
        panic!("expected dense matrix");
    }
}

#[test]
fn test_eval_matrix_first_axis_index() {
    let x = constant_dmatrix(DMatrix::from_row_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let (_, ctx) = make_var_scalar(0.0);
    let value = index(&x, 1).value(&ctx);

    if let Array::Dense(m) = value {
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 1);
        assert_eq!(m[(0, 0)], 5.0);
        assert_eq!(m[(1, 0)], 6.0);
        assert_eq!(m[(2, 0)], 7.0);
        assert_eq!(m[(3, 0)], 8.0);
    } else {
        panic!("expected dense vector");
    }
}

#[test]
fn test_eval_matrix_column_index() {
    let x = constant_dmatrix(DMatrix::from_row_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let (_, ctx) = make_var_scalar(0.0);
    let value = indexc(&x, 2).value(&ctx);

    if let Array::Dense(m) = value {
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 1);
        assert_eq!(m[(0, 0)], 3.0);
        assert_eq!(m[(1, 0)], 7.0);
        assert_eq!(m[(2, 0)], 11.0);
    } else {
        panic!("expected dense vector");
    }
}

#[test]
fn test_eval_matrix_scalar_select() {
    let x = constant_dmatrix(DMatrix::from_row_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let (_, ctx) = make_var_scalar(0.0);
    let value = select(&x, AxisIndex::Index(1), AxisIndex::Index(2)).value(&ctx);

    assert_eq!(value.as_scalar(), Some(7.0));
}

#[test]
fn test_eval_matrix_rectangular_select() {
    let x = constant_dmatrix(DMatrix::from_row_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let (_, ctx) = make_var_scalar(0.0);
    let value = select(&x, AxisIndex::Slice(0, 2), AxisIndex::Slice(1, 3)).value(&ctx);

    if let Array::Dense(m) = value {
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(1, 0)], 6.0);
        assert_eq!(m[(0, 1)], 3.0);
        assert_eq!(m[(1, 1)], 7.0);
    } else {
        panic!("expected dense matrix");
    }
}

#[test]
fn test_eval_matrix_column_slice_alias() {
    let x = constant_dmatrix(DMatrix::from_row_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let (_, ctx) = make_var_scalar(0.0);
    let value = slicec(&x, 1, 3).value(&ctx);

    if let Array::Dense(m) = value {
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(1, 0)], 6.0);
        assert_eq!(m[(2, 0)], 10.0);
        assert_eq!(m[(0, 1)], 3.0);
        assert_eq!(m[(1, 1)], 7.0);
        assert_eq!(m[(2, 1)], 11.0);
    } else {
        panic!("expected dense matrix");
    }
}

#[test]
fn test_eval_sum_squares() {
    let (x, ctx) = make_var_vec(vec![1.0, 2.0, 3.0]);
    let expr = sum_squares(&x);
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 14.0).abs() < 1e-10);
}

#[test]
fn test_eval_abs() {
    let (x, ctx) = make_var_scalar(-5.0);
    let expr = abs(&x);
    let v = expr.value(&ctx).as_scalar().unwrap();
    assert!((v - 5.0).abs() < 1e-10);
}

#[test]
fn test_eval_exp_log() {
    let (x, ctx) = make_var_scalar(1.0);
    let e = exp(&x);
    let ev = e.value(&ctx).as_scalar().unwrap();
    assert!((ev - std::f64::consts::E).abs() < 1e-10);
    let l = log(&x);
    let lv = l.value(&ctx).as_scalar().unwrap();
    assert!(lv.abs() < 1e-10);
}

#[test]
fn test_eval_with_solution() {
    let x = variable(());
    let obj = &x * 2.0;
    let solution = Problem::minimize(x.clone())
        .subject_to([x.ge(3.0)])
        .solve()
        .unwrap();
    // The variable x should be ~3.0, so 2*x ~= 6.0
    let v = obj.value(&solution).as_scalar().unwrap();
    assert!((v - 6.0).abs() < 1e-4);
}
