/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use std::sync::Arc;

use approx::assert_abs_diff_eq;

use super::*;
use crate::atoms::{matmul, promote, quad_form, sum, sum_axis, sum_squares};
use crate::expr::{VariableExt, constant, constant_matrix, constant_vec, variable};
use crate::solver::stuff_problem;

fn sign_attribute_domain_contains(result: &CanonResult, id: ExprId, value: f64) -> bool {
    result.constraints.iter().all(|constraint| {
        let ConeConstraint::NonNeg { a } = constraint else {
            panic!("expected a variable bound");
        };
        let coeff = &a.coeffs[&id];
        let point = DMatrix::from_element(coeff.ncols(), 1, value);
        let offset = a
            .constant
            .clone()
            .reshape_generic(nalgebra::Dyn(a.size()), nalgebra::Dyn(1));
        (csc_to_dense(coeff) * point + offset)
            .iter()
            .all(|&entry| entry >= 0.0)
    })
}

#[test]
fn test_sign_attribute_fresh_domains() {
    for shape in [Shape::scalar(), Shape::vector(3), Shape::matrix(2, 3)] {
        let raw = variable(shape);
        let id = raw.variable_id().unwrap();
        let positive = raw.clone().nonneg();
        let negative = raw.clone().nonpos();
        for (expr, accepts_negative, accepts_positive, bounds) in [
            (raw, true, true, 0),
            (positive.clone(), false, true, 1),
            (negative.clone(), true, false, 1),
            (&positive + &positive, false, true, 1),
            (&positive + &negative, false, false, 2),
            (&negative + &positive, false, false, 2),
        ] {
            // Fresh state must return the complete domain on each call.
            for _ in 0..2 {
                let result = canonicalize(&expr, false, &mut VariableDomains::default());
                assert!(result.aux_vars.is_empty());
                assert_eq!(result.constraints.len(), bounds);
                assert!(sign_attribute_domain_contains(&result, id, 0.0));
                assert_eq!(
                    sign_attribute_domain_contains(&result, id, -1.0),
                    accepts_negative
                );
                assert_eq!(
                    sign_attribute_domain_contains(&result, id, 1.0),
                    accepts_positive
                );
            }
        }
    }
}

#[test]
fn test_sign_attribute_shared_domains_preserve_later_bounds() {
    let raw = variable(());
    let positive = raw.clone().nonneg();
    let negative = raw.nonpos();
    let mut domains = VariableDomains::default();

    let first = canonicalize(&positive, false, &mut domains);
    let repeated = canonicalize(&positive, false, &mut domains);
    let opposite = canonicalize(&negative, false, &mut domains);

    assert_eq!(first.constraints.len(), 1);
    assert!(repeated.constraints.is_empty());
    assert_eq!(opposite.constraints.len(), 1);
    assert!(!sign_attribute_domain_contains(
        &opposite,
        negative.variable_id().unwrap(),
        1.0
    ));
}

#[test]
fn test_canonicalize_variable() {
    let x = variable(5);
    let result = canonicalize(&x, false, &mut VariableDomains::default());
    assert!(result.constraints.is_empty());
    assert!(matches!(result.expr, CanonExpr::Linear(_)));
}

#[test]
fn test_canonicalize_norm2() {
    let x = variable(5);
    let n = Expr::Norm2(Arc::new(x));
    let result = canonicalize(&n, false, &mut VariableDomains::default());
    // Should have 1 SOC constraint + 1 NonNeg (t >= 0), and 1 aux variable
    assert_eq!(result.constraints.len(), 2);
    assert_eq!(result.aux_vars.len(), 1);
}

#[test]
fn test_canonicalize_sum_squares_objective() {
    let x = variable(5);
    let s = Expr::SumSquares(Arc::new(x));
    let result = canonicalize(&s, true, &mut VariableDomains::default());
    assert!(matches!(result.expr, CanonExpr::Quadratic(_)));
    assert!(result.constraints.is_empty());
    assert!(result.aux_vars.is_empty());
}

#[test]
fn test_canonicalize_quad_form_affine_variable_blocks_and_stuffing() {
    let u = variable(2);
    let v = variable(());
    let u_id = u.variable_id().unwrap();
    let v_id = v.variable_id().unwrap();
    let a = constant_matrix(vec![2.0, 1.0, 1.0, 3.0], 2, 2);
    let b = constant_matrix(vec![-1.0, 2.0], 2, 1);
    let offset = constant_vec(vec![1.0, -2.0]);
    let p = constant_matrix(vec![2.0, 1.0, 1.0, 2.0], 2, 2);
    let x = matmul(&a, &u) + matmul(&b, &v) + offset;

    let result = canonicalize(&quad_form(&x, &p), true, &mut VariableDomains::default());
    assert!(result.constraints.is_empty());
    assert!(result.aux_vars.is_empty());
    let CanonExpr::Quadratic(q) = result.expr else {
        panic!("an affine quadratic objective must use the native QP representation");
    };

    // x^T P x = 14 u_1^2 + 34 u_1 u_2 + 26 u_2^2 + 6 u_1 v + 18 u_2 v + 6 v^2
    //           - 6 u_1 - 18 u_2 - 12 v + 6.
    assert_abs_diff_eq!(
        csc_to_dense(&q.quad_coeffs[&(u_id, u_id)]),
        DMatrix::from_row_slice(2, 2, &[14.0, 17.0, 17.0, 26.0]),
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        csc_to_dense(&q.quad_coeffs[&(u_id, v_id)]),
        DMatrix::from_column_slice(2, 1, &[3.0, 9.0]),
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        csc_to_dense(&q.quad_coeffs[&(v_id, u_id)]),
        DMatrix::from_row_slice(1, 2, &[3.0, 9.0]),
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        csc_to_dense(&q.quad_coeffs[&(v_id, v_id)]),
        DMatrix::from_element(1, 1, 6.0),
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        csc_to_dense(&q.linear.coeffs[&u_id]),
        DMatrix::from_row_slice(1, 2, &[-6.0, -18.0]),
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        csc_to_dense(&q.linear.coeffs[&v_id]),
        DMatrix::from_element(1, 1, -12.0),
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(q.constant, 6.0, epsilon = 1e-10);
    assert_abs_diff_eq!(q.linear.constant[(0, 0)], 0.0, epsilon = 1e-10);

    for (variables, expected_p, expected_q) in [
        (
            vec![(u_id, u.shape()), (v_id, v.shape())],
            [28.0, 34.0, 6.0, 0.0, 52.0, 18.0, 0.0, 0.0, 12.0],
            [-6.0, -18.0, -12.0],
        ),
        (
            vec![(v_id, v.shape()), (u_id, u.shape())],
            [12.0, 6.0, 18.0, 0.0, 28.0, 34.0, 0.0, 0.0, 52.0],
            [-12.0, -6.0, -18.0],
        ),
    ] {
        let stuffed = stuff_problem(&q, &[], &variables);
        // Clarabel uses (1/2) x^T P x and stores the upper triangle.
        assert_abs_diff_eq!(
            csc_to_dense(&stuffed.p),
            DMatrix::from_row_slice(3, 3, &expected_p),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(stuffed.q.as_slice(), &expected_q[..], epsilon = 1e-10);
        assert_abs_diff_eq!(stuffed.objective_offset, 6.0, epsilon = 1e-10);
    }
}

#[test]
fn test_canonicalize_quad_form_rectangular_map_with_and_without_offset() {
    let x = variable(3);
    let id = x.variable_id().unwrap();
    let a = constant_matrix(vec![1.0, 0.0, 2.0, -1.0, 0.0, 3.0], 2, 3);
    let p = constant_matrix(vec![2.0, 1.0, 1.0, 2.0], 2, 2);
    let mapped = matmul(&a, &x);
    for (argument, expected_linear, expected_constant) in [
        (mapped.clone(), [0.0, 0.0, 0.0], 0.0),
        (
            mapped + constant_vec(vec![1.0, -2.0]),
            [0.0, 6.0, -18.0],
            6.0,
        ),
    ] {
        let result = canonicalize(
            &quad_form(&argument, &p),
            true,
            &mut VariableDomains::default(),
        );
        assert!(result.constraints.is_empty());
        assert!(result.aux_vars.is_empty());
        let CanonExpr::Quadratic(q) = result.expr else {
            panic!("a rectangular affine map must retain the native quadratic objective");
        };
        assert_abs_diff_eq!(
            csc_to_dense(&q.quad_coeffs[&(id, id)]),
            DMatrix::from_row_slice(3, 3, &[2.0, 3.0, 3.0, 3.0, 6.0, 0.0, 3.0, 0.0, 18.0]),
            epsilon = 1e-10
        );
        let stuffed = stuff_problem(&q, &[], &[(id, x.shape())]);
        assert_abs_diff_eq!(stuffed.q.as_slice(), &expected_linear[..], epsilon = 1e-10);
        assert_abs_diff_eq!(q.constant, expected_constant, epsilon = 1e-10);
    }
}

#[test]
fn test_canonicalize_quad_form_objective_composition_stays_native() {
    let x = variable(());
    let id = x.variable_id().unwrap();
    let convex = quad_form(&x, &constant(3.0));
    let concave = quad_form(&x, &constant(-2.0));

    for (objective, quadratic, linear, offset) in [
        (&convex + 2.0 * &x + constant(7.0), 3.0, 2.0, 7.0),
        (2.0 * &x + &convex, 3.0, 2.0, 0.0),
        (&convex - &concave, 5.0, 0.0, 0.0),
        (-(&concave + 2.0 * &x), 2.0, -2.0, 0.0),
        (-2.0 * (&concave - 3.0 * &x) + &convex, 7.0, 6.0, 0.0),
    ] {
        let result = canonicalize(&objective, true, &mut VariableDomains::default());
        assert!(result.constraints.is_empty());
        assert!(result.aux_vars.is_empty());
        let CanonExpr::Quadratic(q) = result.expr else {
            panic!("addition, negation, and scalar multiplication must retain a QP");
        };
        let stuffed = stuff_problem(&q, &[], &[(id, x.shape())]);
        assert_abs_diff_eq!(
            csc_to_dense(&stuffed.p)[(0, 0)],
            2.0 * quadratic,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(stuffed.q[0], linear, epsilon = 1e-10);
        assert_abs_diff_eq!(stuffed.objective_offset, offset, epsilon = 1e-10);
    }
}

#[test]
fn test_canonicalize_quad_form_constant_objective() {
    let x = constant_vec(vec![1.0, -2.0]);
    for (argument, p, expected) in [
        (
            x.clone(),
            constant_matrix(vec![2.0, 1.0, 1.0, 2.0], 2, 2),
            6.0,
        ),
        (x, constant_matrix(vec![-2.0, -1.0, -1.0, -2.0], 2, 2), -6.0),
        (constant(2.0), constant(3.0), 12.0),
    ] {
        let result = canonicalize(
            &quad_form(&argument, &p),
            true,
            &mut VariableDomains::default(),
        );
        assert!(result.constraints.is_empty());
        assert!(result.aux_vars.is_empty());
        let q = result.expr.into_quadratic();
        assert!(q.quad_coeffs.is_empty());
        assert!(q.linear.coeffs.is_empty());
        assert_abs_diff_eq!(q.constant, expected, epsilon = 1e-10);
    }
}

#[test]
fn test_canonicalize_quad_form_zero_preserves_variable_domains() {
    let x = variable(()).nonneg();
    let id = x.variable_id().unwrap();
    let objective = quad_form(&(&x + constant(2.0)), &constant(0.0));
    for for_objective in [false, true] {
        let result = canonicalize(&objective, for_objective, &mut VariableDomains::default());
        assert!(result.aux_vars.is_empty());
        assert_eq!(result.constraints.len(), 1);
        assert!(!sign_attribute_domain_contains(&result, id, -1.0));
        assert!(sign_attribute_domain_contains(&result, id, 0.0));
        assert!(sign_attribute_domain_contains(&result, id, 1.0));
        let q = result.expr.into_quadratic();
        let stuffed = stuff_problem(&q, &[], &[(id, x.shape())]);
        assert!(stuffed.p.values().iter().all(|v| v.abs() < 1e-10));
        assert_abs_diff_eq!(stuffed.q[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stuffed.objective_offset, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_canonicalize_sum_squares_composed_objectives_stay_native() {
    let x = variable(2);
    let id = x.variable_id().unwrap();
    let residual = &x - constant_vec(vec![1.0, -2.0]);
    for (objective, quadratic, linear, offset) in [
        (
            2.0 * sum_squares(&residual) + sum_squares(&x) - sum(&x) + constant(4.0),
            3.0,
            [-5.0, 7.0],
            14.0,
        ),
        (-(-sum_squares(&residual) + sum(&x)), 1.0, [-3.0, 3.0], 5.0),
    ] {
        let result = canonicalize(&objective, true, &mut VariableDomains::default());
        assert!(result.constraints.is_empty());
        assert!(result.aux_vars.is_empty());
        let CanonExpr::Quadratic(q) = result.expr else {
            panic!("composed sum_squares objectives must retain the native QP representation");
        };
        let stuffed = stuff_problem(&q, &[], &[(id, x.shape())]);
        assert_abs_diff_eq!(
            csc_to_dense(&stuffed.p),
            DMatrix::identity(2, 2) * (2.0 * quadratic),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(stuffed.q.as_slice(), &linear[..], epsilon = 1e-10);
        assert_abs_diff_eq!(stuffed.objective_offset, offset, epsilon = 1e-10);
    }
}

#[test]
fn test_canonicalize_sum_axis_vector_returns_scalar() {
    let x = variable(3);
    let result = canonicalize(&sum_axis(&x, 0), false, &mut VariableDomains::default());

    assert_eq!(result.expr.as_linear().shape, Shape::scalar());
}

#[test]
#[should_panic(expected = "axis 2 out of bounds for shape (2, 3)")]
fn test_canonicalize_sum_axis_invalid_direct_expr_panics() {
    let x = variable((2, 3));
    let _ = canonicalize(
        &Expr::Sum(Arc::new(x), Some(2)),
        false,
        &mut VariableDomains::default(),
    );
}

#[test]
#[should_panic(expected = "cannot reshape size 3 into shape (2, 2)")]
fn test_canonicalize_reshape_size_mismatch_direct_expr_panics() {
    let x = variable(3);
    let expr = Expr::Reshape(Arc::new(x), Shape::matrix(2, 2));
    let _ = canonicalize(&expr, false, &mut VariableDomains::default());
}

#[test]
#[should_panic(expected = "vstack requires matching column counts")]
fn test_canonicalize_vstack_mismatched_columns_direct_expr_panics() {
    let x = variable((2, 3));
    let y = variable((3, 2));
    let expr = Expr::VStack(vec![Arc::new(x), Arc::new(y)]);
    let _ = canonicalize(&expr, false, &mut VariableDomains::default());
}

#[test]
#[should_panic(expected = "hstack requires matching row counts")]
fn test_canonicalize_hstack_mismatched_rows_direct_expr_panics() {
    let x = variable((2, 3));
    let y = variable((3, 3));
    let expr = Expr::HStack(vec![Arc::new(x), Arc::new(y)]);
    let _ = canonicalize(&expr, false, &mut VariableDomains::default());
}

#[test]
#[should_panic(expected = "maximum requires at least one expression")]
fn test_canonicalize_maximum_empty_direct_expr_panics() {
    let _ = canonicalize(
        &Expr::Maximum(Vec::new()),
        false,
        &mut VariableDomains::default(),
    );
}

#[test]
fn test_canonicalize_maximum_direct_expr_broadcasts_scalar_to_vector() {
    let x = variable(3);
    let y = variable(());
    let expr = Expr::Maximum(vec![Arc::new(x), Arc::new(y)]);
    let result = canonicalize(&expr, false, &mut VariableDomains::default());

    assert_eq!(result.expr.as_linear().shape, Shape::vector(3));
}

#[test]
#[should_panic(expected = "cannot broadcast shapes (2,) and (3,)")]
fn test_canonicalize_maximum_direct_expr_incompatible_shapes_panics() {
    let x = variable(2);
    let y = variable(3);
    let expr = Expr::Maximum(vec![Arc::new(x), Arc::new(y)]);
    let _ = canonicalize(&expr, false, &mut VariableDomains::default());
}

#[test]
#[should_panic(expected = "minimum requires at least one expression")]
fn test_canonicalize_minimum_empty_direct_expr_panics() {
    let _ = canonicalize(
        &Expr::Minimum(Vec::new()),
        false,
        &mut VariableDomains::default(),
    );
}

#[test]
fn test_canonicalize_minimum_direct_expr_broadcasts_scalar_to_vector() {
    let x = variable(3);
    let y = variable(());
    let expr = Expr::Minimum(vec![Arc::new(x), Arc::new(y)]);
    let result = canonicalize(&expr, false, &mut VariableDomains::default());

    assert_eq!(result.expr.as_linear().shape, Shape::vector(3));
}

#[test]
#[should_panic(expected = "cannot broadcast shapes (2,) and (3,)")]
fn test_canonicalize_minimum_direct_expr_incompatible_shapes_panics() {
    let x = variable(2);
    let y = variable(3);
    let expr = Expr::Minimum(vec![Arc::new(x), Arc::new(y)]);
    let _ = canonicalize(&expr, false, &mut VariableDomains::default());
}

#[test]
fn test_canonicalize_matmul_preserves_vector_result_shape() {
    let a = constant_matrix(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 2, 3);
    let x = variable(3);
    let result = canonicalize(&matmul(&a, &x), false, &mut VariableDomains::default());

    assert_eq!(result.expr.as_linear().shape, Shape::vector(2));
}

#[test]
fn test_canonicalize_matmul_preserves_column_matrix_result_shape() {
    let a = constant_matrix(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 2, 3);
    let x = variable((3, 1));
    let result = canonicalize(&matmul(&a, &x), false, &mut VariableDomains::default());

    assert_eq!(result.expr.as_linear().shape, Shape::matrix(2, 1));
}

#[test]
fn test_canonicalize_mul_sees_promoted_scalar_constant_privately() {
    let x = variable((2, 2));
    let x_id = x.variable_id().unwrap();
    let promoted = promote(&constant(3.0), (2, 2));
    assert!(promoted.constant_value().is_none());

    let result = canonicalize(&(promoted * x), false, &mut VariableDomains::default());
    let lin = result.expr.as_linear();
    let coeff = csc_to_dense(&lin.coeffs[&x_id]);

    assert_eq!(lin.shape, Shape::matrix(2, 2));
    assert_eq!(coeff, DMatrix::identity(4, 4) * 3.0);
    assert!(lin.constant.iter().all(|v| v.abs() < 1e-10));
}

#[test]
#[should_panic(expected = "cannot canonicalize product of two non-constant expressions")]
fn test_canonicalize_variable_product_panics() {
    let x = variable(2);
    let y = variable(2);
    let _ = canonicalize(&(&x * &y), false, &mut VariableDomains::default());
}

#[test]
#[should_panic(expected = "cannot matrix-multiply shapes (2, 3) and (2,)")]
fn test_canonicalize_invalid_matmul_shape_panics() {
    let a = constant_matrix(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 2, 3);
    let x = variable(2);
    let _ = canonicalize(
        &Expr::MatMul(Arc::new(a), Arc::new(x)),
        false,
        &mut VariableDomains::default(),
    );
}

#[test]
#[should_panic(expected = "cannot canonicalize matrix product of two non-constant expressions")]
fn test_canonicalize_variable_matmul_panics() {
    let x = variable((2, 2));
    let y = variable((2, 2));
    let _ = canonicalize(&matmul(&x, &y), false, &mut VariableDomains::default());
}
