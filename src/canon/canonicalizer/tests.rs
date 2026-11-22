use std::sync::Arc;

use super::*;
use crate::atoms::{matmul, promote, sum_axis};
use crate::expr::{VariableExt, constant, constant_matrix, variable};

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
    // For objective, should produce quadratic or SOC
    assert!(matches!(result.expr, CanonExpr::Quadratic(_)) || !result.constraints.is_empty());
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
