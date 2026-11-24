/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use cvxrust::prelude::*;
use nalgebra::DMatrix;

const TOL: f64 = 1e-4;

/// `sum_squares(A x - b)` must be canonicalized as `||r||^2`, not `||r||_2`.
/// The old bug used a plain SOC `||r||_2 <= t`, which minimized the L2 norm
/// instead of its square, causing `solution.value` to be off by a square root.

#[test]
fn test_sum_squares_scalar_constrained() {
    // minimize (x - 3)^2  s.t.  x <= 1
    // True optimum: x^* = 1,  obj^* = (1 - 3)^2 = 4
    // Old bug would report: |1 - 3| = 2
    let x = variable(());
    let residual = x.clone() - constant(3.0);

    let sol = Problem::minimize(sum_squares(&residual))
        .constraint(constraint!(x <= 1.0))
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let eval_sq = sum_squares(&residual).value(&sol).as_scalar().unwrap();
    let eval_norm = norm2(&residual).value(&sol).as_scalar().unwrap();

    assert!(
        (reported - 4.0).abs() < TOL,
        "objective should be 4.0, got {reported}"
    );
    assert!(
        (reported - eval_sq).abs() < TOL,
        "reported obj must equal sum_squares evaluated at solution"
    );
    assert!((eval_norm - 2.0).abs() < TOL);
    assert!(
        (reported - eval_norm).abs() > 0.5,
        "objective must not equal the L2 norm (old bug)"
    );
}

#[test]
fn test_sum_squares_vector_constrained() {
    // minimize ||x - [3, 4]||^2  s.t.  x <= 2
    // True optimum: x^* = [2, 2],  obj^* = (2-3)^2 + (2-4)^2 = 5
    // Old bug would report: sqrt(5) ~= 2.236
    let x = variable(2);
    let residual = x.clone() - constant_vec(vec![3.0, 4.0]);

    let sol = Problem::minimize(sum_squares(&residual))
        .constraint(constraint!(x <= 2.0))
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let eval_sq = sum_squares(&residual).value(&sol).as_scalar().unwrap();
    let eval_norm = norm2(&residual).value(&sol).as_scalar().unwrap();

    assert!(
        (reported - 5.0).abs() < TOL,
        "objective should be 5.0, got {reported}"
    );
    assert!(
        (reported - eval_sq).abs() < TOL,
        "reported obj must equal sum_squares evaluated at solution"
    );
    assert!((eval_norm - 5f64.sqrt()).abs() < TOL);
    assert!(
        (reported - eval_norm).abs() > 0.5,
        "objective must not equal the L2 norm (old bug)"
    );
}

#[test]
fn test_sum_squares_matmul_constrained() {
    // minimize ||A x - b||^2  s.t.  x <= 1
    // A = [[1], [1]] (2 x 1), b = [2, 4], x scalar
    // Unconstrained LS: x^* = 3, obj^* = (3-2)^2 + (3-4)^2 = 2
    // With x <= 1: x^* = 1, residual = [-1, -3], obj^* = 1 + 9 = 10
    // Old bug would report: sqrt(10) ~= 3.162
    let a = constant_matrix(vec![1.0, 1.0], 2, 1);
    let b = constant_vec(vec![2.0, 4.0]);
    let x = variable(());
    let residual = matmul(&a, &x) - &b;

    let sol = Problem::minimize(sum_squares(&residual))
        .constraint(constraint!(x <= 1.0))
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let eval_sq = sum_squares(&residual).value(&sol).as_scalar().unwrap();
    let eval_norm = norm2(&residual).value(&sol).as_scalar().unwrap();

    assert!(
        (reported - 10.0).abs() < TOL,
        "objective should be 10.0, got {reported}"
    );
    assert!(
        (reported - eval_sq).abs() < TOL,
        "reported obj must equal sum_squares evaluated at solution"
    );
    assert!((eval_norm - 10f64.sqrt()).abs() < TOL);
    assert!(
        (reported - eval_norm).abs() > 0.5,
        "objective must not equal the L2 norm (old bug)"
    );
}

#[test]
fn test_sum_squares_constraint_uses_square_not_norm() {
    // maximize x  s.t.  x^2 <= 4, x >= 0
    // True optimum: x^* = 2. Old bug modeled |x| <= 4 and allowed x^* = 4.
    let x = variable(());

    let sol = Problem::maximize(x.clone())
        .subject_to([sum_squares(&x).le(4.0), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 2.0).abs() < TOL);
    assert!((x.value(&sol).as_scalar().unwrap() - 2.0).abs() < TOL);
}

#[test]
fn test_quad_over_lin_uses_denominator() {
    let x = variable(());

    let sol = Problem::maximize(x.clone())
        .subject_to([quad_over_lin(&x, &constant(0.5)).le(2.0), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((solution_value(&sol, &x) - 1.0).abs() < TOL);
}

#[test]
fn test_quad_over_lin_uses_variable_denominator() {
    let x = variable(());
    let y = variable(());

    let sol = Problem::maximize(x.clone())
        .subject_to([quad_over_lin(&x, &y).le(2.0), x.ge(0.0), y.eq(0.5)])
        .solve()
        .expect("problem should solve");

    assert!((solution_value(&sol, &x) - 1.0).abs() < TOL);
}

#[test]
fn test_quad_over_lin_objective_constant_denominator() {
    let x = variable(());

    let sol = Problem::minimize(quad_over_lin(&x, &constant(0.5)))
        .subject_to([x.ge(2.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 8.0).abs() < TOL);
    assert!((solution_value(&sol, &x) - 2.0).abs() < TOL);
}

#[test]
fn test_quad_over_lin_objective_variable_denominator() {
    let x = variable(());
    let y = variable(());

    let sol = Problem::minimize(quad_over_lin(&x, &y))
        .subject_to([x.eq(2.0), y.eq(2.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 2.0).abs() < TOL);
}

#[test]
fn test_power_zero_is_constant_one() {
    let x = variable(());

    let sol = Problem::minimize(power(&x, 0.0))
        .subject_to([x.ge(1.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 1.0).abs() < TOL);
}

#[test]
fn test_power_zero_vector_is_constant_one_elementwise() {
    let x = variable(3);

    let sol = Problem::minimize(sum(&power(&x, 0.0)))
        .subject_to([x.ge(1.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 3.0).abs() < TOL);
}

#[test]
fn test_power_two_is_elementwise() {
    let x = variable(2);

    let sol = Problem::maximize(sum(&x))
        .subject_to([power(&x, 2.0).le(constant_vec(vec![1.0, 4.0])), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 3.0).abs() < TOL);

    if let Array::Dense(x_vals) = x.value(&sol) {
        assert!((x_vals[(0, 0)] - 1.0).abs() < TOL);
        assert!((x_vals[(1, 0)] - 2.0).abs() < TOL);
    } else {
        panic!("expected dense vector solution");
    }
}

#[test]
fn test_extract_element_matrix_constants_use_column_major_order() {
    let x = variable((2, 2));
    let offset = constant_dmatrix(DMatrix::from_row_slice(2, 2, &[0.0, 10.0, 20.0, 30.0]));
    let upper_sq = constant_dmatrix(DMatrix::from_row_slice(2, 2, &[1.0, 4.0, 25.0, 49.0]));

    let sol = Problem::maximize(sum(&x))
        .subject_to([power(&(&x + &offset), 2.0).le(upper_sq)])
        .solve()
        .expect("problem should solve");

    if let Array::Dense(x_vals) = x.value(&sol) {
        assert!((x_vals[(0, 0)] - 1.0).abs() < TOL);
        assert!((x_vals[(0, 1)] - -8.0).abs() < TOL);
        assert!((x_vals[(1, 0)] - -15.0).abs() < TOL);
        assert!((x_vals[(1, 1)] - -23.0).abs() < TOL);
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_sum_axis_constraints_are_not_total_sum() {
    let x = variable((2, 2));

    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(0.0), sum_axis(&x, 0).eq(constant_vec(vec![1.0, 2.0]))])
        .solve()
        .expect("problem should solve");

    let vals = x.value(&sol);
    if let Array::Dense(m) = vals {
        assert!(((m[(0, 0)] + m[(1, 0)]) - 1.0).abs() < TOL);
        assert!(((m[(0, 1)] + m[(1, 1)]) - 2.0).abs() < TOL);
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_sum_axis_one_constraints_are_not_total_sum() {
    let x = variable((2, 2));

    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(0.0), sum_axis(&x, 1).eq(constant_vec(vec![1.0, 2.0]))])
        .solve()
        .expect("problem should solve");

    let vals = x.value(&sol);
    if let Array::Dense(m) = vals {
        assert!(((m[(0, 0)] + m[(0, 1)]) - 1.0).abs() < TOL);
        assert!(((m[(1, 0)] + m[(1, 1)]) - 2.0).abs() < TOL);
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_scalar_affine_broadcasts_in_vector_constraint() {
    let x = variable(2);
    let y = variable(());

    let sol = Problem::maximize(sum(&x))
        .subject_to([x.le(y.clone()), y.le(1.0), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 2.0).abs() < TOL);
    assert!((solution_value(&sol, &y) - 1.0).abs() < TOL);

    if let Array::Dense(x_vals) = x.value(&sol) {
        assert!((x_vals[(0, 0)] - 1.0).abs() < TOL);
        assert!((x_vals[(1, 0)] - 1.0).abs() < TOL);
    } else {
        panic!("expected dense vector solution");
    }
}

#[test]
fn test_constant_vector_times_scalar_affine_broadcasts() {
    let y = variable(());
    let weighted = constant_vec(vec![2.0, 3.0]) * y.clone();

    let sol = Problem::maximize(sum(&weighted))
        .subject_to([y.eq(1.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 5.0).abs() < TOL);

    if let Array::Dense(weighted_vals) = weighted.value(&sol) {
        assert!((weighted_vals[(0, 0)] - 2.0).abs() < TOL);
        assert!((weighted_vals[(1, 0)] - 3.0).abs() < TOL);
    } else {
        panic!("expected dense vector value");
    }
}

#[test]
fn test_norm1_matrix_variable_canonicalizes_flat_aux_constraints() {
    let x = variable((2, 3));

    let sol = Problem::minimize(norm1(&x))
        .subject_to([x.ge(1.0)])
        .solve()
        .expect("matrix norm1 problem should solve");

    assert!((sol.value.unwrap() - 6.0).abs() < TOL);
}

#[test]
fn test_norm_inf_matrix_variable_canonicalizes_flat_aux_constraints() {
    let x = variable((2, 3));

    let sol = Problem::minimize(norm_inf(&x))
        .subject_to([sum(&x).eq(6.0)])
        .solve()
        .expect("matrix norm_inf problem should solve");

    assert!((sol.value.unwrap() - 1.0).abs() < TOL);
}

#[test]
fn test_sum_squares_matrix_argument_flattens_soc_stack() {
    let x = variable((2, 3));

    let sol = Problem::maximize(sum(&x))
        .subject_to([sum_squares(&x).le(1.0)])
        .solve()
        .expect("matrix sum_squares constraint should solve");

    assert!((sol.value.unwrap() - 6.0_f64.sqrt()).abs() < TOL);
}

#[test]
fn test_quad_over_lin_matrix_argument_flattens_soc_stack() {
    let x = variable((2, 3));

    let sol = Problem::maximize(sum(&x))
        .subject_to([quad_over_lin(&x, &constant(1.0)).le(1.0)])
        .solve()
        .expect("matrix quad_over_lin constraint should solve");

    assert!((sol.value.unwrap() - 6.0_f64.sqrt()).abs() < TOL);
}

#[test]
fn test_maximum_minimum_accept_vector_and_column_shapes() {
    let x = variable(3);
    let y = variable((3, 1));

    assert_eq!(
        maximum(vec![x.clone(), y.clone()]).shape(),
        Shape::vector(3)
    );
    assert_eq!(
        minimum(vec![x.clone(), y.clone()]).shape(),
        Shape::vector(3)
    );
    assert_eq!(max2(&x, &y).shape(), Shape::vector(3));
    assert_eq!(min2(&x, &y).shape(), Shape::vector(3));
}

#[test]
fn test_vstack_matrix_constraint_uses_column_major_interleaving() {
    let x = variable((2, 2));
    let y = variable((1, 2));
    let target = constant_dmatrix(DMatrix::from_row_slice(
        3,
        2,
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    ));

    let sol = Problem::minimize(sum(&x))
        .subject_to([vstack(vec![x.clone(), y.clone()]).eq(target)])
        .solve()
        .expect("matrix vstack equality should solve");

    if let Array::Dense(x_vals) = x.value(&sol) {
        assert!((x_vals[(0, 0)] - 1.0).abs() < TOL);
        assert!((x_vals[(1, 0)] - 2.0).abs() < TOL);
        assert!((x_vals[(0, 1)] - 4.0).abs() < TOL);
        assert!((x_vals[(1, 1)] - 5.0).abs() < TOL);
    } else {
        panic!("expected dense matrix solution for x");
    }

    if let Array::Dense(y_vals) = y.value(&sol) {
        assert!((y_vals[(0, 0)] - 3.0).abs() < TOL);
        assert!((y_vals[(0, 1)] - 6.0).abs() < TOL);
    } else {
        panic!("expected dense matrix solution for y");
    }
}

fn solution_value(sol: &Solution, expr: &Expr) -> f64 {
    expr.value(sol).as_scalar().expect("expected scalar")
}
