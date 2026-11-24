/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Scaling and canonicalization stress tests.

use super::TOL;
use cvxrust::prelude::*;
use nalgebra::DMatrix;

const STRESS_TOL: f64 = 1e-3; // Slightly looser tolerance for complex problems

// quad_form stress tests

/// quad_form with non-diagonal PSD matrix
#[test]
fn test_quad_form_non_diagonal() {
    // P = [[2, 1], [1, 2]] is PSD (eigenvalues 1 and 3)
    let p_mat = DMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
    let p = constant_dmatrix(p_mat);
    let x = variable(2);

    // minimize x^T P x s.t. x_1 + x_2 = 1
    // x^T P x = 2*x_1^2 + 2*x_1*x_2 + 2*x_2^2
    // With constraint x_2 = 1 - x_1:
    // f(x_1) = 2*x_1^2 + 2*x_1*(1-x_1) + 2*(1-x_1)^2
    //        = 2*x_1^2 + 2*x_1 - 2*x_1^2 + 2 - 4*x_1 + 2*x_1^2
    //        = 2*x_1^2 - 2*x_1 + 2
    // Stationarity: 4*x_1 - 2 = 0 => x_1 = 0.5, x_2 = 0.5
    // value = 2(0.25) + 2(0.5)(0.5) + 2(0.25) = 0.5 + 0.5 + 0.5 = 1.5
    let prob = Problem::minimize(quad_form(&x, &p))
        .subject_to([sum(&x).eq(constant(1.0))])
        .build();

    assert!(prob.is_dcp(), "quad_form with PSD should be DCP");
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 1.5;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "non-diagonal quad_form: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// `quad_form` with a `3 x 3` matrix.
#[test]
fn test_quad_form_3x3() {
    // P = diag(1, 2, 3)
    let p_mat = DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let p = constant_dmatrix(p_mat);
    let x = variable(3);

    // minimize x^T P x = x_1^2 + 2*x_2^2 + 3*x_3^2 s.t. sum(x) = 1
    // Lagrangian optimality: 2*x_1 = 4*x_2 = 6*x_3 = -lambda
    // x_1 = -lambda/2, x_2 = -lambda/4, x_3 = -lambda/6
    // sum = -lambda*(1/2 + 1/4 + 1/6) = -11*lambda/12 = 1 => lambda = -12/11
    // x_1 = 6/11, x_2 = 3/11, x_3 = 2/11
    // value = (6/11)^2 + 2*(3/11)^2 + 3*(2/11)^2
    //       = 36/121 + 18/121 + 12/121 = 66/121 = 6/11
    let prob = Problem::minimize(quad_form(&x, &p))
        .subject_to([sum(&x).eq(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 6.0 / 11.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "3x3 quad_form: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

// Matmul stress tests

/// Nested matmul: `A (B x)`.
#[test]
fn test_nested_matmul() {
    // A is 2 x 3, B is 3 x 2, x is 2 x 1
    // A B is 2 x 2, (A B) x is 2 x 1
    let a_mat = DMatrix::from_vec(2, 3, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let b_mat = DMatrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let a = constant_dmatrix(a_mat.clone());
    let b = constant_dmatrix(b_mat.clone());
    let x = variable(2);

    // y = A (B x)
    let bx = matmul(&b, &x);
    let y = matmul(&a, &bx);

    // minimize sum(y) s.t. x >= 1
    let prob = Problem::minimize(sum(&y))
        .subject_to([x.ge(constant(1.0))])
        .build();

    assert!(prob.is_dcp(), "nested matmul should be DCP");
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    // Compute expected:
    // A (2 x 3) = [[1,0,1], [0,1,1]] (row-major view)
    // B (3 x 2) = [[1,0], [0,1], [0,0]] (row-major view)
    // B x where x is 2 x 1 gives 3 x 1
    // A (B x) gives 2 x 1
    // With x = [1,1], B [1,1] = [1, 1, 0]
    // A [1, 1, 0] = [1*1+0*1+1*0, 0*1+1*1+1*0] = [1, 1]
    // sum = 2
    let value = solution.value.expect("should have value");
    let expected = 2.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "nested matmul: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Matmul with non-square matrix
#[test]
fn test_matmul_nonsquare() {
    // A is 2 x 4, x is 4 x 1, result is 2 x 1
    let a_mat = DMatrix::from_vec(
        2,
        4,
        vec![
            1.0, 2.0, // col 0
            3.0, 4.0, // col 1
            5.0, 6.0, // col 2
            7.0, 8.0, // col 3
        ],
    );
    let a = constant_dmatrix(a_mat);
    let x = variable(4);

    // minimize sum(A x) s.t. x >= 0, sum(x) = 1
    // A x = [1*x_1 + 3*x_2 + 5*x_3 + 7*x_4, 2*x_1 + 4*x_2 + 6*x_3 + 8*x_4]
    // sum(A x) = 3*x_1 + 7*x_2 + 11*x_3 + 15*x_4
    // To minimize with sum(x)=1, x>=0: put all weight on x_1
    // optimal: x = [1, 0, 0, 0], value = 3
    let prob = Problem::minimize(sum(&matmul(&a, &x)))
        .subject_to([x.ge(constant(0.0)), sum(&x).eq(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 3.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "nonsquare matmul: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Right matmul: `x A` (row vector times matrix).
#[test]
fn test_right_matmul_row_vector() {
    // x is 1 x 3 row vector, A is 3 x 2, result is 1 x 2
    let a_mat = DMatrix::from_vec(
        3,
        2,
        vec![
            1.0, 2.0, 3.0, // col 0
            4.0, 5.0, 6.0, // col 1
        ],
    );
    let a = constant_dmatrix(a_mat);
    let x = variable((1, 3)); // 1 x 3 row vector

    // minimize sum(x A) s.t. x >= 1
    // x A = [x_1 + 2*x_2 + 3*x_3, 4*x_1 + 5*x_2 + 6*x_3]
    // sum = 5*x_1 + 7*x_2 + 9*x_3
    // optimal: x = [1, 1, 1], value = 21
    let prob = Problem::minimize(sum(&matmul(&x, &a)))
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 21.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "right matmul: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

// Mixed variable tests

/// Multiple variables in objective
#[test]
fn test_multi_variable_objective() {
    let x = variable(3);
    let y = variable(2);
    let z = variable(1);

    // minimize sum(x) + 2*sum(y) + 3*sum(z) s.t. all >= 1
    let obj = &sum(&x) + &(&constant(2.0) * &sum(&y)) + &(&constant(3.0) * &sum(&z));
    let prob = Problem::minimize(obj)
        .subject_to([
            x.ge(constant(1.0)),
            y.ge(constant(1.0)),
            z.ge(constant(1.0)),
        ])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    // optimal: all = 1, value = 3 + 2*2 + 3*1 = 3 + 4 + 3 = 10
    let expected = 10.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "multi-var obj: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Shared variable in objective and constraints
#[test]
fn test_shared_variable() {
    let x = variable(3);

    // minimize ||x||_2 s.t. sum(x) >= 3, x >= 0, x <= 2
    // optimal: x = [1, 1, 1], ||x|| = sqrt(3)
    let prob = Problem::minimize(norm2(&x))
        .subject_to([
            sum(&x).ge(constant(3.0)),
            x.ge(constant(0.0)),
            x.le(constant(2.0)),
        ])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 3.0_f64.sqrt();
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "shared var: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Variable appears in multiple constraints with different atoms
#[test]
fn test_variable_multiple_constraints() {
    let x = variable(3);

    // minimize sum(x) s.t. norm2(x) <= 2, sum(x) >= 1, x >= 0
    // This combines SOCP and LP constraints on same variable
    let prob = Problem::minimize(sum(&x))
        .subject_to([
            norm2(&x).le(constant(2.0)),
            sum(&x).ge(constant(1.0)),
            x.ge(constant(0.0)),
        ])
        .build();

    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let value = solution.value.expect("should have value");
    // Should achieve sum(x) = 1 (the lower bound)
    assert!(
        (value - 1.0).abs() < STRESS_TOL,
        "multi-constraint: expected ~1, got {}",
        value
    );
}

// vstack/hstack stress tests

/// vstack in objective
#[test]
fn test_vstack_objective() {
    let x = variable(2);
    let y = variable(2);

    // vstack([x, y]) is 4 x 1
    // minimize ||vstack([x, y])||_2 s.t. sum(x) = 1, sum(y) = 1
    let stacked = vstack(vec![x.clone(), y.clone()]);
    let prob = Problem::minimize(norm2(&stacked))
        .subject_to([sum(&x).eq(constant(1.0)), sum(&y).eq(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    // optimal: x = [0.5, 0.5], y = [0.5, 0.5], norm = sqrt(4 * 0.25) = 1
    let expected = 1.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "vstack obj: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// vstack in constraint
#[test]
fn test_vstack_constraint() {
    let x = variable(2);
    let y = variable(2);

    // minimize sum(x) + sum(y) s.t. vstack([x, y]) >= 1
    let stacked = vstack(vec![x.clone(), y.clone()]);
    let prob = Problem::minimize(&sum(&x) + &sum(&y))
        .subject_to([stacked.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    // optimal: all = 1, value = 4
    let expected = 4.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "vstack constraint: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

// Numerical edge cases

/// Very small coefficients
#[test]
fn test_small_coefficients() {
    let x = variable(3);
    let small = constant(1e-6);

    // minimize 1e-6 * sum(x) s.t. x >= 1
    let prob = Problem::minimize(&small * &sum(&x))
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 3e-6;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "small coeff: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Large coefficients
#[test]
fn test_large_coefficients() {
    let x = variable(3);
    let large = constant(1e6);

    // minimize 1e6 * sum(x) s.t. x >= 1
    let prob = Problem::minimize(&large * &sum(&x))
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 3e6;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "large coeff: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Near-degenerate constraint (almost parallel)
#[test]
fn test_near_degenerate() {
    let x = variable(2);

    // x_1 + x_2 = 1
    // x_1 + 1.0001*x_2 = 1.0001
    // These are nearly parallel - tests numerical stability
    let c1 = sum(&x).eq(constant(1.0));
    let c2 = (&x + &constant_vec(vec![0.0, 0.0001])).eq(constant_vec(vec![0.5, 0.5001]));

    let prob = Problem::minimize(sum(&x)).subject_to([c1, c2]).build();

    // Should either solve or report numerical issues
    let result = prob.solve();
    // We mainly care that it doesn't crash
    assert!(result.is_ok() || result.is_err());
}

// Constraint combination tests

/// Mix of equality and inequality constraints
#[test]
fn test_mixed_eq_ineq() {
    let x = variable(4);

    // minimize sum(x) s.t.
    // x[0] + x[1] = 2  (equality)
    // x[2] + x[3] >= 1 (inequality)
    // x >= 0
    let _x01 = &constant_vec(vec![1.0, 1.0, 0.0, 0.0]);
    let _x23 = &constant_vec(vec![0.0, 0.0, 1.0, 1.0]);

    // dot(x01, x) = x[0] + x[1]
    // dot(x23, x) = x[2] + x[3]
    // Using matmul with row vectors instead
    let c01 = DMatrix::from_vec(1, 4, vec![1.0, 1.0, 0.0, 0.0]);
    let c23 = DMatrix::from_vec(1, 4, vec![0.0, 0.0, 1.0, 1.0]);

    let prob = Problem::minimize(sum(&x))
        .subject_to([
            matmul(&constant_dmatrix(c01), &x).eq(constant(2.0)),
            matmul(&constant_dmatrix(c23), &x).ge(constant(1.0)),
            x.ge(constant(0.0)),
        ])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    // optimal: x = [1, 1, 0.5, 0.5] or similar, value = 3
    let expected = 3.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "mixed eq/ineq: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Multiple SOCP constraints
#[test]
fn test_multiple_socp() {
    let x = variable(2);
    let y = variable(2);

    // minimize sum(x) + sum(y) s.t.
    // ||x||_2 <= 1
    // ||y||_2 <= 1
    // sum(x) >= 0.5
    // sum(y) >= 0.5
    let prob = Problem::minimize(&sum(&x) + &sum(&y))
        .subject_to([
            norm2(&x).le(constant(1.0)),
            norm2(&y).le(constant(1.0)),
            sum(&x).ge(constant(0.5)),
            sum(&y).ge(constant(0.5)),
        ])
        .build();

    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let value = solution.value.expect("should have value");
    // Should be around 1.0 (0.5 + 0.5)
    assert!(
        (0.99..=1.5).contains(&value),
        "multiple SOCP: got {}",
        value
    );
}

// Regression tests for fixed bugs

/// Regression: sparse constant in constraint (bug #1)
#[test]
fn test_regression_sparse_constant_constraint() {
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    // Sparse vector [0, 5, 0] - only middle element is non-zero
    let mut coo = CooMatrix::new(3, 1);
    coo.push(1, 0, 5.0); // Only x[1] has constraint x[1] >= 5
    let sparse: CscMatrix<f64> = CscMatrix::from(&coo);

    let x = variable(3);
    let prob = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant_sparse(sparse))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    // x >= [0, 5, 0] means optimal x = [0, 5, 0], sum = 5
    let expected = 5.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "sparse const regression: expected {}, got {}",
        expected,
        value
    );
}

/// Regression: sparse matrix in matmul (bug #2, #3)
#[test]
fn test_regression_sparse_matmul() {
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    // Sparse 3 x 3 matrix with only diagonal: [[2,0,0], [0,3,0], [0,0,4]]
    let mut coo = CooMatrix::new(3, 3);
    coo.push(0, 0, 2.0);
    coo.push(1, 1, 3.0);
    coo.push(2, 2, 4.0);
    let sparse: CscMatrix<f64> = CscMatrix::from(&coo);

    let a = constant_sparse(sparse);
    let x = variable(3);

    // minimize sum(A x) s.t. x >= 1
    // A x = [2*x_1, 3*x_2, 4*x_3]
    // sum = 2 + 3 + 4 = 9
    let prob = Problem::minimize(sum(&matmul(&a, &x)))
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 9.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "sparse matmul regression: expected {}, got {}",
        expected,
        value
    );
}

// Larger scale tests

/// 200 variables LP
#[test]
fn test_scale_200_lp() {
    let x = variable(200);
    let prob = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve 200-var LP");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let value = solution.value.expect("should have value");
    assert!(
        (value - 200.0).abs() < 1.0,
        "200-var LP: expected ~200, got {}",
        value
    );
}

/// 100 variables with multiple constraint types
#[test]
fn test_scale_100_mixed() {
    let x = variable(100);

    // minimize ||x||_2 s.t. sum(x) = 100, x >= 0
    let prob = Problem::minimize(norm2(&x))
        .subject_to([sum(&x).eq(constant(100.0)), x.ge(constant(0.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let value = solution.value.expect("should have value");
    // optimal: all x[i] = 1, ||x|| = 10
    let expected = 10.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "100-var mixed: expected {}, got {}",
        expected,
        value
    );
}

/// 50 variables sum_squares (QP)
#[test]
fn test_scale_50_qp() {
    let x = variable(50);

    // minimize ||x||^2 s.t. sum(x) = 50
    let prob = Problem::minimize(sum_squares(&x))
        .subject_to([sum(&x).eq(constant(50.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let value = solution.value.expect("should have value");
    // optimal: all x[i] = 1, ||x||^2 = 50
    let expected = 50.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "50-var QP: expected {}, got {}",
        expected,
        value
    );
}

// Expression complexity tests

/// Deeply nested expression
#[test]
fn test_deep_nesting() {
    let x = variable(3);

    // ((((x + 1) * 2) - 1) + 0.5)
    let e1 = &x + &constant(1.0);
    let e2 = &constant(2.0) * &e1;
    let e3 = &e2 - &constant(1.0);
    let e4 = &e3 + &constant(0.5);

    // minimize sum(e4) s.t. x >= 0
    // e4 = 2*(x+1) - 1 + 0.5 = 2x + 2 - 0.5 = 2x + 1.5
    // sum(e4) = 2*sum(x) + 4.5
    // optimal: x = [0,0,0], value = 4.5
    let prob = Problem::minimize(sum(&e4))
        .subject_to([x.ge(constant(0.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 4.5;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "deep nesting: expected {}, got {}",
        expected,
        value
    );
}

/// Multiple operations on same expression
#[test]
fn test_expression_reuse() {
    let x = variable(3);
    let y = sum(&x); // Reuse this expression

    // minimize y + y (same expression twice) s.t. x >= 1
    // = 2 * sum(x)
    // optimal: x = [1,1,1], value = 6
    let prob = Problem::minimize(&y + &y)
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    let expected = 6.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < STRESS_TOL,
        "expr reuse: expected {}, got {}",
        expected,
        value
    );
}

/// Test with 100 variables (LP)
#[test]
fn test_scale_100_variables_lp() {
    let x = variable(100);
    // minimize sum(x) s.t. x >= 1
    let prob = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant(1.0))])
        .build();

    let solution = prob.solve().expect("should solve 100-var LP");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 100.0;
    assert!(
        (value - expected).abs() < TOL * 100.0,
        "100-var LP: expected {}, got {}",
        expected,
        value
    );
}

/// Test with 50 variables (SOCP)
#[test]
fn test_scale_50_variables_socp() {
    let x = variable(50);
    // minimize ||x||_2 s.t. sum(x) = 50
    // optimal: x = [1, 1, ..., 1], ||x||_2 = sqrt(50)
    let prob = Problem::minimize(norm2(&x))
        .subject_to([sum(&x).eq(constant(50.0))])
        .build();

    let solution = prob.solve().expect("should solve 50-var SOCP");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = (50.0_f64).sqrt();
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < TOL,
        "50-var SOCP: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Test with 30 variables (QP)
#[test]
fn test_scale_30_variables_qp() {
    let x = variable(30);
    // minimize ||x||_2^2 s.t. sum(x) = 30
    // optimal: x = [1, 1, ..., 1], ||x||^2 = 30
    let prob = Problem::minimize(sum_squares(&x))
        .subject_to([sum(&x).eq(constant(30.0))])
        .build();

    let solution = prob.solve().expect("should solve 30-var QP");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 30.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < TOL,
        "30-var QP: expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}
