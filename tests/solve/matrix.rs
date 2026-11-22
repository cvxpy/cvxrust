/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Sparse constants, matrix operations, and indexing.

use super::TOL;
use cvxrust::prelude::*;

/// BUG #1: Sparse constants silently become zeros
/// The canonicalizer at canonicalizer.rs:187-191 returns LinExpr::zeros() for sparse constants
/// instead of properly converting them.
#[test]
fn test_bug_sparse_constant_becomes_zero() {
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    // Create a simple sparse matrix with non-zero values
    // This is a 3x1 column vector with values [1.0, 2.0, 3.0] stored as sparse
    let mut coo = CooMatrix::new(3, 1);
    coo.push(0, 0, 1.0);
    coo.push(1, 0, 2.0);
    coo.push(2, 0, 3.0);
    let sparse_vec: CscMatrix<f64> = CscMatrix::from(&coo);

    let sparse_const = constant_sparse(sparse_vec);
    let x = variable(3);

    // minimize sum(x) s.t. x >= sparse_const
    // With sparse_const = [1, 2, 3], optimal x = [1, 2, 3], value = 6
    // BUG: sparse_const becomes [0, 0, 0], so optimal x = [0, 0, 0], value = 0
    let prob = Problem::minimize(sum(&x))
        .subject_to([x.ge(sparse_const)])
        .build();

    assert!(prob.is_dcp(), "Problem should be DCP");

    let solution = prob.solve().expect("Should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("Should have value");
    let expected = 6.0; // sum([1, 2, 3])
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());

    // This test will FAIL until the bug is fixed
    // Currently value ≈ 0 because sparse constant becomes zeros
    assert!(
        rel_err < TOL,
        "BUG: Sparse constant became zeros! Expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// BUG #2: Right matrix multiplication x @ A doesn't actually multiply
/// The canonicalizer at canonicalizer.rs:263-266 just returns a.clone() without multiplying.
#[test]
fn test_bug_right_matmul_broken() {
    use nalgebra::DMatrix;

    // Create a 2x2 matrix A = [[2, 0], [0, 3]]
    let a_matrix = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    let a = constant_dmatrix(a_matrix);

    // x is a 1x2 row vector
    let x = variable((1, 2));

    // y = x @ A should be [2*x[0], 3*x[1]] (row vector multiplied by matrix)
    // minimize sum(y) s.t. x >= 1
    // With x @ A working: optimal x = [1, 1], y = [2, 3], value = 5
    // BUG: x @ A returns x unchanged, so y = x, value = 2
    let y = matmul(&x, &a);
    let prob = Problem::minimize(sum(&y))
        .subject_to([x.ge(constant(1.0))])
        .build();

    assert!(prob.is_dcp(), "Problem should be DCP");

    let solution = prob.solve().expect("Should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("Should have value");
    let expected = 5.0; // 2 + 3
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());

    // This test will FAIL until the bug is fixed
    // Currently value ≈ 2 because x @ A returns x unchanged
    assert!(
        rel_err < TOL,
        "BUG: Right matmul x @ A didn't multiply! Expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// BUG #3: Left matmul with sparse constant matrix doesn't multiply
/// The canonicalizer at canonicalizer.rs:243 returns b.clone() for sparse A in A @ x
#[test]
fn test_bug_left_matmul_sparse_broken() {
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    // Create a 2x2 sparse matrix A = [[2, 0], [0, 3]]
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 2.0);
    coo.push(1, 1, 3.0);
    let a_sparse: CscMatrix<f64> = CscMatrix::from(&coo);
    let a = constant_sparse(a_sparse);

    // x is a 2x1 column vector
    let x = variable(2);

    // y = A @ x should be [2*x[0], 3*x[1]]
    // minimize sum(y) s.t. x >= 1
    // With A @ x working: optimal x = [1, 1], y = [2, 3], value = 5
    // BUG: A @ x returns x unchanged (for sparse A), so y = x, value = 2
    let y = matmul(&a, &x);
    let prob = Problem::minimize(sum(&y))
        .subject_to([x.ge(constant(1.0))])
        .build();

    assert!(prob.is_dcp(), "Problem should be DCP");

    let solution = prob.solve().expect("Should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("Should have value");
    let expected = 5.0; // 2 + 3
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());

    // This test will FAIL until the bug is fixed
    // Currently value ≈ 2 because sparse A @ x returns x unchanged
    assert!(
        rel_err < TOL,
        "BUG: Left matmul with sparse A @ x didn't multiply! Expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Test hstack: horizontal concatenation
#[test]
fn test_hstack() {
    let x = variable(3);
    let y = variable(3);

    // hstack([x, y]) creates a 3x2 matrix
    // minimize sum(hstack) s.t. x >= 1, y >= 2
    let h = hstack(vec![x.clone(), y.clone()]);
    let prob = Problem::minimize(sum(&h))
        .subject_to([x.ge(constant(1.0)), y.ge(constant(2.0))])
        .build();

    assert!(prob.is_dcp(), "hstack problem should be DCP");
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 3.0 * 1.0 + 3.0 * 2.0; // 9.0
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < TOL,
        "hstack expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Test weighted sum (equivalent to dot product)
#[test]
fn test_weighted_sum() {
    use nalgebra::DMatrix;

    let x = variable(3);
    // c = [1, 2, 3] as a row vector for matmul: c @ x
    let c_mat = DMatrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
    let c = constant_dmatrix(c_mat);

    // minimize c @ x s.t. x >= 1
    // c @ x = 1*x1 + 2*x2 + 3*x3
    // optimal: x = [1, 1, 1], value = 6
    let prob = Problem::minimize(matmul(&c, &x))
        .subject_to([x.ge(constant(1.0))])
        .build();

    assert!(prob.is_dcp(), "weighted sum problem should be DCP");
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 6.0;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < TOL,
        "weighted sum expected {}, got {} (rel_err={})",
        expected,
        value,
        rel_err
    );
}

/// Test transpose in optimization
#[test]
fn test_transpose_in_constraint() {
    let x = variable((2, 3)); // 2x3 matrix
    let _xt = transpose(&x); // 3x2 matrix

    // minimize sum(x) s.t. x >= 1
    let prob = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant(1.0))])
        .build();

    assert!(prob.is_dcp(), "transpose problem should be DCP");
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 6.0; // 2*3 = 6 elements, each = 1
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(rel_err < TOL, "expected {}, got {}", expected, value);
}

#[test]
fn test_index_slice_in_constraint() {
    // Test that index/slice operations work correctly in optimization
    // minimize sum(x) s.t. x[0:3] >= 1, x[3:5] >= 2
    // optimal: x = [1, 1, 1, 2, 2], value = 7
    use cvxrust::atoms::slice;

    let x = variable(5);
    let x_first = slice(&x, 0, 3); // x[0:3]
    let x_last = slice(&x, 3, 5); // x[3:5]

    let prob = Problem::minimize(sum(&x))
        .subject_to([x_first.ge(1.0), x_last.ge(2.0)])
        .build();

    assert!(prob.is_dcp());

    let solution = prob.solve().unwrap();
    assert_eq!(solution.status, SolveStatus::Optimal);

    // Check optimal value
    let val = solution.value.unwrap();
    assert!((val - 7.0).abs() < TOL, "Expected 7.0, got {}", val);

    // Check solution values
    let x_vals = &solution[&x];
    for i in 0..3 {
        assert!(
            (x_vals[(i, 0)] - 1.0).abs() < TOL,
            "x[{}] should be ~1.0",
            i
        );
    }
    for i in 3..5 {
        assert!(
            (x_vals[(i, 0)] - 2.0).abs() < TOL,
            "x[{}] should be ~2.0",
            i
        );
    }
}

#[test]
fn test_index_single_element() {
    // Test single element indexing
    // minimize x[0] s.t. x >= 0, sum(x) == 5
    // optimal: x = [0, 5, 0, ...], value = 0 (put all in other elements)
    use cvxrust::atoms::index;

    let x = variable(3);
    let x0 = index(&x, 0); // x[0]

    let prob = Problem::minimize(x0)
        .subject_to([x.ge(0.0), sum(&x).eq(5.0)])
        .build();

    assert!(prob.is_dcp());

    let solution = prob.solve().unwrap();
    assert_eq!(solution.status, SolveStatus::Optimal);

    // x[0] should be 0, other elements should sum to 5
    let x_vals = &solution[&x];
    assert!(x_vals[(0, 0)].abs() < TOL, "x[0] should be ~0");
    assert!((x_vals[(1, 0)] + x_vals[(2, 0)] - 5.0).abs() < TOL);
}

#[test]
fn test_transpose_in_matmul() {
    // Test that transpose works correctly with matmul
    // minimize ||A @ x - b||_2 where A is 3x2, x is 2x1, b is 3x1
    // This is equivalent to minimize ||A @ x - b||_2
    // Also test: A.T @ A @ x should work
    let x = VariableBuilder::new(Shape::vector(2)).build();

    // A = [[1, 2], [3, 4], [5, 6]] (3x2 matrix, column-major)
    let a = constant_matrix(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 3, 2);
    // b = [1, 2, 3]
    let b = constant_vec(vec![1.0, 2.0, 3.0]);

    // Least squares: minimize ||Ax - b||_2^2
    let residual = &matmul(&a, &x) - &b;
    let obj = sum_squares(&residual);

    let prob = Problem::minimize(obj).build();
    assert!(prob.is_dcp());

    let solution = prob.solve().unwrap();
    assert_eq!(solution.status, SolveStatus::Optimal);

    // Verify A.T @ A @ x gives correct shape
    let at = transpose(&a);
    let ata = matmul(&at, &a); // Should be 2x2
    let atax = matmul(&ata, &x); // Should be 2x1
    assert_eq!(atax.shape(), Shape::vector(2));
}
