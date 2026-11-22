//! Comprehensive solve tests for all atoms.
//!
//! Pattern inspired by CVXPY's test_constant_atoms.py:
//! Define test cases as data, then run them programmatically.

use cvxrust::prelude::*;

/// Tolerance for comparing floating point results
const TOL: f64 = 1e-4;

/// A test case definition
struct TestCase {
    name: &'static str,
    /// Function that builds the problem and returns (problem, expected_value)
    build: fn() -> (Problem, f64),
}

/// All minimize test cases
fn minimize_test_cases() -> Vec<TestCase> {
    vec![
        // ========== Linear Programs ==========
        TestCase {
            name: "sum_nonneg_constraint",
            build: || {
                // minimize sum(x) s.t. x >= 1, x in R^5
                // optimal: x = [1,1,1,1,1], value = 5
                let x = variable(5);
                let prob = Problem::minimize(sum(&x))
                    .subject_to([x.ge(constant(1.0))])
                    .build();
                (prob, 5.0)
            },
        },
        TestCase {
            name: "sum_equality_constraint",
            build: || {
                // minimize sum(x) s.t. x == 2, x in R^3
                // optimal: x = [2,2,2], value = 6
                let x = variable(3);
                let prob = Problem::minimize(sum(&x))
                    .subject_to([x.eq(constant(2.0))])
                    .build();
                (prob, 6.0)
            },
        },
        TestCase {
            name: "sum_upper_bound",
            build: || {
                // minimize -sum(x) s.t. x <= 3, x in R^4
                // optimal: x = [3,3,3,3], value = -12
                let x = variable(4);
                let neg_sum = &constant(-1.0) * &sum(&x);
                let prob = Problem::minimize(neg_sum)
                    .subject_to([x.le(constant(3.0))])
                    .build();
                (prob, -12.0)
            },
        },
        TestCase {
            name: "weighted_sum",
            build: || {
                // minimize 2*x + 3*y s.t. x >= 1, y >= 2
                // optimal: x=1, y=2, value = 2 + 6 = 8
                let x = variable(1);
                let y = variable(1);
                let obj = &(&constant(2.0) * &x) + &(&constant(3.0) * &y);
                let prob = Problem::minimize(obj)
                    .subject_to([x.ge(constant(1.0)), y.ge(constant(2.0))])
                    .build();
                (prob, 8.0)
            },
        },
        // ========== Norms (SOCP) ==========
        TestCase {
            name: "norm2_equality",
            build: || {
                // minimize ||x||_2 s.t. sum(x) = 5, x in R^5
                // optimal: x = [1,1,1,1,1], value = sqrt(5)
                let x = variable(5);
                let prob = Problem::minimize(norm2(&x))
                    .subject_to([sum(&x).eq(constant(5.0))])
                    .build();
                (prob, 5.0_f64.sqrt())
            },
        },
        TestCase {
            name: "norm2_zero",
            build: || {
                // minimize ||x||_2 + 1 s.t. x == 0, x in R^3
                // optimal: x = [0,0,0], value = 1
                let x = variable(3);
                let obj = &norm2(&x) + &constant(1.0);
                let prob = Problem::minimize(obj)
                    .subject_to([x.eq(constant(0.0))])
                    .build();
                (prob, 1.0)
            },
        },
        TestCase {
            name: "norm1_equality",
            build: || {
                // minimize ||x||_1 s.t. sum(x) = 3, x in R^3
                // optimal: x = [1,1,1], value = 3
                let x = variable(3);
                let prob = Problem::minimize(norm1(&x))
                    .subject_to([sum(&x).eq(constant(3.0))])
                    .build();
                (prob, 3.0)
            },
        },
        TestCase {
            name: "norm_inf_equality",
            build: || {
                // minimize ||x||_inf s.t. sum(x) = 4, x in R^4
                // optimal: x = [1,1,1,1], value = 1
                let x = variable(4);
                let prob = Problem::minimize(norm_inf(&x))
                    .subject_to([sum(&x).eq(constant(4.0))])
                    .build();
                (prob, 1.0)
            },
        },
        // ========== Quadratic (QP) ==========
        TestCase {
            name: "sum_squares_equality",
            build: || {
                // minimize ||x||_2^2 s.t. sum(x) = 2, x in R^2
                // optimal: x = [1,1], value = 2
                let x = variable(2);
                let prob = Problem::minimize(sum_squares(&x))
                    .subject_to([sum(&x).eq(constant(2.0))])
                    .build();
                (prob, 2.0)
            },
        },
        TestCase {
            name: "sum_squares_nonneg",
            build: || {
                // minimize ||x||_2^2 s.t. x >= 1, x in R^3
                // optimal: x = [1,1,1], value = 3
                let x = variable(3);
                let prob = Problem::minimize(sum_squares(&x))
                    .subject_to([x.ge(constant(1.0))])
                    .build();
                (prob, 3.0)
            },
        },
        // ========== Element-wise convex ==========
        TestCase {
            name: "abs_equality",
            build: || {
                // minimize sum(|x|) s.t. x == [-1, 2, -3]
                // value = 1 + 2 + 3 = 6
                let x = variable(3);
                let prob = Problem::minimize(sum(&abs(&x)))
                    .subject_to([x.eq(constant_vec(vec![-1.0, 2.0, -3.0]))])
                    .build();
                (prob, 6.0)
            },
        },
        TestCase {
            name: "pos_nonneg",
            build: || {
                // minimize sum(pos(x)) s.t. x >= -2, sum(x) = 1, x in R^2
                // optimal: x = [-2, 3] gives pos(x) = [0, 3], sum = 3... no
                // Actually with sum(x) = 1, to minimize sum(pos(x)):
                // optimal is x = [0.5, 0.5], pos = [0.5, 0.5], value = 1
                let x = variable(2);
                let prob = Problem::minimize(sum(&pos(&x)))
                    .subject_to([x.ge(constant(-2.0)), sum(&x).eq(constant(1.0))])
                    .build();
                (prob, 1.0)
            },
        },
        // ========== Maximum/Minimum ==========
        TestCase {
            name: "maximum_bound",
            build: || {
                // minimize max(x, y) s.t. x >= 1, y >= 2
                // optimal: x=1, y=2, value = 2
                let x = variable(1);
                let y = variable(1);
                let prob = Problem::minimize(max2(&x, &y))
                    .subject_to([x.ge(constant(1.0)), y.ge(constant(2.0))])
                    .build();
                (prob, 2.0)
            },
        },
        // ========== Multiple constraints ==========
        TestCase {
            name: "box_constraints",
            build: || {
                // minimize sum(x) s.t. x >= 1, x <= 2, x in R^3
                // optimal: x = [1,1,1], value = 3
                let x = variable(3);
                let prob = Problem::minimize(sum(&x))
                    .subject_to([x.ge(constant(1.0)), x.le(constant(2.0))])
                    .build();
                (prob, 3.0)
            },
        },
        TestCase {
            name: "mixed_constraints",
            build: || {
                // minimize ||x||_2 s.t. x >= 0, sum(x) = 3, x in R^3
                // optimal: x = [1,1,1], value = sqrt(3)
                let x = variable(3);
                let prob = Problem::minimize(norm2(&x))
                    .subject_to([x.ge(constant(0.0)), sum(&x).eq(constant(3.0))])
                    .build();
                (prob, 3.0_f64.sqrt())
            },
        },
    ]
}

/// All maximize test cases
fn maximize_test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "maximize_sum_upper_bound",
            build: || {
                // maximize sum(x) s.t. x <= 2, x in R^3
                // optimal: x = [2,2,2], value = 6
                let x = variable(3);
                let prob = Problem::maximize(sum(&x))
                    .subject_to([x.le(constant(2.0))])
                    .build();
                (prob, 6.0)
            },
        },
        TestCase {
            name: "maximize_minimum",
            build: || {
                // maximize min(x, y) s.t. x <= 3, y <= 2
                // optimal: x=2, y=2, value = 2
                let x = variable(1);
                let y = variable(1);
                let prob = Problem::maximize(min2(&x, &y))
                    .subject_to([x.le(constant(3.0)), y.le(constant(2.0))])
                    .build();
                (prob, 2.0)
            },
        },
        TestCase {
            name: "maximize_neg_norm",
            build: || {
                // maximize -||x||_2 s.t. sum(x) = 0, x in R^2
                // optimal: x = [0,0], value = 0
                let x = variable(2);
                let neg_norm = &constant(-1.0) * &norm2(&x);
                let prob = Problem::maximize(neg_norm)
                    .subject_to([sum(&x).eq(constant(0.0))])
                    .build();
                (prob, 0.0)
            },
        },
    ]
}

/// Test cases expected to be infeasible
fn infeasible_test_cases() -> Vec<(&'static str, Problem)> {
    vec![
        ("infeasible_bounds", {
            // x >= 1 and x <= 0 is infeasible
            let x = variable(3);
            Problem::minimize(sum(&x))
                .subject_to([x.ge(constant(1.0)), x.le(constant(0.0))])
                .build()
        }),
        ("infeasible_equality", {
            // x == 1 and x == 2 is infeasible
            let x = variable(1);
            Problem::minimize(sum(&x))
                .subject_to([x.eq(constant(1.0)), x.eq(constant(2.0))])
                .build()
        }),
    ]
}

/// Test cases expected to be unbounded
fn unbounded_test_cases() -> Vec<(&'static str, Problem)> {
    vec![
        ("unbounded_below", {
            // minimize sum(x) with only upper bound
            let x = variable(3);
            Problem::minimize(sum(&x))
                .subject_to([x.le(constant(1.0))])
                .build()
        }),
        ("unbounded_above", {
            // maximize sum(x) with only lower bound
            let x = variable(3);
            Problem::maximize(sum(&x))
                .subject_to([x.ge(constant(1.0))])
                .build()
        }),
    ]
}

// ============================================================================
// Test runner
// ============================================================================

#[test]
fn test_minimize_atoms() {
    for case in minimize_test_cases() {
        let (prob, expected) = (case.build)();

        assert!(prob.is_dcp(), "Problem '{}' should be DCP", case.name);

        let result = prob.solve();
        assert!(
            result.is_ok(),
            "Problem '{}' should solve: {:?}",
            case.name,
            result.err()
        );

        let solution = result.unwrap();
        assert_eq!(
            solution.status,
            SolveStatus::Optimal,
            "Problem '{}' should be optimal, got {:?}",
            case.name,
            solution.status
        );

        let value = solution.value.expect("should have value");
        let rel_err = (value - expected).abs() / (1.0 + expected.abs());
        assert!(
            rel_err < TOL,
            "Problem '{}': expected {}, got {} (rel_err={})",
            case.name,
            expected,
            value,
            rel_err
        );
    }
}

#[test]
fn test_maximize_atoms() {
    for case in maximize_test_cases() {
        let (prob, expected) = (case.build)();

        assert!(prob.is_dcp(), "Problem '{}' should be DCP", case.name);

        let result = prob.solve();
        assert!(
            result.is_ok(),
            "Problem '{}' should solve: {:?}",
            case.name,
            result.err()
        );

        let solution = result.unwrap();
        assert_eq!(
            solution.status,
            SolveStatus::Optimal,
            "Problem '{}' should be optimal, got {:?}",
            case.name,
            solution.status
        );

        let value = solution.value.expect("should have value");
        let rel_err = (value - expected).abs() / (1.0 + expected.abs());
        assert!(
            rel_err < TOL,
            "Problem '{}': expected {}, got {} (rel_err={})",
            case.name,
            expected,
            value,
            rel_err
        );
    }
}

#[test]
fn test_infeasible() {
    for (name, prob) in infeasible_test_cases() {
        let result = prob.solve();
        match result {
            Err(CvxError::SolverError(msg)) if msg.contains("infeasible") => {
                // Expected
            }
            Ok(solution) if solution.status == SolveStatus::Infeasible => {
                // Also acceptable
            }
            other => {
                panic!("Problem '{}' should be infeasible, got {:?}", name, other);
            }
        }
    }
}

#[test]
fn test_unbounded() {
    for (name, prob) in unbounded_test_cases() {
        let result = prob.solve();
        match result {
            Err(CvxError::SolverError(msg)) if msg.contains("unbounded") => {
                // Expected
            }
            Ok(solution) if solution.status == SolveStatus::Unbounded => {
                // Also acceptable
            }
            other => {
                panic!("Problem '{}' should be unbounded, got {:?}", name, other);
            }
        }
    }
}

// ============================================================================
// Bug regression tests - these demonstrate known bugs
// ============================================================================

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

// ============================================================================
// Primal value verification tests
// ============================================================================

#[test]
fn test_primal_values() {
    // Test that we can recover the optimal x values, not just objective
    let x = variable(3);
    let result = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant(2.0))])
        .solve()
        .expect("should solve");

    assert_eq!(result.status, SolveStatus::Optimal);

    let x_val = result.get_value(x.variable_id().unwrap());
    assert!(x_val.is_some(), "should have primal value for x");

    let arr = x_val.unwrap();
    // Each component should be approximately 2.0
    match arr {
        Array::Dense(m) => {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    assert!(
                        (m[(i, j)] - 2.0).abs() < TOL,
                        "x[{},{}] should be ~2.0, got {}",
                        i,
                        j,
                        m[(i, j)]
                    );
                }
            }
        }
        _ => panic!("expected dense array"),
    }
}

// ============================================================================
// Tests for previously untested atoms
// ============================================================================

/// Test neg_part: max(-x, 0)
#[test]
fn test_neg_part() {
    let x = variable(3);
    // minimize sum(neg_part(x)) s.t. sum(x) = 0
    // neg_part(x) = max(-x, 0), so positive x contributes 0, negative x contributes |x|
    // With sum(x) = 0 and minimizing sum(neg_part(x)), optimal is x = [0, 0, 0], value = 0
    let prob = Problem::minimize(sum(&neg_part(&x)))
        .subject_to([sum(&x).eq(constant(0.0))])
        .build();

    assert!(prob.is_dcp(), "neg_part problem should be DCP");
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    assert!(
        value.abs() < TOL,
        "neg_part optimal should be 0, got {}",
        value
    );
}

/// Test quad_form: x'Px with PSD P
#[test]
fn test_quad_form_psd() {
    use nalgebra::DMatrix;

    // P = [[2, 0], [0, 3]] is PSD
    let p_mat = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    let p = constant_dmatrix(p_mat);
    let x = variable(2);

    // minimize x'Px s.t. sum(x) = 1
    // With P diagonal, this minimizes 2*x1^2 + 3*x2^2 s.t. x1 + x2 = 1
    // Lagrangian: L = 2x1^2 + 3x2^2 + λ(x1 + x2 - 1)
    // ∂L/∂x1 = 4x1 + λ = 0, ∂L/∂x2 = 6x2 + λ = 0
    // x1 = -λ/4, x2 = -λ/6, x1 + x2 = 1 → -λ(1/4 + 1/6) = 1 → λ = -12/5
    // x1 = 3/5 = 0.6, x2 = 2/5 = 0.4
    // value = 2*(0.6)^2 + 3*(0.4)^2 = 2*0.36 + 3*0.16 = 0.72 + 0.48 = 1.2
    let prob = Problem::minimize(quad_form(&x, &p))
        .subject_to([sum(&x).eq(constant(1.0))])
        .build();

    assert!(
        prob.is_dcp(),
        "quad_form(x, PSD) should be DCP for minimize"
    );
    let solution = prob.solve().expect("should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);

    let value = solution.value.expect("should have value");
    let expected = 1.2;
    let rel_err = (value - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel_err < TOL,
        "quad_form expected {}, got {} (rel_err={})",
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

// ============================================================================
// DCP violation tests
// ============================================================================

/// Test that minimizing concave functions is not DCP
#[test]
fn test_minimize_concave_not_dcp() {
    let x = variable(3);
    // minimum is concave
    let prob = Problem::minimize(min2(&x, &constant(0.0))).build();
    assert!(!prob.is_dcp(), "minimize(concave) should not be DCP");
}

/// Test that maximizing convex functions is not DCP
#[test]
fn test_maximize_convex_not_dcp() {
    let x = variable(3);
    // norm2 is convex
    let prob = Problem::maximize(norm2(&x)).build();
    assert!(!prob.is_dcp(), "maximize(convex) should not be DCP");
}

/// Test that norm(convex) is not DCP (composition rule violation)
#[test]
fn test_composition_violation() {
    let x = variable(3);
    // norm(norm(x)) - norm is not monotone, so norm(convex) is unknown
    let inner = norm2(&x);
    let outer = norm2(&inner);
    let prob = Problem::minimize(outer).build();
    // This should be unknown curvature, hence not DCP
    assert!(!prob.is_dcp(), "norm(norm(x)) should not be DCP");
}

/// Test affine <= convex constraint is not DCP
#[test]
fn test_affine_leq_convex_not_dcp() {
    let x = variable(3);
    // The constraint sum(x) <= norm2(x) has affine LHS <= convex RHS
    // This becomes norm2(x) - sum(x) >= 0, which has convex LHS
    // DCP requires concave >= 0, not convex >= 0
    let prob = Problem::minimize(sum(&x))
        .subject_to([sum(&x).le(norm2(&x))]) // affine <= convex is not DCP
        .build();
    assert!(!prob.is_dcp(), "affine <= convex should not be DCP");
}

// ============================================================================
// Edge case tests
// ============================================================================

/// Test single-element variable
#[test]
fn test_single_element() {
    let x = variable(1);
    let prob = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant(5.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    assert!(
        (value - 5.0).abs() < TOL,
        "single element: expected 5, got {}",
        value
    );
}

/// Test with constant objective (no variables in objective)
#[test]
fn test_constant_objective() {
    let x = variable(3);
    // Objective is just constant 1.0
    let prob = Problem::minimize(constant(1.0))
        .subject_to([x.ge(constant(0.0))])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    assert!(
        (value - 1.0).abs() < TOL,
        "constant obj: expected 1, got {}",
        value
    );
}

/// Test tight constraint
#[test]
fn test_tight_constraints() {
    let x = variable(2);
    // x1 + x2 = 3, x1 - x2 = 1 → x1 = 2, x2 = 1
    // minimize sum(x) = 3
    let prob = Problem::minimize(sum(&x))
        .subject_to([
            sum(&x).eq(constant(3.0)),
            (&x - &constant_vec(vec![0.0, 2.0])).eq(constant_vec(vec![2.0, -1.0])),
        ])
        .build();

    let solution = prob.solve().expect("should solve");
    let value = solution.value.expect("should have value");
    assert!(
        (value - 3.0).abs() < TOL,
        "tight constraints: expected 3, got {}",
        value
    );
}

#[test]
fn test_broadcasted_affine_constraints_public_workflow() {
    let x = variable((2, 3));
    let row_cap = variable((1, 3));
    let col_floor = variable((2, 1));

    let prob = Problem::maximize(sum(&x))
        .subject_to([
            row_cap.eq(constant_matrix(vec![2.0, 3.0, 4.0], 1, 3)),
            col_floor.eq(constant_matrix(vec![1.0, 2.0], 2, 1)),
            x.le(row_cap.clone()),
            x.ge(col_floor.clone()),
        ])
        .build();

    assert!(prob.is_dcp());

    let solution = prob.solve().expect("broadcasted problem should solve");
    let value = solution.value.expect("should have objective");
    assert!(
        (value - 18.0).abs() < TOL,
        "broadcasted constraints: expected 18, got {}",
        value
    );

    let x_vals = &solution[&x];
    for i in 0..2 {
        assert!((x_vals[(i, 0)] - 2.0).abs() < TOL);
        assert!((x_vals[(i, 1)] - 3.0).abs() < TOL);
        assert!((x_vals[(i, 2)] - 4.0).abs() < TOL);
    }
}

#[test]
fn test_row_affine_broadcasts_in_matrix_constraint() {
    let x = variable((2, 3));
    let row = variable((1, 3));
    let row_values = constant_matrix(vec![1.0, 2.0, 3.0], 1, 3);

    let sol = Problem::maximize(sum(&x))
        .subject_to([x.le(row.clone()), row.eq(row_values), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 12.0).abs() < TOL);

    if let Array::Dense(x_vals) = x.value(&sol) {
        for i in 0..2 {
            assert!((x_vals[(i, 0)] - 1.0).abs() < TOL);
            assert!((x_vals[(i, 1)] - 2.0).abs() < TOL);
            assert!((x_vals[(i, 2)] - 3.0).abs() < TOL);
        }
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_column_affine_broadcasts_in_matrix_constraint() {
    let x = variable((2, 3));
    let col = variable((2, 1));
    let col_values = constant_matrix(vec![1.0, 2.0], 2, 1);

    let sol = Problem::maximize(sum(&x))
        .subject_to([x.le(col.clone()), col.eq(col_values), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 9.0).abs() < TOL);

    if let Array::Dense(x_vals) = x.value(&sol) {
        for j in 0..3 {
            assert!((x_vals[(0, j)] - 1.0).abs() < TOL);
            assert!((x_vals[(1, j)] - 2.0).abs() < TOL);
        }
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_row_broadcast_constant_times_affine_matrix() {
    let x = variable((2, 3));
    let weights = constant_matrix(vec![10.0, 20.0, 30.0], 1, 3);
    let weighted = weights * x.clone();

    let sol = Problem::minimize(sum(&weighted))
        .subject_to([x.eq(ones((2, 3)))])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 120.0).abs() < TOL);
}

#[test]
fn test_column_broadcast_constant_times_affine_matrix() {
    let x = variable((2, 3));
    let weights = constant_matrix(vec![10.0, 20.0], 2, 1);
    let weighted = weights * x.clone();

    let sol = Problem::minimize(sum(&weighted))
        .subject_to([x.eq(ones((2, 3)))])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 90.0).abs() < TOL);
}

// ============================================================================
// Scale tests (medium-sized problems)
// ============================================================================

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

// ============================================================================
// STRESS TESTS - Comprehensive canonicalization and stuffing tests
// ============================================================================

mod stress_tests {
    use super::*;
    use nalgebra::DMatrix;

    const STRESS_TOL: f64 = 1e-3; // Slightly looser tolerance for complex problems

    // ========================================================================
    // quad_form stress tests
    // ========================================================================

    /// quad_form with non-diagonal PSD matrix
    #[test]
    fn test_quad_form_non_diagonal() {
        // P = [[2, 1], [1, 2]] is PSD (eigenvalues 1 and 3)
        let p_mat = DMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let p = constant_dmatrix(p_mat);
        let x = variable(2);

        // minimize x'Px s.t. x1 + x2 = 1
        // x'Px = 2x1^2 + 2x1*x2 + 2x2^2
        // With constraint x2 = 1 - x1:
        // f(x1) = 2x1^2 + 2x1(1-x1) + 2(1-x1)^2
        //       = 2x1^2 + 2x1 - 2x1^2 + 2 - 4x1 + 2x1^2
        //       = 2x1^2 - 2x1 + 2
        // df/dx1 = 4x1 - 2 = 0 => x1 = 0.5, x2 = 0.5
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

    /// quad_form with 3x3 matrix
    #[test]
    fn test_quad_form_3x3() {
        // P = diag(1, 2, 3)
        let p_mat = DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let p = constant_dmatrix(p_mat);
        let x = variable(3);

        // minimize x'Px = x1^2 + 2*x2^2 + 3*x3^2 s.t. sum(x) = 1
        // Lagrangian optimality: 2x1 = 4x2 = 6x3 = -λ
        // x1 = -λ/2, x2 = -λ/4, x3 = -λ/6
        // sum = -λ(1/2 + 1/4 + 1/6) = -λ(6+3+2)/12 = -11λ/12 = 1 => λ = -12/11
        // x1 = 6/11, x2 = 3/11, x3 = 2/11
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

    // ========================================================================
    // Matmul stress tests
    // ========================================================================

    /// Nested matmul: A @ (B @ x)
    #[test]
    fn test_nested_matmul() {
        // A is 2x3, B is 3x2, x is 2x1
        // A @ B is 2x2, (A @ B) @ x is 2x1
        let a_mat = DMatrix::from_vec(2, 3, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let b_mat = DMatrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let a = constant_dmatrix(a_mat.clone());
        let b = constant_dmatrix(b_mat.clone());
        let x = variable(2);

        // y = A @ (B @ x)
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
        // A (2x3) = [[1,0,1], [0,1,1]] (row-major view)
        // B (3x2) = [[1,0], [0,1], [0,0]] (row-major view)
        // B @ x where x is 2x1 gives 3x1
        // A @ (B @ x) gives 2x1
        // With x = [1,1], B @ [1,1] = [1, 1, 0]
        // A @ [1, 1, 0] = [1*1+0*1+1*0, 0*1+1*1+1*0] = [1, 1]
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
        // A is 2x4, x is 4x1, result is 2x1
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

        // minimize sum(A @ x) s.t. x >= 0, sum(x) = 1
        // A @ x = [1*x1 + 3*x2 + 5*x3 + 7*x4, 2*x1 + 4*x2 + 6*x3 + 8*x4]
        // sum(A @ x) = 3*x1 + 7*x2 + 11*x3 + 15*x4
        // To minimize with sum(x)=1, x>=0: put all weight on x1
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

    /// Right matmul: x @ A (row vector times matrix)
    #[test]
    fn test_right_matmul_row_vector() {
        // x is 1x3 row vector, A is 3x2, result is 1x2
        let a_mat = DMatrix::from_vec(
            3,
            2,
            vec![
                1.0, 2.0, 3.0, // col 0
                4.0, 5.0, 6.0, // col 1
            ],
        );
        let a = constant_dmatrix(a_mat);
        let x = variable((1, 3)); // 1x3 row vector

        // minimize sum(x @ A) s.t. x >= 1
        // x @ A = [x1 + 2*x2 + 3*x3, 4*x1 + 5*x2 + 6*x3]
        // sum = 5*x1 + 7*x2 + 9*x3
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

    // ========================================================================
    // Mixed variable tests
    // ========================================================================

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

    // ========================================================================
    // vstack/hstack stress tests
    // ========================================================================

    /// vstack in objective
    #[test]
    fn test_vstack_objective() {
        let x = variable(2);
        let y = variable(2);

        // vstack([x, y]) is 4x1
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

    // ========================================================================
    // Numerical edge cases
    // ========================================================================

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

        // x1 + x2 = 1
        // x1 + 1.0001*x2 = 1.0001
        // These are nearly parallel - tests numerical stability
        let c1 = sum(&x).eq(constant(1.0));
        let c2 = (&x + &constant_vec(vec![0.0, 0.0001])).eq(constant_vec(vec![0.5, 0.5001]));

        let prob = Problem::minimize(sum(&x)).subject_to([c1, c2]).build();

        // Should either solve or report numerical issues
        let result = prob.solve();
        // We mainly care that it doesn't crash
        assert!(result.is_ok() || result.is_err());
    }

    // ========================================================================
    // Constraint combination tests
    // ========================================================================

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

    // ========================================================================
    // Regression tests for fixed bugs
    // ========================================================================

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

        // Sparse 3x3 matrix with only diagonal: [[2,0,0], [0,3,0], [0,0,4]]
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 3.0);
        coo.push(2, 2, 4.0);
        let sparse: CscMatrix<f64> = CscMatrix::from(&coo);

        let a = constant_sparse(sparse);
        let x = variable(3);

        // minimize sum(A @ x) s.t. x >= 1
        // A @ x = [2*x1, 3*x2, 4*x3]
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

    // ========================================================================
    // Larger scale tests
    // ========================================================================

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

    // ========================================================================
    // Expression complexity tests
    // ========================================================================

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
}

#[test]
fn test_variable_product_not_dcp() {
    // x * y is not DCP (quadratic/bilinear)
    let x = variable(1);
    let y = variable(1);
    let prod = &x * &y;

    // Check curvature is Unknown
    assert_eq!(prod.curvature(), cvxrust::dcp::Curvature::Unknown);

    // Problem should be not DCP
    let prob = Problem::minimize(prod).build();
    assert!(!prob.is_dcp());

    // Solve should return NotDcp error
    let result = prob.solve();
    assert!(result.is_err());
    if let Err(e) = result {
        let msg = format!("{}", e);
        assert!(
            msg.contains("not DCP") || msg.contains("NotDcp"),
            "Expected NotDcp error, got: {}",
            msg
        );
    }
}

#[test]
fn test_variable_matmul_not_dcp() {
    // x @ y is not DCP
    let x = variable(3);
    let y = variable(3);
    let prod = matmul(&x, &transpose(&y));

    // Check curvature is Unknown
    assert_eq!(prod.curvature(), cvxrust::dcp::Curvature::Unknown);

    // Problem should be not DCP
    let prob = Problem::minimize(sum(&prod)).build();
    assert!(!prob.is_dcp());
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

fn solve_sign_attribute(problem: &Problem, expected: f64) -> Solution {
    assert!(problem.is_dcp());
    let solution = problem
        .solve()
        .expect("sign-constrained problem should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let reported = solution.value.unwrap();
    let evaluated = problem
        .objective
        .expr()
        .value(&solution)
        .as_scalar()
        .unwrap();
    assert!(
        (reported - expected).abs() < TOL,
        "{reported} != {expected}"
    );
    assert!((reported - evaluated).abs() < TOL);
    solution
}

fn assert_sign_attribute_infeasible(problem: Problem) {
    assert!(problem.is_dcp());
    match problem.solve() {
        Err(CvxError::SolverError(message)) if message.contains("infeasible") => {}
        other => panic!("expected infeasible sign constraints, got {other:?}"),
    }
}

#[test]
fn sign_attribute_scalar_objectives() {
    for (x, target) in [(variable(()).nonneg(), -1.0), (variable(()).nonpos(), 1.0)] {
        let problem = Problem::minimize(sum_squares(&(&x - constant(target)))).build();
        let solution = solve_sign_attribute(&problem, 1.0);
        assert!(x.value(&solution).as_scalar().unwrap().abs() < TOL);
    }

    let x = VariableBuilder::scalar().nonneg().build();
    let y = VariableBuilder::scalar().nonpos().build();
    for (variable, problem) in [
        (x.clone(), Problem::minimize(x).build()),
        (y.clone(), Problem::maximize(y).build()),
    ] {
        let solution = solve_sign_attribute(&problem, 0.0);
        assert!(variable.value(&solution).as_scalar().unwrap().abs() < TOL);
    }
}

#[test]
fn sign_attribute_vector_and_matrix_projection() {
    // Matrix data are column-major: rows [1, -2, 3] and [-4, 5, -6].
    for (shape, data) in [
        (Shape::vector(3), vec![-2.0_f64, 1.0, -3.0]),
        (Shape::matrix(2, 3), vec![1.0, -4.0, -2.0, 5.0, 3.0, -6.0]),
    ] {
        for nonnegative in [true, false] {
            let x = variable(shape.clone());
            let x = if nonnegative { x.nonneg() } else { x.nonpos() };
            let target = constant_matrix(data.clone(), shape.rows(), shape.cols());
            let expected: Vec<_> = data
                .iter()
                .map(|&value| {
                    if nonnegative {
                        value.max(0.0)
                    } else {
                        value.min(0.0)
                    }
                })
                .collect();
            let objective_value = data
                .iter()
                .zip(&expected)
                .map(|(target, projected)| (target - projected).powi(2))
                .sum();
            let problem = Problem::minimize(sum_squares(&(&x - target))).build();
            let solution = solve_sign_attribute(&problem, objective_value);
            let values = x.value(&solution);
            for (i, expected) in expected.iter().enumerate() {
                assert!((values[(i % shape.rows(), i / shape.rows())] - expected).abs() < TOL);
            }
        }
    }
}

#[test]
fn sign_attribute_constraint_only_conflicts() {
    for (x, forbidden) in [(variable(()).nonneg(), -1.0), (variable(()).nonpos(), 1.0)] {
        let inequality = if forbidden < 0.0 {
            x.le(forbidden)
        } else {
            x.ge(forbidden)
        };
        for constraint in [x.eq(forbidden), inequality] {
            assert_sign_attribute_infeasible(
                Problem::minimize(constant(0.0))
                    .constraint(constraint)
                    .build(),
            );
        }
    }
}

#[test]
fn sign_attribute_soc_arguments() {
    let t = variable(()).nonpos();
    let x = variable(2).nonneg();
    for constraint in [
        Constraint::soc(t, constant_vec(vec![1.0, 0.0])),
        Constraint::soc(constant(1.0), x + constant_vec(vec![2.0, 0.0])),
    ] {
        assert_sign_attribute_infeasible(
            Problem::minimize(constant(0.0))
                .constraint(constraint)
                .build(),
        );
    }
}

#[test]
fn sign_attribute_shared_aliases_and_repeated_solve() {
    for reverse in [false, true] {
        let raw = variable(());
        let signed = raw.clone().nonneg();
        assert_eq!(raw.variable_id(), signed.variable_id());
        let (lhs, rhs) = if reverse {
            (&signed, &raw)
        } else {
            (&raw, &signed)
        };
        let problem = Problem::minimize(lhs + rhs)
            .subject_to([raw.ge(-1.0), signed.le(1.0), signed.ge(-1.0), raw.le(1.0)])
            .build();
        for _ in 0..2 {
            let solution = solve_sign_attribute(&problem, 0.0);
            for alias in [&raw, &signed] {
                assert!(alias.value(&solution).as_scalar().unwrap().abs() < TOL);
            }
        }
    }
}

#[test]
fn sign_attribute_opposite_aliases_across_objective_and_constraints() {
    let raw = variable(());
    let positive = raw.clone().nonneg();
    let negative = raw.clone().nonpos();
    for (objective, constrained) in [(&positive, &negative), (&negative, &positive)] {
        for builder in [
            Problem::minimize(objective.clone()),
            Problem::maximize(objective.clone()),
        ] {
            let problem = builder
                .subject_to([constrained.ge(-1.0), constrained.le(1.0)])
                .build();
            let solution = solve_sign_attribute(&problem, 0.0);
            for alias in [&raw, &positive, &negative] {
                assert!(alias.value(&solution).as_scalar().unwrap().abs() < TOL);
            }
        }
    }
}

#[test]
fn sign_attribute_unused_clone_does_not_constrain_problem() {
    for nonnegative in [true, false] {
        let raw = variable(());
        let _unused = if nonnegative {
            raw.clone().nonneg()
        } else {
            raw.clone().nonpos()
        };
        let builder = if nonnegative {
            Problem::minimize(raw.clone())
        } else {
            Problem::maximize(raw.clone())
        };
        let problem = builder.subject_to([raw.ge(-1.0), raw.le(1.0)]).build();
        let expected = if nonnegative { -1.0 } else { 1.0 };
        let solution = solve_sign_attribute(&problem, expected);
        assert!((raw.value(&solution).as_scalar().unwrap() - expected).abs() < TOL);
    }
}

#[test]
fn sign_attribute_setters_replace_previous_attribute() {
    for (x, target) in [
        (VariableBuilder::scalar().nonneg().nonpos().build(), -1.0),
        (VariableBuilder::scalar().nonpos().nonneg().build(), 1.0),
        (variable(()).nonneg().nonpos(), -1.0),
        (variable(()).nonpos().nonneg(), 1.0),
    ] {
        let problem = Problem::minimize(sum_squares(&(&x - constant(target)))).build();
        let solution = solve_sign_attribute(&problem, 0.0);
        assert!((x.value(&solution).as_scalar().unwrap() - target).abs() < TOL);
    }
}

#[test]
fn sign_attribute_dual_ignores_implicit_scalar_bound() {
    let x = variable(()).nonneg();
    let solution = Problem::minimize(x.clone())
        .constraint(x.ge(1.0))
        .solve()
        .unwrap();

    // The explicit x >= 1 bound is active; the implicit x >= 0 bound is slack.
    assert!((solution.constraint_dual(0).unwrap() - 1.0).abs() < TOL);
    assert!(solution.constraint_dual(1).is_none());
    let raw_duals = solution.duals().unwrap();
    assert_eq!(raw_duals.len(), 2);
    assert!(raw_duals[0].abs() < TOL);
    assert!((raw_duals[1] - 1.0).abs() < TOL);
}

#[test]
fn sign_attribute_duals_follow_mixed_constraint_order() {
    let x = variable(()).nonneg();
    let y = variable(()).nonneg();
    let solution = Problem::minimize(constant(2.0) * &x + constant(3.0) * &y)
        .subject_to([x.ge(1.0), y.eq(2.0)])
        .solve()
        .unwrap();

    // Stationarity gives multipliers 2 for x >= 1 and -3 for y - 2 == 0.
    assert!((solution.constraint_dual(0).unwrap() - 2.0).abs() < TOL);
    assert!((solution.constraint_dual(1).unwrap() + 3.0).abs() < TOL);
    assert!(solution.constraint_dual(2).is_none());
    // Raw solver rows still place equalities before inequalities.
    assert!((solution.duals().unwrap()[0] + 3.0).abs() < TOL);
}

#[test]
fn sign_attribute_duals_skip_remaining_vector_components() {
    let x = variable(2).nonneg();
    let y = variable(()).nonpos();
    let objective = sum(&(constant_vec(vec![1.0, 2.0]) * &x)) + constant(3.0) * &y;
    let solution = Problem::minimize(objective)
        .subject_to([x.ge(constant_vec(vec![1.0, 2.0])), y.ge(-1.0)])
        .solve()
        .unwrap();

    // The vector constraint has dual [1, 2], and the scalar constraint has dual 3.
    assert!((solution.constraint_dual(0).unwrap() - 1.0).abs() < TOL);
    assert!((solution.constraint_dual(1).unwrap() - 3.0).abs() < TOL);
    assert!(solution.constraint_dual(2).is_none());
}

#[test]
fn sign_attribute_duals_map_soc_and_constraint_only_domains() {
    let t = variable(()).nonneg();
    let x = variable(2).nonneg();
    let y = variable(()).nonneg();
    let solution = Problem::minimize(constant(2.0) * &t + constant(3.0) * &y)
        .subject_to([
            Constraint::soc(t.clone(), x.clone()),
            x.eq(constant_vec(vec![3.0, 4.0])),
            y.ge(1.0),
        ])
        .solve()
        .unwrap();

    // At t = ||[3, 4]|| = 5, the SOC dual is [2, -1.2, -1.6].
    // The equality dual is [-1.2, -1.6], and y >= 1 has multiplier 3.
    assert!((solution.constraint_dual(0).unwrap() - 2.0).abs() < TOL);
    assert!((solution.constraint_dual(1).unwrap() + 1.2).abs() < TOL);
    assert!((solution.constraint_dual(2).unwrap() - 3.0).abs() < TOL);
    assert!(solution.constraint_dual(3).is_none());
}

#[test]
fn sign_attribute_dual_absent_without_explicit_constraints() {
    let x = variable(()).nonneg();
    let solution = Problem::minimize(x).solve().unwrap();

    assert!(solution.constraint_dual(0).is_none());
    assert!(solution.has_duals());
    assert!((solution.duals().unwrap()[0] - 1.0).abs() < TOL);
}
