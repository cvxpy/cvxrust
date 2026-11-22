/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Comprehensive solve tests for all atoms.
//!
//! Pattern inspired by CVXPY's test_constant_atoms.py:
//! Define test cases as data, then run them programmatically.

use super::TOL;
use cvxrust::prelude::*;

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
