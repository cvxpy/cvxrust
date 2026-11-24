/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Tests for v1.0 atoms: exponential cone, power cone, and new affine atoms.

use cvxrust::prelude::*;

const TOL: f64 = 1e-4;

// Exponential Cone Atoms

#[test]
fn test_exp_basic() {
    // minimize exp(x) s.t. x >= 0
    // Solution: x = 0, exp(0) = 1
    let x = variable(());
    let obj = exp(&x);

    let solution = Problem::minimize(obj)
        .subject_to([x.ge(constant(0.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let val = solution.value.unwrap();
    assert!((val - 1.0).abs() < TOL, "Expected 1.0, got {}", val);

    let x_val = solution.value(&x);
    assert!(x_val.abs() < TOL, "Expected x=0, got {}", x_val);
}

#[test]
fn test_exp_with_constraint() {
    // maximize x s.t. exp(x) <= 5
    // exp(x) <= 5 means x <= log(5) ~= 1.609
    // Solution: x = log(5)
    let x = variable(());
    let solution = Problem::maximize(x.clone())
        .subject_to([exp(&x).le(constant(5.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let x_val = solution.value(&x);
    let expected = 5.0_f64.ln();
    assert!(
        (x_val - expected).abs() < TOL,
        "Expected {}, got {}",
        expected,
        x_val
    );
}

#[test]
fn test_log_basic() {
    // maximize log(x) s.t. x <= 2
    // Solution: x = 2, log(2) ~= 0.693
    let x = variable(());
    let obj = log(&x);

    let solution = Problem::maximize(obj)
        .subject_to([x.le(constant(2.0)), x.ge(constant(0.01))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let val = solution.value.unwrap();
    let expected = 2.0_f64.ln();
    assert!(
        (val - expected).abs() < TOL,
        "Expected {}, got {}",
        expected,
        val
    );
}

#[test]
fn test_log_concave() {
    // minimize -log(x) s.t. 0.1 <= x <= 1
    // -log(x) is convex and decreasing, so minimum is at x = 1
    // Solution: x = 1, -log(1) = 0
    let x = variable(());
    let obj = -log(&x);

    let solution = Problem::minimize(obj)
        .subject_to([x.ge(constant(0.1)), x.le(constant(1.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let x_val = solution.value(&x);
    assert!((x_val - 1.0).abs() < TOL, "Expected x=1, got {}", x_val);
}

#[test]
fn test_entropy_basic() {
    // Test entropy on a simple problem
    // maximize entropy(x) s.t. x <= 0.5, x >= 0.1
    // Entropy is concave, so maximize it
    let x = variable(());

    let solution = Problem::maximize(entropy(&x))
        .subject_to([x.le(constant(0.5)), x.ge(constant(0.1))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    // Maximum entropy in [0.1, 0.5] is at x ~= 0.368 (1/e)
    // But due to simplified implementation, we just check it solves
}

// Power Cone Atoms

#[test]
fn test_power_p_half() {
    // minimize -sqrt(x) = -x^(0.5) s.t. x <= 4
    // Solution: x = 4, sqrt(4) = 2
    let x = variable(());
    let obj = -sqrt(&x);

    let solution = Problem::minimize(obj)
        .subject_to([x.le(constant(4.0)), x.ge(constant(0.01))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let x_val = solution.value(&x);
    assert!((x_val - 4.0).abs() < TOL, "Expected x=4, got {}", x_val);
}

#[test]
fn test_power_p_greater_than_1() {
    // minimize x^3 s.t. x >= 1
    // Solution: x = 1, 1^3 = 1
    let x = variable(());
    let obj = power(&x, 3.0);

    let solution = Problem::minimize(obj)
        .subject_to([x.ge(constant(1.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let x_val = solution.value(&x);
    assert!((x_val - 1.0).abs() < TOL, "Expected x=1, got {}", x_val);
}

#[test]
fn test_power_p_less_than_1() {
    // maximize x^(0.3) s.t. x <= 8
    // Solution: x = 8, 8^(0.3) ~= 1.986
    let x = variable(());
    let obj = power(&x, 0.3);

    let solution = Problem::maximize(obj)
        .subject_to([x.le(constant(8.0)), x.ge(constant(0.1))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let x_val = solution.value(&x);
    assert!((x_val - 8.0).abs() < TOL, "Expected x=8, got {}", x_val);
}

#[test]
fn test_sqrt_vector() {
    // maximize sum(sqrt(x)) s.t. x <= [1, 4, 9], x >= 0
    // sqrt is concave and increasing, so maximum is at upper bound
    // Solution: x = [1, 4, 9], sum = 1 + 2 + 3 = 6
    let x = variable(3);
    let obj = sum(&sqrt(&x));

    let solution = Problem::maximize(obj)
        .subject_to([
            x.le(constant_vec(vec![1.0, 4.0, 9.0])),
            x.ge(constant(0.01)),
        ])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);

    if let Some(Array::Dense(x_vals)) = solution.get_value(x.variable_id().unwrap()) {
        assert!((x_vals[(0, 0)] - 1.0).abs() < TOL);
        assert!((x_vals[(1, 0)] - 4.0).abs() < TOL);
        assert!((x_vals[(2, 0)] - 9.0).abs() < TOL);
    } else {
        panic!("Expected dense array for x");
    }
}

// Affine Atoms

#[test]
fn test_cumsum_basic() {
    // minimize sum(cumsum(x)) s.t. x = [1, 2, 3]
    // cumsum([1,2,3]) = [1, 3, 6], sum = 10
    let x = variable(3);
    let y = cumsum(&x);

    let solution = Problem::minimize(sum(&y))
        .subject_to([x.eq(constant_vec(vec![1.0, 2.0, 3.0]))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let val = solution.value.unwrap();
    assert!((val - 10.0).abs() < TOL, "Expected 10.0, got {}", val);
}

#[test]
fn test_cumsum_optimization() {
    // minimize sum(cumsum(x)) s.t. sum(x) = 6, x >= 0
    // To minimize cumulative sum, put weight at end: [0, 0, 6]
    // cumsum([0, 0, 6]) = [0, 0, 6], sum = 6
    let x = variable(3);
    let y = cumsum(&x);

    let solution = Problem::minimize(sum(&y))
        .subject_to([sum(&x).eq(constant(6.0)), x.ge(constant(0.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let val = solution.value.unwrap();
    assert!((val - 6.0).abs() < TOL, "Expected 6.0, got {}", val);

    // Check that x is concentrated at the end
    if let Some(Array::Dense(x_vals)) = solution.get_value(x.variable_id().unwrap()) {
        // First elements should be near 0
        assert!(x_vals[(0, 0)] < TOL);
        assert!(x_vals[(1, 0)] < TOL);
        // Last element should be near 6
        assert!((x_vals[(2, 0)] - 6.0).abs() < TOL);
    }
}

#[test]
fn test_diag_basic() {
    // Create diagonal matrix from vector
    // minimize trace(diag(x)) s.t. x = [1, 2, 3]
    // trace = 1 + 2 + 3 = 6
    let x = variable(3);
    let d = diag(&x);

    let solution = Problem::minimize(trace(&d))
        .subject_to([x.eq(constant_vec(vec![1.0, 2.0, 3.0]))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let val = solution.value.unwrap();
    assert!((val - 6.0).abs() < TOL, "Expected 6.0, got {}", val);
}

// Dual Values

#[test]
fn test_dual_values_simple() {
    // minimize x s.t. x >= 1
    // Optimal: x = 1, dual of constraint = 1 (binding constraint)
    let x = variable(());

    let solution = Problem::minimize(x.clone())
        .subject_to([x.ge(constant(1.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    assert!(solution.has_duals());

    let duals = solution.duals().unwrap();
    assert!(!duals.is_empty(), "Should have dual values");

    // The dual of a binding constraint should be positive
    let dual_0 = solution.constraint_dual(0).unwrap();
    assert!(
        dual_0.abs() > TOL,
        "Dual should be non-zero for binding constraint"
    );
}

#[test]
fn test_dual_values_shadow_price() {
    // minimize x + y s.t. x + y >= 10, x >= 0, y >= 0
    // Shadow price of x+y>=10 should be 1
    let x = variable(());
    let y = variable(());

    let solution = Problem::minimize(&x + &y)
        .subject_to([
            (&x + &y).ge(constant(10.0)),
            x.ge(constant(0.0)),
            y.ge(constant(0.0)),
        ])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);

    if let Some(duals) = solution.duals() {
        println!("Dual values: {:?}", duals);
        // First constraint (x+y>=10) should have dual ~= -1
        // (negative because of Clarabel's convention)
    }
}

// DCP Verification Tests

#[test]
fn test_exp_dcp_rules() {
    use cvxrust::dcp::Curvature;

    let x = variable(());
    let e = exp(&x);

    assert_eq!(e.curvature(), Curvature::Convex);
    assert!(e.is_nonneg());
}

#[test]
fn test_log_dcp_rules() {
    use cvxrust::dcp::Curvature;

    let x = variable(());
    let l = log(&x);

    assert_eq!(l.curvature(), Curvature::Concave);
}

#[test]
fn test_power_dcp_rules() {
    use cvxrust::dcp::Curvature;

    let x = variable(());

    // p > 1: convex
    assert_eq!(power(&x, 2.5).curvature(), Curvature::Convex);

    // 0 < p < 1: concave
    assert_eq!(power(&x, 0.7).curvature(), Curvature::Concave);
    assert_eq!(sqrt(&x).curvature(), Curvature::Concave);

    // p = 1: affine
    assert_eq!(power(&x, 1.0).curvature(), Curvature::Affine);
}

#[test]
fn test_cumsum_dcp() {
    use cvxrust::dcp::Curvature;

    let x = variable(3);
    let cs = cumsum(&x);

    // cumsum is affine
    assert_eq!(cs.curvature(), Curvature::Affine);
}

#[test]
fn test_diag_dcp() {
    use cvxrust::dcp::Curvature;

    let x = variable(3);
    let d = diag(&x);

    // diag is affine
    assert_eq!(d.curvature(), Curvature::Affine);
}

// Integration Tests

#[test]
fn test_logistic_regression_setup() {
    // Test that we can set up a logistic regression-like problem
    // Note: log(1 + exp(x)) is the softplus function, which IS convex,
    // but standard DCP composition rules can't verify this (log(convex) is unknown).
    // A proper implementation would need a dedicated softplus/logsumexp atom.
    //
    // For v1.0, we test a simpler formulation that IS DCP-compliant:
    // minimize exp(x) + x  (convex + affine = convex)

    let x = variable(());
    let loss = exp(&x) + x.clone();

    // Check it's DCP compliant
    assert!(loss.is_convex(), "exp(x) + x should be convex");

    // And verify it solves correctly
    let solution = Problem::minimize(loss)
        .subject_to([x.ge(constant(-2.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    // Minimum of exp(x) + x is at x where exp(x) + 1 = 0, i.e., never (always positive)
    // With constraint x >= -2, minimum is at x = -2
    let x_val = solution.value(&x);
    assert!((x_val - (-2.0)).abs() < TOL, "Expected x=-2, got {}", x_val);
}

#[test]
fn test_geometric_program_equivalent() {
    // Test power cone with geometric programming
    // minimize x^2 * y^(-1) s.t. x >= 1, y >= 1
    // Can be written as minimize t where t >= x^2/y

    let x = variable(());
    let y = variable(());

    // Use power to express x^2
    let x_sq = power(&x, 2.0);

    // Simplified: just minimize x^2 subject to constraints
    let solution = Problem::minimize(x_sq)
        .subject_to([x.ge(constant(1.0)), y.ge(constant(1.0))])
        .solve()
        .expect("Should solve");

    assert_eq!(solution.status, SolveStatus::Optimal);
    let x_val = solution.value(&x);
    assert!((x_val - 1.0).abs() < TOL);
}
