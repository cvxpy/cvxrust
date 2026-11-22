/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use cvxrust::prelude::*;

const TOL: f64 = 1e-4;

// ── helpers ──────────────────────────────────────────────────────────────────

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

// ── variable access ───────────────────────────────────────────────────────────

#[test]
fn test_value_scalar_variable() {
    let x = variable(());
    let sol = Problem::minimize(x.clone())
        .subject_to([x.ge(2.0)])
        .solve()
        .unwrap();

    // expr.value(&sol) should give the same result as solution.value(&x)
    let via_expr = x.value(&sol).as_scalar().unwrap();
    let via_sol = sol.value(&x);
    assert!(approx(via_expr, via_sol));
    assert!(approx(via_expr, 2.0));
}

#[test]
fn test_value_vector_variable_indexing() {
    let x = variable(3);
    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    let vals = x.value(&sol);
    // Index operator should behave like &solution[&x]
    let via_index = &sol[&x];
    assert!(approx(vals[(0, 0)], via_index[(0, 0)]));
    assert!(approx(vals[(1, 0)], via_index[(1, 0)]));
    assert!(approx(vals[(2, 0)], via_index[(2, 0)]));
    // All elements should be ~1.0
    for i in 0..3 {
        assert!(approx(vals[(i, 0)], 1.0));
    }
}

// ── affine expressions ────────────────────────────────────────────────────────

#[test]
fn test_value_affine_scale_and_shift() {
    // minimize x  s.t.  x >= 3  → x* = 3
    // expr = 2*x + 1  →  value = 7
    let x = variable(());
    let expr = &x * 2.0 + constant(1.0);
    let sol = Problem::minimize(x.clone())
        .subject_to([x.ge(3.0)])
        .solve()
        .unwrap();

    let v = expr.value(&sol).as_scalar().unwrap();
    assert!(approx(v, 7.0), "expected 7.0, got {v}");
}

#[test]
fn test_value_promote_public_atom() {
    let x = variable(());
    let promoted = promote(&x, (2, 2));
    let sol = Problem::minimize(x.clone())
        .subject_to([x.eq(3.0)])
        .solve()
        .unwrap();

    let vals = promoted.value(&sol);
    assert!(approx(vals[(0, 0)], 3.0));
    assert!(approx(vals[(1, 0)], 3.0));
    assert!(approx(vals[(0, 1)], 3.0));
    assert!(approx(vals[(1, 1)], 3.0));
}

#[test]
fn test_value_broadcasted_row_and_column_affine_expressions() {
    let matrix = constant_matrix(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3);
    let row = constant_matrix(vec![10.0, 20.0, 30.0], 1, 3);
    let col = constant_matrix(vec![100.0, 200.0], 2, 1);
    let x = variable(());
    let sol = Problem::minimize(x.clone())
        .subject_to([x.eq(0.0)])
        .solve()
        .unwrap();

    let row_sum = (&matrix + &row).value(&sol);
    assert!(approx(row_sum[(0, 0)], 11.0));
    assert!(approx(row_sum[(1, 0)], 14.0));
    assert!(approx(row_sum[(0, 2)], 33.0));
    assert!(approx(row_sum[(1, 2)], 36.0));

    let col_weighted = (&col * &matrix).value(&sol);
    assert!(approx(col_weighted[(0, 0)], 100.0));
    assert!(approx(col_weighted[(1, 0)], 800.0));
    assert!(approx(col_weighted[(0, 2)], 300.0));
    assert!(approx(col_weighted[(1, 2)], 1200.0));
}

#[test]
fn test_value_matmul_residual() {
    // minimize ||Ax - b||^2  s.t.  x >= 0
    // A = I (2x2), b = [3, 4]  →  x* = [3, 4], residual = [0, 0]
    let a = constant_matrix(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    let b = constant_vec(vec![3.0, 4.0]);
    let x = variable(2);
    let residual = matmul(&a, &x) - &b;

    let sol = Problem::minimize(sum_squares(&residual))
        .subject_to([x.ge(0.0)])
        .solve()
        .unwrap();

    let r = residual.value(&sol);
    assert!(approx(r[(0, 0)], 0.0));
    assert!(approx(r[(1, 0)], 0.0));
}

#[test]
fn test_value_sum_of_vector() {
    // minimize sum(x)  s.t.  x >= 2  →  x* = [2,2,2], sum = 6
    let x = variable(3);
    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(2.0)])
        .solve()
        .unwrap();

    let s = sum(&x).value(&sol).as_scalar().unwrap();
    assert!(approx(s, 6.0), "expected 6.0, got {s}");
}

// ── nonlinear atoms ───────────────────────────────────────────────────────────

#[test]
fn test_value_norm2_matches_objective() {
    // minimize norm2(x)  s.t.  x >= 1  →  x* = [1,1], norm2 = sqrt(2)
    let x = variable(2);
    let obj = norm2(&x);
    let sol = Problem::minimize(obj.clone())
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let via_eval = obj.value(&sol).as_scalar().unwrap();
    assert!(
        approx(reported, via_eval),
        "reported={reported}, eval={via_eval}"
    );
    assert!(approx(reported, 2f64.sqrt()));
}

#[test]
fn test_value_norm1() {
    // minimize norm1(x)  s.t.  x >= 1  →  x* = [1,1,1], norm1 = 3
    let x = variable(3);
    let obj = norm1(&x);
    let sol = Problem::minimize(obj.clone())
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    let v = obj.value(&sol).as_scalar().unwrap();
    assert!(approx(v, 3.0), "expected 3.0, got {v}");
}

#[test]
fn test_value_norm_inf() {
    // minimize norm_inf(x)  s.t.  x >= [1, 2]  →  x* = [1, 2], norm_inf = 2
    let x = variable(2);
    let obj = norm_inf(&x);
    let sol = Problem::minimize(obj.clone())
        .subject_to([x.ge(constant_vec(vec![1.0, 2.0]))])
        .solve()
        .unwrap();

    let v = obj.value(&sol).as_scalar().unwrap();
    assert!(approx(v, 2.0), "expected 2.0, got {v}");
}

#[test]
fn test_value_sum_squares_matches_objective() {
    // minimize sum_squares(x)  s.t.  x >= 2  →  x* = [2,2], sum_squares = 8
    let x = variable(2);
    let obj = sum_squares(&x);
    let sol = Problem::minimize(obj.clone())
        .subject_to([x.ge(2.0)])
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let via_eval = obj.value(&sol).as_scalar().unwrap();
    assert!(
        approx(reported, via_eval),
        "reported={reported}, eval={via_eval}"
    );
    assert!(approx(reported, 8.0));
}

#[test]
fn test_value_abs() {
    // minimize x  s.t.  x >= -5  →  x* = -5, abs(x) = 5
    let x = variable(());
    let sol = Problem::minimize(x.clone())
        .subject_to([x.ge(-5.0)])
        .solve()
        .unwrap();

    let v = abs(&x).value(&sol).as_scalar().unwrap();
    assert!(approx(v, 5.0), "expected 5.0, got {v}");
}

// ── eval returns Result ───────────────────────────────────────────────────────

#[test]
fn test_eval_missing_variable_returns_err() {
    // Create a variable that is never added to any solution
    let x = variable(());
    let y = variable(()); // y is not in the solution for x

    let sol = Problem::minimize(x.clone())
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    // Evaluating y against a solution that only contains x should fail
    assert!(y.eval(&sol).is_err());
}

#[test]
fn test_eval_returns_ok_for_constant() {
    let x = variable(());
    let sol = Problem::minimize(x.clone())
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    // Constants don't need any variables, so eval always succeeds
    let c = constant(42.0);
    assert!(c.eval(&sol).is_ok());
    assert!(approx(c.value(&sol).as_scalar().unwrap(), 42.0));
}

// ── multiple variables ────────────────────────────────────────────────────────

#[test]
fn test_value_expression_with_two_variables() {
    // minimize x + y  s.t.  x >= 1, y >= 2  →  x*=1, y*=2
    // evaluate x - y  →  1 - 2 = -1
    let x = variable(());
    let y = variable(());
    let obj = x.clone() + y.clone();
    let diff = x.clone() - y.clone();

    let sol = Problem::minimize(obj)
        .subject_to([x.ge(1.0), y.ge(2.0)])
        .solve()
        .unwrap();

    let v = diff.value(&sol).as_scalar().unwrap();
    assert!(approx(v, -1.0), "expected -1.0, got {v}");
}

// ── old API / new API parity ──────────────────────────────────────────────────

/// solution.value(&x)  ==  x.value(&solution).as_scalar()
#[test]
fn test_parity_scalar_variable() {
    let x = variable(());
    let sol = Problem::minimize(x.clone())
        .subject_to([x.ge(5.0)])
        .solve()
        .unwrap();

    let old = sol.value(&x);
    let new = x.value(&sol).as_scalar().unwrap();
    assert!(approx(old, new), "old={old}, new={new}");
}

/// &solution[&x][(i,j)]  ==  x.value(&solution)[(i,j)]  for all elements
#[test]
fn test_parity_vector_variable_all_elements() {
    let x = variable(4);
    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant_vec(vec![1.0, 2.0, 3.0, 4.0]))])
        .solve()
        .unwrap();

    let old = &sol[&x];
    let new = x.value(&sol);
    for i in 0..4 {
        assert!(approx(old[(i, 0)], new[(i, 0)]), "mismatch at row {i}");
    }
}

/// &solution[&x][(i,j)]  ==  x.value(&solution)[(i,j)]  for a larger vector
#[test]
fn test_parity_large_vector_variable() {
    let x = variable(5);
    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(constant_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]))])
        .solve()
        .unwrap();

    let old = &sol[&x];
    let new = x.value(&sol);
    for i in 0..5 {
        assert!(approx(old[(i, 0)], new[(i, 0)]), "mismatch at row {i}");
        assert!(
            approx(new[(i, 0)], (i + 1) as f64),
            "wrong value at row {i}"
        );
    }
}

// ── value() vs solution.value consistency ────────────────────────────────────

#[test]
fn test_value_objective_matches_solution_value() {
    // For any solved problem, obj.value(&sol) should equal sol.value
    let x = variable(2);
    let obj = norm2(&x);
    let sol = Problem::minimize(obj.clone())
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    let via_sol = sol.value.unwrap();
    let via_expr = obj.value(&sol).as_scalar().unwrap();
    assert!(
        approx(via_sol, via_expr),
        "sol.value={via_sol} != obj.value(&sol)={via_expr}"
    );
}

#[test]
fn test_value_lp_objective_matches_solution_value() {
    let x = variable(3);
    let c = constant_vec(vec![1.0, 2.0, 3.0]);
    let obj = dot(&c, &x);
    let sol = Problem::minimize(obj.clone())
        .subject_to([x.ge(1.0)])
        .solve()
        .unwrap();

    let via_sol = sol.value.unwrap();
    let via_expr = obj.value(&sol).as_scalar().unwrap();
    assert!(
        approx(via_sol, via_expr),
        "sol.value={via_sol} != obj.value(&sol)={via_expr}"
    );
    assert!(approx(via_sol, 6.0)); // 1*1 + 2*1 + 3*1
}
