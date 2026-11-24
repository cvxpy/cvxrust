/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use cvxrust::prelude::*;

const TOL: f64 = 1e-4;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= TOL * (1.0 + expected.abs()),
        "expected {expected}, got {actual}"
    );
}

fn solve(problem: Problem, expected: f64) -> Solution {
    assert!(problem.is_dcp());
    let solution = problem.solve().expect("quadratic problem should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    let reported = solution.value.unwrap();
    let evaluated = problem
        .objective
        .expr()
        .value(&solution)
        .as_scalar()
        .unwrap();
    assert_close(reported, expected);
    assert_close(reported, evaluated);
    solution
}

fn assert_vector(x: &Expr, solution: &Solution, expected: &[f64]) {
    let values = x.value(solution);
    for (row, &value) in expected.iter().enumerate() {
        assert_close(values[(row, 0)], value);
    }
}

#[test]
fn quad_form_issue_14_objective_sum() {
    let x = variable(2);
    let p = constant_matrix(vec![2.0, 0.0, 0.0, 2.0], 2, 2);
    let solution = solve(
        Problem::minimize(quad_form(&x, &p) - sum(&x)).build(),
        -0.25,
    );
    assert_vector(&x, &solution, &[0.25, 0.25]);
}

#[test]
fn quad_form_linear_maps_and_offsets() {
    // With x[0] = 1, (Ax)'P(Ax) = 2 + 10*x[1] + 15*x[1]^2.
    let x = variable(2);
    let a = constant_matrix(vec![1.0, 0.0, 2.0, 1.0], 2, 2);
    let p = constant_matrix(vec![2.0, 1.0, 1.0, 3.0], 2, 2);
    let solution = solve(
        Problem::minimize(quad_form(&matmul(&a, &x), &p))
            .constraint(select(&x, AxisIndex::Index(0), AxisIndex::All).eq(1.0))
            .build(),
        1.0 / 3.0,
    );
    assert_vector(&x, &solution, &[1.0, -1.0 / 3.0]);

    // The normal equations are [[6, 4], [4, 7]] x = [2, 6].
    let x = variable(2);
    let a = constant_matrix(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 3, 2);
    let p = constant_matrix(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], 3, 3);
    let residual = matmul(&a, &x) - constant_vec(vec![1.0, 2.0, 0.0]);
    let solution = solve(
        Problem::minimize(quad_form(&residual, &p)).build(),
        108.0 / 13.0,
    );
    assert_vector(&x, &solution, &[-5.0 / 13.0, 14.0 / 13.0]);
}

#[test]
fn quad_form_cross_variable_terms() {
    let x = variable(());
    let y = variable(());
    let z = vstack(vec![&x + constant(1.0), &y - constant(1.0)]);
    let p = constant_matrix(vec![2.0, 1.0, 1.0, 3.0], 2, 2);
    // P * [2, 1] = [5, 5], so z = [2, 1] minimizes z'Pz - 10*sum(z).
    let solution = solve(
        Problem::minimize(quad_form(&z, &p) - 10.0 * sum(&z)).build(),
        -15.0,
    );
    assert_close(solution.value(&x), 1.0);
    assert_close(solution.value(&y), 2.0);
}

#[test]
fn quad_form_with_norm1_regularization() {
    let x = variable(2);
    let p = constant_matrix(vec![2.0, 0.0, 0.0, 3.0], 2, 2);
    let residual = &x - constant_vec(vec![2.0, 3.0]);
    let solution = solve(
        Problem::minimize(quad_form(&residual, &p) + norm1(&x)).build(),
        115.0 / 24.0,
    );
    assert_vector(&x, &solution, &[7.0 / 4.0, 17.0 / 6.0]);
}

#[test]
fn quad_form_scaling_sums_and_maximization() {
    let x = variable(());
    let q2 = quad_form(&x, &constant(2.0));
    let q3 = quad_form(&x, &constant_matrix(vec![3.0], 1, 1));
    let objective = &q2 * 2.0 + 3.0 * (&q3 / 3.0) - 14.0 * &x + constant(5.0);
    let solution = solve(Problem::minimize(objective.clone()).build(), -2.0);
    assert_close(solution.value(&x), 1.0);
    let solution = solve(Problem::maximize(-objective).build(), 2.0);
    assert_close(solution.value(&x), 1.0);

    let objective = quad_form(&x, &constant(-7.0)) + 14.0 * &x - constant(5.0);
    let solution = solve(Problem::maximize(objective).build(), 2.0);
    assert_close(solution.value(&x), 1.0);
}

#[test]
fn quad_form_constraints_preserve_sign_and_duals() {
    for sign in [1.0, -1.0] {
        let x = variable(2).nonneg();
        let residual = &x - constant_vec(vec![1.0, 1.0]);
        let p = constant_matrix(vec![4.0 * sign, 0.0, 0.0, sign], 2, 2);
        let q = quad_form(&residual, &p);
        let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
        let first = select(&x, AxisIndex::Index(0), AxisIndex::All);
        let second = select(&x, AxisIndex::Index(1), AxisIndex::All);
        let solution = solve(
            Problem::maximize(first.clone())
                .subject_to([second.ge(0.5), bound, first.le(2.0)])
                .build(),
            1.5,
        );
        assert_vector(&x, &solution, &[1.5, 1.0]);
        assert_close(q.value(&solution).as_scalar().unwrap(), sign);
        assert_close(solution.constraint_dual(0).unwrap(), 0.0);
        assert_close(solution.constraint_dual(1).unwrap(), 0.25);
        assert_close(solution.constraint_dual(2).unwrap(), 0.0);
        assert!(solution.constraint_dual(3).is_none());
    }
}

#[test]
fn quad_form_nondiagonal_affine_constraints() {
    for sign in [1.0, -1.0] {
        let x = variable(());
        let y = variable(());
        let z = vstack(vec![
            2.0 * &x + &y + constant(1.0),
            &x + 3.0 * &y - constant(2.0),
        ]);
        let p = constant_matrix(vec![2.0 * sign, sign, sign, 2.0 * sign], 2, 2);
        let q = quad_form(&z, &p);
        let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
        // Ellipsoid center: (-1, 1); quadratic matrix: [[14, 17], [17, 26]].
        let radius = (26.0_f64 / 75.0).sqrt();
        let solution = solve(
            Problem::maximize(x.clone()).constraint(bound).build(),
            -1.0 + radius,
        );
        assert_close(solution.value(&x), -1.0 + radius);
        assert_close(solution.value(&y), 1.0 - 17.0 / (75.0 * radius));
        assert_close(q.value(&solution).as_scalar().unwrap(), sign);
    }
}

#[test]
fn quad_form_nested_nonlinear_objectives() {
    let x = variable(());
    let q = quad_form(&x, &constant_matrix(vec![2.0], 1, 1));
    let solution = solve(
        Problem::minimize(max2(&q, &constant(1.0)))
            .constraint(x.ge(1.0))
            .build(),
        2.0,
    );
    assert_close(solution.value(&x), 1.0);

    let q = quad_form(&x, &constant_matrix(vec![-2.0], 1, 1));
    let solution = solve(
        Problem::maximize(log(&(q + constant(3.0))))
            .constraint(x.ge(0.5))
            .build(),
        2.5_f64.ln(),
    );
    assert_close(solution.value(&x), 0.5);
}

#[test]
fn quad_form_scalar_and_dense_one_by_one() {
    for weight in [2.0, -2.0] {
        for p in [constant(weight), constant_matrix(vec![weight], 1, 1)] {
            let x = variable(());
            let q = quad_form(&x, &p);
            let objective = if weight > 0.0 {
                Problem::minimize(q.clone())
            } else {
                Problem::maximize(q.clone())
            };
            let solution = solve(objective.constraint(x.ge(1.0)).build(), weight);
            assert_close(solution.value(&x), 1.0);

            let bound = if weight > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
            let radius = 1.0 / weight.abs().sqrt();
            let solution = solve(
                Problem::maximize(x.clone()).constraint(bound).build(),
                radius,
            );
            assert_close(solution.value(&x), radius);
            assert_close(q.value(&solution).as_scalar().unwrap(), weight.signum());
        }
    }

    let x = variable(());
    let q = quad_form(&x, &constant(0.0));
    let solution = solve(
        Problem::minimize(q.clone() + &x)
            .constraint(x.ge(2.0))
            .build(),
        2.0,
    );
    assert_close(solution.value(&x), 2.0);
    let solution = solve(
        Problem::minimize(max2(&q, &constant(1.0)))
            .constraint(x.eq(2.0))
            .build(),
        1.0,
    );
    assert_close(solution.value(&x), 2.0);
}

#[test]
fn quad_form_preserves_declared_variable_domains() {
    for (x, offset) in [(variable(()).nonneg(), 1.0), (variable(()).nonpos(), -1.0)] {
        let q = quad_form(&(&x + constant(offset)), &constant(2.0));
        let solution = solve(Problem::minimize(q).build(), 2.0);
        assert_close(solution.value(&x), 0.0);
    }
}

#[test]
fn quad_form_rejects_indefinite_and_nonsymmetric_matrices() {
    for p in [
        constant_matrix(vec![1.0, 0.0, 0.0, -1.0], 2, 2),
        constant_matrix(vec![1.0, 0.0, 2.0, 1.0], 2, 2),
    ] {
        let x = variable(2);
        let q = quad_form(&x, &p);
        for problem in [
            Problem::minimize(q.clone()).build(),
            Problem::maximize(q).build(),
        ] {
            assert!(!problem.is_dcp());
            assert!(matches!(problem.solve(), Err(CvxError::NotDcp(_))));
        }
    }
}
