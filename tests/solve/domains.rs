/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Variable sign domains and explicit constraint duals.

use super::TOL;
use cvxrust::prelude::*;

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
    let solution = problem.solve().expect("infeasibility is a solver outcome");
    assert_eq!(solution.status, SolveStatus::Infeasible);
    assert_eq!(solution.value, Some(f64::INFINITY));
    assert!(solution.primal.is_none());
    assert!(solution.dual.is_none());
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
