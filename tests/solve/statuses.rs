/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Solver outcomes, including reduced accuracy and incomplete iterates.

use cvxrust::prelude::*;

fn limit_settings() -> [(Settings, SolveStatus); 2] {
    [
        (
            Settings {
                max_iter: 0,
                ..Settings::default()
            },
            SolveStatus::MaxIterations,
        ),
        (
            Settings {
                time_limit: 0.0,
                ..Settings::default()
            },
            SolveStatus::MaxTime,
        ),
    ]
}

#[test]
fn optimal_inaccurate_retains_solution_values() {
    let x = variable(());
    let solution = Problem::minimize(x.clone())
        .subject_to([x.ge(1.0), x.le(2.0)])
        .solve_with(Settings {
            // Full convergence requires a strictly positive gap tolerance;
            // Clarabel's default reduced tolerances still allow AlmostSolved.
            tol_gap_abs: 0.0,
            tol_gap_rel: 0.0,
            ..Settings::default()
        })
        .expect("a reduced-accuracy optimum should be returned");

    assert_eq!(solution.status, SolveStatus::OptimalInaccurate);
    assert!((solution.value.unwrap() - 1.0).abs() < 1e-5);
    assert!((solution.value(&x) - 1.0).abs() < 1e-5);
    assert!((solution.constraint_dual(0).unwrap() - 1.0).abs() < 1e-5);
    assert!(solution.constraint_dual(1).unwrap().abs() < 1e-5);
    assert!(solution.constraint_dual(2).is_none());
    assert!(solution.iterations > 0);
    assert!(solution.solve_time.is_finite() && solution.solve_time >= 0.0);
}

#[test]
fn small_socp_returns_a_labeled_solution() {
    let x = variable(1);
    let solution = Problem::minimize(-sum(&x))
        .subject_to([sum_squares(&x).le(1e-7)])
        .solve()
        .expect("the small SOCP should return its solution");

    // Whether Clarabel meets its full tolerances can vary by platform.
    assert!(matches!(
        solution.status,
        SolveStatus::Optimal | SolveStatus::OptimalInaccurate
    ));
    let optimum = 1e-7_f64.sqrt();
    let primal = solution[&x][(0, 0)];
    // Use Clarabel's reduced absolute gap tolerance for this scaled problem.
    let tolerance = 5e-5;
    assert!(
        (solution.value.unwrap() + optimum).abs() < tolerance,
        "expected {}, got {:?}",
        -optimum,
        solution.value
    );
    assert!((primal - optimum).abs() < tolerance, "primal = {primal}");
    assert!((solution.value.unwrap() + primal).abs() < 1e-10);
    assert!(solution.constraint_dual(0).unwrap().is_finite());
}

#[test]
fn limits_retain_iterates_with_objective_offsets_and_sense() {
    for (settings, expected_status) in limit_settings() {
        for maximize in [false, true] {
            let x = variable(());
            let objective = constant(3.0) * &x + constant(7.0);
            let builder = if maximize {
                Problem::maximize(objective)
            } else {
                Problem::minimize(objective)
            };
            let solution = builder
                .subject_to([x.ge(1.0), x.le(2.0)])
                .solve_with(settings.clone())
                .expect("a solver limit should return the current iterate");

            assert_eq!(solution.status, expected_status);
            let primal = solution.value(&x);
            let objective = solution.value.unwrap();
            assert!(primal.is_finite() && objective.is_finite());
            assert!((objective - (3.0 * primal + 7.0)).abs() < 1e-8);
            assert_eq!(solution.duals().unwrap().len(), 2);
            assert!(solution.duals().unwrap().iter().all(|v| v.is_finite()));
            assert_eq!(solution.iterations, 0);
            assert!(solution.solve_time.is_finite() && solution.solve_time >= 0.0);
        }
    }
}

#[test]
fn limits_evaluate_original_nonlinear_objectives() {
    for (settings, expected_status) in limit_settings() {
        for sign in [1.0, -1.0] {
            let x = variable(()).nonneg();
            let z = vstack(vec![3.0 * &x + constant(1.0), 4.0 * &x - constant(2.0)]);
            let objective = sign * norm2(&z) + constant(7.0);
            let builder = if sign > 0.0 {
                Problem::minimize(objective)
            } else {
                Problem::maximize(objective)
            };
            let solution = builder
                .subject_to([x.ge(1.0), x.le(2.0)])
                .solve_with(settings.clone())
                .expect("a solver limit should return the current iterate");

            assert_eq!(solution.status, expected_status);
            let primal = solution.try_value(&x).unwrap();
            let expected = sign * (3.0 * primal + 1.0).hypot(4.0 * primal - 2.0) + 7.0;
            let reported = solution.value.unwrap();
            assert!(
                (reported - expected).abs() < 1e-10,
                "expected {expected}, got {reported}"
            );
            assert!(solution.constraint_dual(0).unwrap().is_finite());
            assert!(solution.constraint_dual(1).unwrap().is_finite());
            assert!(solution.constraint_dual(2).is_none());
            assert_eq!(solution.iterations, 0);
            assert!(solution.solve_time.is_finite() && solution.solve_time >= 0.0);
        }
    }
}

#[test]
fn limit_duals_skip_implicit_bounds_and_auxiliary_constraints() {
    for (settings, expected_status) in limit_settings() {
        let x = variable(()).nonneg();
        let solution = Problem::minimize(norm2(&x))
            .subject_to([x.ge(1.0), x.le(2.0)])
            .solve_with(settings)
            .expect("a solver limit should retain dual iterates");

        assert_eq!(solution.status, expected_status);
        let dual = solution.duals().unwrap();
        // The cone rows are `x >= 0`, `t >= 0`, `x >= 1`, `x <= 2`,
        // followed by the auxiliary SOC for `||x||_2 <= t`.
        assert_eq!(dual.len(), 6);
        assert!((solution.constraint_dual(0).unwrap() - dual[2]).abs() < 1e-10);
        assert!((solution.constraint_dual(1).unwrap() - dual[3]).abs() < 1e-10);
        assert!(solution.constraint_dual(2).is_none());
        assert!(solution.get_value(x.variable_id().unwrap()).is_some());
    }
}

#[test]
fn infeasible_outcomes_have_infinity_with_the_objective_sense() {
    for maximize in [false, true] {
        let x = variable(());
        let builder = if maximize {
            Problem::maximize(x.clone())
        } else {
            Problem::minimize(x.clone())
        };
        let solution = builder
            .subject_to([x.ge(1.0), x.le(0.0)])
            .solve()
            .expect("infeasibility should be returned as an outcome");

        assert_eq!(solution.status, SolveStatus::Infeasible);
        let expected = if maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        assert_eq!(solution.value, Some(expected));
        assert!(solution.primal.is_none());
        assert!(solution.dual.is_none());
        assert!(solution.get_value(x.variable_id().unwrap()).is_none());
        assert!(solution.constraint_dual(0).is_none());
    }
}

#[test]
fn unbounded_outcomes_have_infinity_with_the_objective_sense() {
    for maximize in [false, true] {
        let x = variable(());
        let builder = if maximize {
            Problem::maximize(x.clone()).subject_to([x.ge(0.0)])
        } else {
            Problem::minimize(x.clone()).subject_to([x.le(0.0)])
        };
        let solution = builder
            .solve()
            .expect("unboundedness should be returned as an outcome");

        assert_eq!(solution.status, SolveStatus::Unbounded);
        let expected = if maximize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        assert_eq!(solution.value, Some(expected));
        assert!(solution.primal.is_none());
        assert!(solution.dual.is_none());
        assert!(solution.get_value(x.variable_id().unwrap()).is_none());
        assert!(solution.constraint_dual(0).is_none());
    }
}
