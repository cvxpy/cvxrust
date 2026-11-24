/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use approx::assert_abs_diff_eq;
use cvxrust::prelude::*;

fn assert_constant_solution(problem: &Problem, expected: f64) -> Solution {
    assert!(problem.variables().is_empty());
    let solution = problem.solve().expect("constant problem should solve");
    assert_eq!(solution.status, SolveStatus::Optimal);
    assert_abs_diff_eq!(solution.value.unwrap(), expected, epsilon = 1e-7);
    assert_abs_diff_eq!(
        problem
            .objective
            .expr()
            .value(&solution)
            .as_scalar()
            .unwrap(),
        expected,
        epsilon = 1e-7
    );
    assert!(solution.primal.as_ref().unwrap().is_empty());
    solution
}

#[test]
fn constant_objectives_preserve_offsets_and_sense() {
    let scalar = constant(2.0);
    let vector = constant_vec(vec![1.0, -2.0]);
    let p = constant_matrix(vec![2.0, 1.0, 1.0, 2.0], 2, 2);
    let negative_p = constant_matrix(vec![-2.0, -1.0, -1.0, -2.0], 2, 2);
    for (problem, expected) in [
        (Problem::minimize(constant(7.0)).build(), 7.0),
        (Problem::maximize(constant(-3.0)).build(), -3.0),
        (
            Problem::minimize(quad_form(&scalar, &constant(1.0)) + constant(1.0)).build(),
            5.0,
        ),
        (
            Problem::minimize(quad_form(&scalar, &constant(3.0))).build(),
            12.0,
        ),
        (
            Problem::minimize(quad_form(&vector, &p) + constant(4.0)).build(),
            10.0,
        ),
        (
            Problem::maximize(quad_form(&vector, &negative_p) + constant(4.0)).build(),
            -2.0,
        ),
        (
            Problem::minimize(sum_squares(&scalar) + constant(1.0)).build(),
            5.0,
        ),
        (
            Problem::maximize(-sum_squares(&scalar) + constant(1.0)).build(),
            -3.0,
        ),
        (
            Problem::minimize(quad_form(&scalar, &constant(0.0))).build(),
            0.0,
        ),
    ] {
        let solution = assert_constant_solution(&problem, expected);
        assert!(solution.constraint_dual(0).is_none());
    }
}

#[test]
fn feasible_constant_constraints_preserve_explicit_dual_mapping() {
    let problem = Problem::minimize(constant(4.0))
        .subject_to([
            constant_vec(vec![1.0, 2.0]).ge(0.0),
            constant(2.0).eq(2.0),
            Constraint::soc(constant(5.0), constant_vec(vec![3.0, 4.0])),
        ])
        .build();
    let solution = assert_constant_solution(&problem, 4.0);
    let rows = solution.constraint_rows.as_ref().unwrap();
    assert_eq!(rows.len(), problem.constraints.len());
    for (index, size) in [2, 1, 3].into_iter().enumerate() {
        assert_eq!(rows[index].len(), size);
        assert!(solution.constraint_dual(index).unwrap().is_finite());
    }
    assert!(solution.constraint_dual(3).is_none());
}

#[test]
fn infeasible_constant_constraints_return_errors() {
    for constraint in [
        constant(1.0).eq(2.0),
        constant_vec(vec![1.0, -1.0]).ge(0.0),
        Constraint::soc(constant(4.0), constant_vec(vec![3.0, 4.0])),
    ] {
        let problem = Problem::minimize(constant(7.0))
            .constraint(constraint)
            .build();
        assert!(problem.is_dcp());
        assert!(matches!(problem.solve(), Err(CvxError::SolverError(_))));
    }
}
