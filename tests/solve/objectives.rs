/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use approx::assert_abs_diff_eq;
use cvxrust::prelude::*;
use cvxrust::sparse::csc_from_triplets;
use nalgebra_sparse::CscMatrix;

#[test]
fn objectives_require_exactly_one_entry() {
    for objective in [
        variable(2),
        constant_vec(vec![1.0, 2.0]),
        constant_matrix(vec![1.0, 2.0], 1, 2),
        constant_matrix(vec![1.0; 4], 2, 2),
        variable(0),
        constant_vec(Vec::new()),
        constant_matrix(Vec::new(), 1, 0),
        constant_sparse(CscMatrix::zeros(0, 1)),
    ] {
        for problem in [
            Problem::minimize(objective.clone()).build(),
            Problem::maximize(objective.clone()).build(),
        ] {
            assert!(matches!(problem.solve(), Err(CvxError::InvalidProblem(_))));
        }
    }
}

#[test]
fn objective_size_is_checked_before_dcp() {
    for (shape, scalar) in [(Shape::scalar(), true), (Shape::vector(2), false)] {
        let problem = Problem::maximize(power(&variable(shape), 2.0)).build();
        assert!(!problem.is_dcp());
        match problem.solve() {
            Err(CvxError::NotDcp(_)) if scalar => {}
            Err(CvxError::InvalidProblem(_)) if !scalar => {}
            result => panic!("unexpected result for scalar={scalar}: {result:?}"),
        }
    }
}

#[test]
fn scalar_objective_representations_preserve_values() {
    for value in [-3.0, 0.0, 2.0] {
        for objective in [
            constant(value),
            reshape(&constant(value), 1),
            constant_matrix(vec![value], 1, 1),
            constant_sparse(csc_from_triplets(1, 1, vec![0], vec![0], vec![value])),
        ] {
            for problem in [
                Problem::minimize(objective.clone()).build(),
                Problem::maximize(objective.clone()).build(),
            ] {
                let solution = problem.solve().unwrap();
                assert_eq!(solution.status, SolveStatus::Optimal);
                assert_abs_diff_eq!(solution.value.unwrap(), value, epsilon = 1e-7);
                assert!(solution.primal.as_ref().unwrap().is_empty());
            }
        }
    }
    for problem in [
        Problem::minimize(constant_sparse(CscMatrix::zeros(1, 1))).build(),
        Problem::maximize(constant_sparse(CscMatrix::zeros(1, 1))).build(),
    ] {
        let solution = problem.solve().unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert_abs_diff_eq!(solution.value.unwrap(), 0.0, epsilon = 1e-7);
    }

    for shape in [Shape::scalar(), Shape::vector(1), Shape::matrix(1, 1)] {
        let x = variable(shape);
        let objective = 3.0 * &x + constant(7.0);
        for problem in [
            Problem::minimize(objective.clone())
                .constraint(x.eq(2.0))
                .build(),
            Problem::maximize(objective.clone())
                .constraint(x.eq(2.0))
                .build(),
        ] {
            let solution = problem.solve().unwrap();
            assert_eq!(solution.status, SolveStatus::Optimal);
            assert_abs_diff_eq!(solution.value.unwrap(), 13.0, epsilon = 1e-7);
            assert_abs_diff_eq!(solution.try_value(&x).unwrap(), 2.0, epsilon = 1e-7);
        }
    }
}

#[test]
fn sparse_scalar_objectives_compose() {
    let x = variable(2);
    let scalar = constant_sparse(csc_from_triplets(1, 1, vec![0], vec![0], vec![2.0]));
    for (objective, expected) in [
        (sum(&(&x + &scalar)), 6.0),
        (quad_over_lin(&x, &scalar), 1.0),
    ] {
        for problem in [
            Problem::minimize(objective.clone())
                .constraint(x.eq(1.0))
                .build(),
            Problem::maximize(-&objective).constraint(x.eq(1.0)).build(),
        ] {
            let solution = problem.solve().unwrap();
            assert_eq!(solution.status, SolveStatus::Optimal);
            let expected = if problem.objective.is_minimize() {
                expected
            } else {
                -expected
            };
            assert_abs_diff_eq!(solution.value.unwrap(), expected, epsilon = 1e-7);
            assert_abs_diff_eq!(
                problem
                    .objective
                    .expr()
                    .eval(&solution)
                    .unwrap()
                    .as_scalar()
                    .unwrap(),
                expected,
                epsilon = 1e-7
            );
        }
    }
}
