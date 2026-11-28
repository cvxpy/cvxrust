/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use approx::assert_abs_diff_eq;
use cvxrust::prelude::*;
use cvxrust::sparse::csc_from_triplets;
use nalgebra_sparse::CscMatrix;
use std::sync::Arc;

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

#[test]
fn direct_arithmetic_objectives_broadcast_scalar_constants() {
    let x = variable(2);
    for scalar in [
        constant(2.0),
        constant_matrix(vec![2.0], 1, 1),
        constant_sparse(csc_from_triplets(1, 1, vec![0], vec![0], vec![2.0])),
    ] {
        for (a, b) in [(x.clone(), scalar.clone()), (scalar, x.clone())] {
            for (expr, expected) in [
                (Expr::Add(Arc::new(a.clone()), Arc::new(b.clone())), 6.0),
                (Expr::Mul(Arc::new(a.clone()), Arc::new(b.clone())), 4.0),
            ] {
                for problem in [
                    Problem::minimize(sum(&expr)).constraint(x.eq(1.0)).build(),
                    Problem::maximize(sum(&expr)).constraint(x.eq(1.0)).build(),
                ] {
                    let solution = problem.solve().unwrap();
                    assert_eq!(solution.status, SolveStatus::Optimal);
                    assert_abs_diff_eq!(solution.value.unwrap(), expected, epsilon = 1e-7);
                }
            }
        }
    }
}

#[test]
fn direct_extrema_objectives_broadcast_in_both_operand_orders() {
    let x = variable(2);
    for scalar in [
        constant(1.0),
        constant_matrix(vec![1.0], 1, 1),
        constant_sparse(csc_from_triplets(1, 1, vec![0], vec![0], vec![1.0])),
    ] {
        for args in [
            vec![Arc::new(x.clone()), Arc::new(scalar.clone())],
            vec![Arc::new(scalar), Arc::new(x.clone())],
        ] {
            for (expr, minimize, expected) in [
                (Expr::Maximum(args.clone()), true, 4.0),
                (Expr::Minimum(args), false, -1.0),
            ] {
                assert_eq!(expr.shape(), Shape::vector(2));
                let builder = if minimize {
                    Problem::minimize(sum(&expr))
                } else {
                    Problem::maximize(sum(&expr))
                };
                let solution = builder
                    .constraint(x.eq(constant_vec(vec![-2.0, 3.0])))
                    .solve()
                    .unwrap();
                assert_eq!(solution.status, SolveStatus::Optimal);
                assert_abs_diff_eq!(solution.value.unwrap(), expected, epsilon = 1e-7);
                assert!(matches!(
                    Problem::minimize(expr).solve(),
                    Err(CvxError::InvalidProblem(_))
                ));
            }
        }
    }
}

#[test]
fn direct_extrema_objectives_broadcast_rows_and_columns() {
    let x = variable((2, 2));
    let args = vec![
        Arc::new(x.clone()),
        Arc::new(constant_matrix(vec![1.0, 2.0], 1, 2)),
        Arc::new(constant_matrix(vec![-1.0, 2.0], 2, 1)),
    ];
    for (builder, expected) in [
        (Problem::minimize(sum(&Expr::Maximum(args.clone()))), 10.0),
        (Problem::maximize(sum(&Expr::Minimum(args))), -7.0),
    ] {
        let solution = builder
            .constraint(x.eq(constant_matrix(vec![-2.0, 4.0, 3.0, -5.0], 2, 2)))
            .solve()
            .unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert_abs_diff_eq!(solution.value.unwrap(), expected, epsilon = 1e-7);
    }
}

#[test]
fn vector_matrix_objectives_preserve_vector_results() {
    let x = variable(1);
    let y = variable((1, 2));
    let row = constant_matrix(vec![2.0, 3.0], 1, 2);
    let sparse_row = constant_sparse(csc_from_triplets(
        1,
        2,
        vec![0, 0],
        vec![0, 1],
        vec![2.0, 3.0],
    ));
    for (product, constraint) in [
        (matmul(&x, &row), x.eq(2.0)),
        (matmul(&x, &sparse_row), x.eq(2.0)),
        (matmul(&reshape(&constant(2.0), 1), &y), y.eq(row)),
    ] {
        assert_eq!(product.shape(), Shape::vector(2));
        let objective = sum_axis(&product, 0);
        assert_eq!(objective.shape(), Shape::scalar());
        for builder in [
            Problem::minimize(objective.clone()),
            Problem::maximize(objective.clone()),
        ] {
            let solution = builder.constraint(constraint.clone()).solve().unwrap();
            assert_eq!(solution.status, SolveStatus::Optimal);
            assert_abs_diff_eq!(solution.value.unwrap(), 10.0, epsilon = 1e-7);
            let Array::Dense(values) = product.eval(&solution).unwrap() else {
                panic!("vector product must use column storage");
            };
            assert_eq!(values.shape(), (2, 1));
            assert_abs_diff_eq!(values[(0, 0)], 4.0, epsilon = 1e-7);
            assert_abs_diff_eq!(values[(1, 0)], 6.0, epsilon = 1e-7);
        }
    }
}
