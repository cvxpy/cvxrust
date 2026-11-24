/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use approx::assert_abs_diff_eq;
use cvxrust::dcp::PsdStatus;
use cvxrust::prelude::*;
use cvxrust::sparse::dense_to_csc;
use nalgebra::{DMatrix, DVector};

fn diagonal(values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(values))
}

fn representations(matrix: DMatrix<f64>) -> [Expr; 2] {
    [
        constant_sparse(dense_to_csc(&matrix)),
        constant_dmatrix(matrix),
    ]
}

fn solve(problem: Problem, expected: f64) -> Solution {
    assert!(problem.is_dcp());
    let solution = problem.solve().unwrap();
    assert_eq!(solution.status, SolveStatus::Optimal);
    assert_abs_diff_eq!(solution.value.unwrap(), expected, epsilon = 1e-5);
    assert_abs_diff_eq!(
        problem
            .objective
            .expr()
            .value(&solution)
            .as_scalar()
            .unwrap(),
        expected,
        epsilon = 1e-5
    );
    solution
}

#[test]
fn singular_diagonal_objectives() {
    for sign in [1.0, -1.0] {
        for p in representations(diagonal(&[sign, 0.0])) {
            let x = variable(2);
            let q = quad_form(&x, &p);
            assert_eq!(
                q.curvature(),
                if sign > 0.0 {
                    Curvature::Convex
                } else {
                    Curvature::Concave
                }
            );
            assert_eq!(
                q.sign(),
                if sign > 0.0 {
                    Sign::Nonnegative
                } else {
                    Sign::Nonpositive
                }
            );
            let problem = if sign > 0.0 {
                Problem::minimize(q)
            } else {
                Problem::maximize(q)
            };
            let solution = solve(problem.constraint(x.ge(1.0)).build(), sign);
            let values = x.value(&solution);
            assert_abs_diff_eq!(values[(0, 0)], 1.0, epsilon = 1e-5);
            assert!(values[(1, 0)] >= 1.0 - 1e-5);
        }
    }
}

#[test]
fn rank_one_objectives_constraints_and_composition() {
    for sign in [1.0, -1.0] {
        for p in representations(DMatrix::from_element(2, 2, sign)) {
            let x = variable(2);
            let q = quad_form(&x, &p);
            let problem = if sign > 0.0 {
                Problem::minimize(q.clone())
            } else {
                Problem::maximize(q.clone())
            };
            solve(problem.constraint(x.ge(1.0)).build(), 4.0 * sign);

            let bound = if sign > 0.0 { q.le(4.0) } else { q.ge(-4.0) };
            let solution = solve(
                Problem::maximize(sum(&x))
                    .subject_to([bound, x.ge(0.0)])
                    .build(),
                2.0,
            );
            assert_abs_diff_eq!(
                q.value(&solution).as_scalar().unwrap(),
                4.0 * sign,
                epsilon = 1e-5
            );

            let (problem, expected) = if sign > 0.0 {
                (Problem::minimize(max2(&q, &constant(1.0))), 4.0)
            } else {
                (Problem::maximize(log(&(q + constant(10.0)))), 6.0_f64.ln())
            };
            solve(problem.constraint(x.ge(1.0)).build(), expected);
        }
    }
}

#[test]
fn singular_gram_forms_survive_permutations() {
    let a = DMatrix::from_row_slice(2, 3, &[5.0, 0.0, 9.0, 4.0, 1.0, -10.0]);
    let matrix = a.transpose() * a;
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let permuted = DMatrix::from_fn(3, 3, |i, j| matrix[(order[i], order[j])]);
        for sign in [1.0, -1.0] {
            for p in representations(&permuted * sign) {
                let array = p.constant_value().unwrap();
                assert_eq!(array.is_psd(), Some(sign > 0.0), "order={order:?}");
                assert_eq!(
                    PsdStatus::of_array(array),
                    if sign > 0.0 {
                        PsdStatus::Psd
                    } else {
                        PsdStatus::Nsd
                    }
                );
                let x = variable(3);
                let q = quad_form(&x, &p);
                let native = if sign > 0.0 {
                    Problem::minimize(q)
                } else {
                    Problem::maximize(q)
                };
                solve(native.constraint(x.eq(1.0)).build(), sign * 221.0);

                let t = variable(());
                let z = promote(&t, 3);
                let q = quad_form(&z, &p);
                let bound = if sign > 0.0 {
                    q.le(221.0)
                } else {
                    q.ge(-221.0)
                };
                let solution = solve(Problem::maximize(t.clone()).constraint(bound).build(), 1.0);
                assert_abs_diff_eq!(solution.value(&t), 1.0, epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn singular_affine_maps_and_multiple_variables() {
    for sign in [1.0, -1.0] {
        for p in representations(diagonal(&[sign, 2.0 * sign, 0.0])) {
            let u = variable(());
            let v = variable(());
            let a = constant_dmatrix(DMatrix::from_row_slice(
                3,
                2,
                &[1.0, 1.0, 1.0, -1.0, 2.0, 0.0],
            ));
            let z =
                matmul(&a, &vstack(vec![u.clone(), v.clone()])) + constant_vec(vec![1.0, 2.0, 1.0]);
            let q = quad_form(&z, &p);
            // For positive P: (u + v + 1)^2 + 2*(u - v + 2)^2 - 2*u.
            let objective = &q - 2.0 * sign * &u;
            let problem = if sign > 0.0 {
                Problem::minimize(objective)
            } else {
                Problem::maximize(objective)
            };
            let solution = solve(problem.build(), sign * 21.0 / 8.0);
            assert_abs_diff_eq!(solution.value(&u), -9.0 / 8.0, epsilon = 1e-5);
            assert_abs_diff_eq!(solution.value(&v), 5.0 / 8.0, epsilon = 1e-5);

            let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
            solve(
                Problem::maximize(u.clone()).constraint(bound).build(),
                -1.5 + (3.0_f64 / 8.0).sqrt(),
            );
        }
    }
}

#[test]
fn zero_forms_preserve_domains_and_duals() {
    let mut zero_matrices = vec![constant(0.0)];
    for n in [1, 2] {
        zero_matrices.extend(representations(DMatrix::zeros(n, n)));
    }
    for p in zero_matrices {
        let x = variable(p.shape().rows());
        let q = quad_form(&x.clone().nonneg(), &p);
        assert_eq!(q.curvature(), Curvature::Constant);
        assert_eq!(q.sign(), Sign::Zero);
        for problem in [Problem::minimize(q.clone()), Problem::maximize(q.clone())] {
            solve(problem.constraint(sum(&x).eq(2.0)).build(), 0.0);
        }

        let solution = solve(
            Problem::minimize(&q + sum(&x))
                .subject_to([q.eq(0.0), sum(&x).ge(2.0), sum(&x).le(3.0)])
                .build(),
            2.0,
        );
        assert_eq!(solution.constraint_rows.as_ref().unwrap().len(), 3);
        assert!(solution.constraint_dual(0).unwrap().is_finite());
        assert_abs_diff_eq!(solution.constraint_dual(1).unwrap(), 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(solution.constraint_dual(2).unwrap(), 0.0, epsilon = 1e-5);
        assert!(solution.constraint_dual(3).is_none());

        let infeasible = Problem::minimize(q).constraint(x.eq(-1.0)).build();
        assert!(infeasible.is_dcp());
        assert!(matches!(infeasible.solve(), Err(CvxError::SolverError(_))));
        assert!(!Problem::minimize(quad_form(&abs(&x), &p)).build().is_dcp());
    }
}

#[test]
fn zero_forms_in_products_preserve_variable_domains() {
    let x = variable(());
    let q = quad_form(&x.clone().nonneg(), &constant(0.0));
    let y = variable(2);
    let matrix = variable((2, 2));
    let zero_matrix = promote(&q, (2, 2));
    for (product, expected) in [
        (&q * &y, 0.0),
        (&y * &q, 0.0),
        ((&q + constant(2.0)) * &y, 4.0),
        (&y * (&q + constant(2.0)), 4.0),
        (matmul(&zero_matrix, &matrix), 0.0),
        (matmul(&matrix, &zero_matrix), 0.0),
    ] {
        solve(
            Problem::minimize(sum(&product))
                .subject_to([y.eq(1.0), matrix.eq(1.0), x.eq(1.0)])
                .build(),
            expected,
        );
        let infeasible = Problem::minimize(sum(&product))
            .subject_to([y.eq(1.0), matrix.eq(1.0), x.le(-1.0)])
            .build();
        assert!(infeasible.is_dcp());
        assert!(matches!(infeasible.solve(), Err(CvxError::SolverError(_))));
    }
}

#[test]
fn cumulative_constant_multipliers_preserve_values_and_domains() {
    let x = variable(());
    let zero = quad_form(&x.clone().nonneg(), &constant(0.0));
    let y = variable(2);
    let matrix = variable((2, 2));
    let coefficients = cumsum(&(&zero + constant_vec(vec![1.0, 2.0])));
    let matrix_constant = &zero + constant_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let row_sums = cumsum(&matrix_constant);
    let column_sums = Expr::Cumsum(std::sync::Arc::new(matrix_constant), Some(1));
    for (objective, expected) in [
        (cumsum(&(&zero + constant(2.0))) * &x, 6.0),
        (sum(&(&coefficients * &y)), 4.0),
        (sum(&(&y * &coefficients)), 4.0),
        (matmul(&transpose(&coefficients), &y), 4.0),
        (matmul(&transpose(&y), &coefficients), 4.0),
        (sum(&(cumsum(&coefficients) * &y)), 5.0),
        (sum(&(&row_sums * &matrix)), 14.0),
        (sum(&matmul(&row_sums, &y)), 14.0),
        (sum(&(&column_sums * &matrix)), 13.0),
    ] {
        solve(
            Problem::minimize(objective.clone())
                .subject_to([x.eq(3.0), y.eq(1.0), matrix.eq(1.0)])
                .build(),
            expected,
        );
        let infeasible = Problem::minimize(objective)
            .subject_to([x.le(-1.0), y.eq(1.0), matrix.eq(1.0)])
            .build();
        assert!(infeasible.is_dcp());
        assert!(matches!(infeasible.solve(), Err(CvxError::SolverError(_))));
    }
}

#[test]
fn singular_constant_arguments() {
    for sign in [1.0, -1.0] {
        for p in representations(DMatrix::from_element(2, 2, sign)) {
            let q = quad_form(&constant_vec(vec![1.0, 2.0]), &p);
            let problem = if sign > 0.0 {
                Problem::minimize(q)
            } else {
                Problem::maximize(q)
            };
            let solution = solve(problem.build(), 9.0 * sign);
            assert!(solution.primal.as_ref().unwrap().is_empty());
        }
    }
}

#[test]
fn sparse_identity_and_one_by_one_matrices() {
    for n in [1, 2] {
        for sign in [1.0, -1.0] {
            let p = constant_sparse(dense_to_csc(&(DMatrix::identity(n, n) * sign)));
            let x = variable(n);
            let q = quad_form(&x, &p);
            let problem = if sign > 0.0 {
                Problem::minimize(q.clone())
            } else {
                Problem::maximize(q.clone())
            };
            solve(problem.constraint(x.ge(1.0)).build(), n as f64 * sign);

            let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
            solve(
                Problem::maximize(sum(&x)).constraint(bound).build(),
                (n as f64).sqrt(),
            );
        }
    }
}

#[test]
fn small_positive_directions_remain_active() {
    for p in representations(diagonal(&[1.0, 1e-18, 0.0])) {
        let x = variable(());
        let z = vstack(vec![constant(0.0), 1e9 * &x, constant(0.0)]);
        let q = quad_form(&z, &p);
        let solution = solve(
            Problem::minimize(&q - 2.0 * &x)
                .subject_to([x.ge(0.0), x.le(2.0)])
                .build(),
            -1.0,
        );
        assert_abs_diff_eq!(solution.value(&x), 1.0, epsilon = 1e-5);
        let solution = solve(
            Problem::maximize(x.clone()).constraint(q.le(1.0)).build(),
            1.0,
        );
        assert_abs_diff_eq!(q.value(&solution).as_scalar().unwrap(), 1.0, epsilon = 1e-5);
    }
}

#[test]
fn roundoff_projection_agrees_in_native_and_cone_forms() {
    for sign in [1.0, -1.0] {
        for p in representations(diagonal(&[sign, -sign * 1e-15])) {
            let z = constant_vec(vec![0.0, 1e8]);
            let q = 100.0 * quad_form(&z, &p);
            let native = if sign > 0.0 {
                Problem::minimize(q.clone())
            } else {
                Problem::maximize(q.clone())
            };
            let cone = if sign > 0.0 {
                Problem::minimize(max2(&q, &constant(-1.0)))
            } else {
                Problem::maximize(min2(&q, &constant(1.0)))
            };
            for problem in [native.build(), cone.build()] {
                assert!(problem.is_dcp());
                let solution = problem.solve().unwrap();
                let reported = solution.value.unwrap();
                assert_abs_diff_eq!(reported, 0.0, epsilon = 1e-5);
                let evaluated = q.value(&solution).as_scalar().unwrap();
                assert_abs_diff_eq!(evaluated, -sign * 1000.0, epsilon = 1e-8);
                let error_bound = 16.0 * 2.0 * f64::EPSILON * 100.0 * 1e16;
                assert!((reported - evaluated).abs() <= error_bound);
            }
        }
    }
}

#[test]
fn definiteness_tolerance_scales_with_the_matrix() {
    for scale in [1e-12, 1.0, 1e12] {
        for sign in [1.0, -1.0] {
            for (negative, accepted) in [(1e-15, true), (1e-12, false)] {
                let matrix = diagonal(&[sign * scale, -sign * scale * negative]);
                for p in representations(matrix) {
                    let x = variable(2);
                    let q = quad_form(&x, &p);
                    let problem = if sign > 0.0 {
                        Problem::minimize(q)
                    } else {
                        Problem::maximize(q)
                    }
                    .build();
                    assert_eq!(problem.is_dcp(), accepted);
                    if !accepted {
                        assert!(matches!(problem.solve(), Err(CvxError::NotDcp(_))));
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_matrices_are_rejected_before_solving() {
    let x = variable(2);
    let mut matrices = vec![variable((2, 2))];
    for matrix in [
        diagonal(&[1.0, -1.0]),
        DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.0, 1.0]),
        diagonal(&[1.0, f64::NAN]),
        diagonal(&[1.0, f64::INFINITY]),
    ] {
        matrices.extend(representations(matrix));
    }
    for p in matrices {
        let q = quad_form(&x, &p);
        for problem in [Problem::minimize(q.clone()), Problem::maximize(q)] {
            let problem = problem.build();
            assert!(!problem.is_dcp());
            assert!(matches!(problem.solve(), Err(CvxError::NotDcp(_))));
        }
    }
}
