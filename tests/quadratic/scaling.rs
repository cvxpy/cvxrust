/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use approx::assert_abs_diff_eq;
use cvxrust::prelude::*;

#[test]
fn scaled_quadratics_preserve_small_quadratic_coefficients() {
    for use_quad_form in [true, false] {
        let x = variable(());
        let argument = 1e-8 * &x;
        let quadratic = if use_quad_form {
            quad_form(&argument, &constant(1.0))
        } else {
            sum_squares(&argument)
        };
        let objective = 1e16 * quadratic - 2.0 * &x;
        let problem = Problem::minimize(objective.clone())
            .subject_to([x.ge(0.0), x.le(2.0)])
            .build();
        let solution = problem.solve().unwrap();

        assert_eq!(solution.status, SolveStatus::Optimal);
        assert_abs_diff_eq!(solution.value(&x), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(solution.value.unwrap(), -1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(
            objective.value(&solution).as_scalar().unwrap(),
            -1.0,
            epsilon = 1e-8
        );
    }
}

#[test]
fn scaled_quadratics_preserve_small_linear_coefficients() {
    for use_quad_form in [true, false] {
        let x = variable(());
        let argument = 1e-7 * &x - constant(1e-10);
        let quadratic = if use_quad_form {
            quad_form(&argument, &constant(1.0))
        } else {
            sum_squares(&argument)
        };
        let objective = 1e14 * quadratic;
        let problem = Problem::minimize(objective.clone()).build();
        let solution = problem.solve().unwrap();

        assert_eq!(solution.status, SolveStatus::Optimal);
        assert_abs_diff_eq!(solution.value(&x), 1e-3, epsilon = 1e-8);
        assert_abs_diff_eq!(solution.value.unwrap(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(
            objective.value(&solution).as_scalar().unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }
}
