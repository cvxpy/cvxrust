/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use cvxrust::canon::{VariableDomains, canonicalize};
use cvxrust::prelude::*;

#[test]
fn quad_form_rejects_nearly_symmetric_indefinite_matrices() {
    for sign in [1.0, -1.0] {
        let x = variable(2);
        // The lower triangle is definite, but the symmetric part is indefinite.
        let p = constant_matrix(
            vec![sign, sign, sign * (1.0 + 9e-11), sign * (1.0 + 1e-11)],
            2,
            2,
        );
        let q = quad_form(&x, &p);
        let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
        for problem in [
            Problem::minimize(q.clone()).build(),
            Problem::maximize(q).build(),
            Problem::maximize(sum(&x)).constraint(bound).build(),
        ] {
            assert!(!problem.is_dcp());
            assert!(matches!(problem.solve(), Err(CvxError::NotDcp(_))));
        }
    }
}

#[test]
fn quad_form_accepts_nearly_symmetric_definite_matrices() {
    let off_diagonal = 1.0 + 4.5e-11;
    for sign in [1.0, -1.0] {
        let x = variable(2);
        let p = constant_matrix(
            vec![2.0 * sign, sign, sign * (1.0 + 9e-11), 2.0 * sign],
            2,
            2,
        );
        let q = quad_form(&x, &p);
        let objective = if sign > 0.0 {
            Problem::minimize(q.clone())
        } else {
            Problem::maximize(q.clone())
        };
        let problem = objective.constraint(sum(&x).eq(1.0)).build();
        assert!(problem.is_dcp());
        let solution = problem.solve().unwrap();
        let expected = sign * (2.0 + off_diagonal) / 2.0;
        assert!((solution.value.unwrap() - expected).abs() < 1e-6);
        assert!((q.value(&solution).as_scalar().unwrap() - expected).abs() < 1e-6);

        let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
        let problem = Problem::maximize(sum(&x)).constraint(bound).build();
        assert!(problem.is_dcp());
        let solution = problem.solve().unwrap();
        let expected = (2.0 / (2.0 + off_diagonal)).sqrt();
        assert!((solution.value.unwrap() - expected).abs() < 1e-6);
        assert!((q.value(&solution).as_scalar().unwrap() - sign).abs() < 1e-6);
    }
}

#[test]
fn quad_form_preserves_subnormal_diagonal_during_symmetrization() {
    for sign in [1.0, -1.0] {
        let x = variable(());
        let p = constant_matrix(vec![sign * f64::from_bits(1)], 1, 1);
        let q = quad_form(&x, &p);
        let problem = if sign > 0.0 {
            Problem::minimize(q.clone()).build()
        } else {
            Problem::maximize(q.clone()).build()
        };
        assert!(problem.is_dcp());
        // The supported nonzero diagonal must still admit a cone factorization.
        canonicalize(&q, false, &mut VariableDomains::default());
    }
}
