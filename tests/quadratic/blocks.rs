/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

use approx::assert_abs_diff_eq;
use cvxrust::canon::{ConeConstraint, VariableDomains, canonicalize};
use cvxrust::dcp::PsdStatus;
use cvxrust::prelude::*;
use cvxrust::sparse::{csc_to_dense, dense_to_csc};
use nalgebra::DMatrix;

fn representations(matrix: DMatrix<f64>) -> [Expr; 2] {
    [
        constant_sparse(dense_to_csc(&matrix)),
        constant_dmatrix(matrix),
    ]
}

fn assert_mapped_quadratic(p: &Expr, map: &DMatrix<f64>, expected: &DMatrix<f64>, sign: f64) {
    let x = variable(map.ncols());
    let id = x.variable_id().unwrap();
    let q = quad_form(&matmul(&constant_dmatrix(map.clone()), &x), p);
    let native = canonicalize(&q, true, &mut VariableDomains::default())
        .expr
        .into_quadratic();
    assert_abs_diff_eq!(
        csc_to_dense(&native.quad_coeffs[&(id, id)]),
        expected,
        epsilon = 1e-12
    );
    let cone = canonicalize(&q, false, &mut VariableDomains::default());
    let soc = cone
        .constraints
        .iter()
        .find_map(|constraint| match constraint {
            ConeConstraint::SOC { x, .. } => Some(x),
            _ => None,
        })
        .unwrap();
    let coefficients = csc_to_dense(&soc.coeffs[&id]);
    let factor = coefficients.rows(1, coefficients.nrows() - 1) * 0.5;
    assert_abs_diff_eq!(
        factor.transpose() * factor * sign,
        expected,
        epsilon = 1e-12
    );
}

fn two_blocks(small: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(4, 4, |i, j| match (i < 2, j < 2) {
        (true, true) => 1.0,
        (false, false) => small[(i - 2, j - 2)],
        _ => 0.0,
    })
}

fn small_block_map(scale: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        4,
        2,
        &[
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / scale.sqrt(),
            0.0,
            0.0,
            1.0 / scale.sqrt(),
        ],
    )
}

#[test]
fn disconnected_and_interleaved_blocks_preserve_quadratics() {
    let small = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let matrix = two_blocks(&(&small * 1e-16));
    let map = small_block_map(1e-16);
    for order in [[0, 1, 2, 3], [0, 2, 1, 3]] {
        let permuted = DMatrix::from_fn(4, 4, |i, j| matrix[(order[i], order[j])]);
        let map = DMatrix::from_fn(4, 2, |i, j| map[(order[i], j)]);
        for sign in [1.0, -1.0] {
            for p in representations(&permuted * sign) {
                assert_mapped_quadratic(&p, &map, &(&small * sign), sign);

                let y = variable(());
                let z = matmul(&constant_dmatrix(map.columns(0, 1).into_owned()), &y);
                let q = quad_form(&z, &p);
                let bound = if sign > 0.0 { q.le(2.0) } else { q.ge(-2.0) };
                let solution = Problem::maximize(y.clone())
                    .constraint(bound)
                    .build()
                    .solve()
                    .unwrap();
                assert_eq!(solution.status, SolveStatus::Optimal);
                assert_abs_diff_eq!(solution.value(&y), 1.0, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn scalar_blocks_survive_global_normalization_underflow() {
    for small in [1e-18, f64::from_bits(1)] {
        let matrix = DMatrix::from_row_slice(
            3,
            3,
            &[1e308, 1e308, 0.0, 1e308, 1e308, 0.0, 0.0, 0.0, small],
        );
        let map = DMatrix::from_column_slice(3, 1, &[0.0, 0.0, 1.0 / small.sqrt()]);
        for sign in [1.0, -1.0] {
            for p in representations(&matrix * sign) {
                assert_mapped_quadratic(&p, &map, &DMatrix::from_element(1, 1, sign), sign);
            }
        }
    }
}

#[test]
fn connected_small_positive_directions_survive_permutations() {
    for (large, small) in [(1.0, 1e-16_f64), (1.0, 1e-18), (3.0, 1e-18), (1e308, 1e-18)] {
        let matrix = DMatrix::from_row_slice(
            3,
            3,
            &[
                large, small, large, small, small, small, large, small, large,
            ],
        );
        let map = DMatrix::from_column_slice(3, 1, &[0.0, 1.0 / small.sqrt(), 0.0]);
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let permuted = DMatrix::from_fn(3, 3, |i, j| matrix[(order[i], order[j])]);
            let map = DMatrix::from_fn(3, 1, |i, j| map[(order[i], j)]);
            for sign in [1.0, -1.0] {
                for p in representations(&permuted * sign) {
                    assert_eq!(
                        PsdStatus::of_array(p.constant_value().unwrap()),
                        if sign > 0.0 {
                            PsdStatus::Psd
                        } else {
                            PsdStatus::Nsd
                        },
                        "large={large}, small={small}, order={order:?}"
                    );
                    assert_mapped_quadratic(&p, &map, &DMatrix::from_element(1, 1, sign), sign);
                    let y = variable(());
                    let q = quad_form(&matmul(&constant_dmatrix(map.clone()), &y), &p);
                    let bound = if sign > 0.0 { q.le(1.0) } else { q.ge(-1.0) };
                    let solution = Problem::maximize(y.clone())
                        .constraint(bound)
                        .build()
                        .solve()
                        .unwrap();
                    assert_eq!(solution.status, SolveStatus::Optimal);
                    assert_abs_diff_eq!(solution.value(&y), 1.0, epsilon = 1e-6);
                }
            }
        }
    }
}

#[test]
fn weakly_connected_small_blocks_retain_all_positive_directions() {
    let small_block = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    for (large, small) in [(1.0, 1e-18_f64), (1e308, 1e-18)] {
        let matrix = DMatrix::from_fn(4, 4, |i, j| match (i < 2, j < 2) {
            (true, true) => large,
            (false, false) => small * small_block[(i - 2, j - 2)],
            _ => small,
        });
        let map = small_block_map(small);
        for order in [[0, 1, 2, 3], [0, 2, 1, 3], [2, 0, 3, 1], [3, 2, 1, 0]] {
            let permuted = DMatrix::from_fn(4, 4, |i, j| matrix[(order[i], order[j])]);
            let map = DMatrix::from_fn(4, 2, |i, j| map[(order[i], j)]);
            for sign in [1.0, -1.0] {
                for p in representations(&permuted * sign) {
                    assert_mapped_quadratic(&p, &map, &(&small_block * sign), sign);
                }
            }
        }
    }
}

#[test]
fn mixed_sign_blocks_use_the_global_spectral_tolerance() {
    // The large block has spectral radius 2; the full matrix has dimension 4.
    let tolerance = 16.0 * 4.0 * f64::EPSILON * 2.0;
    for sign in [1.0, -1.0] {
        for (small, accepted) in [(0.75 * tolerance, true), (1.25 * tolerance, false)] {
            let matrix = two_blocks(&DMatrix::from_row_slice(2, 2, &[0.0, small, small, 0.0]));
            for p in representations(matrix * sign) {
                let status = PsdStatus::of_array(p.constant_value().unwrap());
                if accepted {
                    assert_eq!(
                        status,
                        if sign > 0.0 {
                            PsdStatus::Psd
                        } else {
                            PsdStatus::Nsd
                        }
                    );
                    // Clip the wrong-sign eigenvalue, retaining the small correctly
                    // signed eigenvalue despite its size relative to the large block.
                    assert_mapped_quadratic(
                        &p,
                        &small_block_map(small),
                        &DMatrix::from_element(2, 2, sign * 0.5),
                        sign,
                    );
                } else {
                    assert_eq!(status, PsdStatus::Neither);
                    assert!(
                        !Problem::minimize(quad_form(&variable(4), &p))
                            .build()
                            .is_dcp()
                    );
                }
            }
        }
    }
}

#[test]
fn entirely_wrong_sign_blocks_can_be_global_roundoff() {
    let small = 1e-15;
    for sign in [1.0, -1.0] {
        let matrix = two_blocks(&DMatrix::from_row_slice(
            2,
            2,
            &[-2.0 * small, -small, -small, -2.0 * small],
        ));
        for p in representations(matrix * sign) {
            for map in [small_block_map(small), small_block_map(1e-32)] {
                assert_mapped_quadratic(&p, &map, &DMatrix::zeros(2, 2), sign);
            }
        }
        let matrix =
            DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -small]);
        for p in representations(matrix * sign) {
            let map = DMatrix::from_column_slice(3, 1, &[0.0, 0.0, 1.0 / small.sqrt()]);
            assert_mapped_quadratic(&p, &map, &DMatrix::zeros(1, 1), sign);
        }
    }
}
