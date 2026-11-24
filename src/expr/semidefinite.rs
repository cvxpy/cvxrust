/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Shared classification and factors for symmetric quadratic forms.

use nalgebra::DMatrix;

use super::{Array, symmetric_part};
use crate::sparse::csc_to_dense;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Definiteness {
    Zero,
    Psd,
    Nsd,
    Indefinite,
}

pub(crate) struct Analysis {
    pub(crate) kind: Definiteness,
    pub(crate) matrix: DMatrix<f64>,
    pub(crate) factor: Option<DMatrix<f64>>,
}

fn relative_tolerance(n: usize) -> f64 {
    16.0 * n as f64 * f64::EPSILON
}

fn classify(min: f64, max: f64, tolerance: f64) -> Definiteness {
    if min >= -tolerance {
        Definiteness::Psd
    } else if max <= tolerance {
        Definiteness::Nsd
    } else {
        Definiteness::Indefinite
    }
}

pub(crate) fn analyze(array: &Array) -> Option<Analysis> {
    let matrix = match array {
        Array::Scalar(value) => DMatrix::from_element(1, 1, *value),
        Array::Dense(matrix) => matrix.clone(),
        Array::Sparse(matrix) => csc_to_dense(matrix),
    };
    if !matrix.is_square() || !matrix.iter().all(|value| value.is_finite()) {
        return None;
    }
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-10 {
                return None;
            }
        }
    }
    let matrix = symmetric_part(&matrix);
    let scale = matrix
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if scale == 0.0 {
        return Some(Analysis {
            kind: Definiteness::Zero,
            matrix,
            factor: None,
        });
    }
    if (0..n).all(|i| ((i + 1)..n).all(|j| matrix[(i, j)] == 0.0)) {
        return Some(analyze_diagonal(matrix, scale));
    }
    for (kind, sign) in [(Definiteness::Psd, 1.0), (Definiteness::Nsd, -1.0)] {
        if let Some(cholesky) = (&matrix * sign).cholesky() {
            let factor = cholesky.l().transpose();
            if factor.iter().all(|value| value.is_finite()) {
                return Some(Analysis {
                    kind,
                    matrix,
                    factor: Some(factor),
                });
            }
        }
    }
    analyze_eigen(matrix, scale, n.saturating_mul(100))
}

fn analyze_diagonal(mut matrix: DMatrix<f64>, scale: f64) -> Analysis {
    let n = matrix.nrows();
    let min = (0..n)
        .map(|i| matrix[(i, i)] / scale)
        .fold(f64::INFINITY, f64::min);
    let max = (0..n)
        .map(|i| matrix[(i, i)] / scale)
        .fold(f64::NEG_INFINITY, f64::max);
    let kind = classify(min, max, relative_tolerance(n));
    let factor = if kind == Definiteness::Indefinite {
        None
    } else {
        let sign = if kind == Definiteness::Psd { 1.0 } else { -1.0 };
        let mut factor = DMatrix::zeros(n, n);
        for i in 0..n {
            if sign * matrix[(i, i)] < 0.0 {
                matrix[(i, i)] = 0.0;
            }
            // Keep original coefficients: normalization can underflow small diagonals.
            factor[(i, i)] = matrix[(i, i)].abs().sqrt();
        }
        Some(factor)
    };
    Analysis {
        kind,
        matrix,
        factor,
    }
}

fn analyze_eigen(mut matrix: DMatrix<f64>, scale: f64, max_iterations: usize) -> Option<Analysis> {
    let n = matrix.nrows();
    let normalized = matrix.map(|value| value / scale);
    let eigen = normalized.try_symmetric_eigen(f64::EPSILON, max_iterations)?;
    if !eigen
        .eigenvalues
        .iter()
        .chain(eigen.eigenvectors.iter())
        .all(|value| value.is_finite())
    {
        return None;
    }
    let min = eigen.eigenvalues.min();
    let max = eigen.eigenvalues.max();
    let tolerance = relative_tolerance(n) * min.abs().max(max.abs());
    let kind = classify(min, max, tolerance);
    if kind == Definiteness::Indefinite {
        return Some(Analysis {
            kind,
            matrix,
            factor: None,
        });
    }
    let sign = if kind == Definiteness::Psd { 1.0 } else { -1.0 };
    let root_scale = scale.sqrt();
    let mut factor = DMatrix::zeros(n, n);
    for k in 0..n {
        let value = eigen.eigenvalues[k];
        if sign * value < 0.0 {
            // Correct only clipped eigenpairs, retaining the other matrix entries.
            for i in 0..n {
                for j in i..n {
                    let correction =
                        value * eigen.eigenvectors[(i, k)] * eigen.eigenvectors[(j, k)] * scale;
                    let corrected = matrix[(i, j)] - correction;
                    matrix[(i, j)] = corrected;
                    matrix[(j, i)] = corrected;
                }
            }
        } else {
            let root = root_scale * value.abs().sqrt();
            for j in 0..n {
                factor[(k, j)] = root * eigen.eigenvectors[(j, k)];
            }
        }
    }
    if !matrix
        .iter()
        .chain(factor.iter())
        .all(|value| value.is_finite())
    {
        return None;
    }
    let normalized_factor = factor.map(|value| value / root_scale);
    let residual = normalized_factor.transpose() * normalized_factor * sign
        - matrix.map(|value| value / scale);
    if !residual
        .iter()
        .all(|value| value.is_finite() && value.abs() <= 4.0 * tolerance)
    {
        return None;
    }
    Some(Analysis {
        kind,
        matrix,
        factor: Some(factor),
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::sparse::dense_to_csc;

    fn check_factor(analysis: &Analysis) {
        let sign = match analysis.kind {
            Definiteness::Psd => 1.0,
            Definiteness::Nsd => -1.0,
            _ => {
                assert!(analysis.factor.is_none());
                return;
            }
        };
        let factor = analysis.factor.as_ref().unwrap();
        assert_eq!(factor.shape(), analysis.matrix.shape());
        let scale = analysis.matrix.amax();
        let normalized = factor.map(|value| value / scale.sqrt());
        let residual =
            normalized.transpose() * normalized * sign - analysis.matrix.map(|value| value / scale);
        assert!(
            residual.amax() < 1e-12,
            "factor residual: {}",
            residual.amax()
        );
    }

    #[test]
    fn classify_dense_and_sparse_semidefinite_matrices() {
        for (matrix, expected) in [
            (DMatrix::identity(3, 3), Definiteness::Psd),
            (
                DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 0.0, 0.0])),
                Definiteness::Psd,
            ),
            (
                DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]),
                Definiteness::Psd,
            ),
            (
                DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]),
                Definiteness::Indefinite,
            ),
            (DMatrix::zeros(3, 3), Definiteness::Zero),
            (DMatrix::zeros(0, 0), Definiteness::Zero),
        ] {
            for sign in [1.0, -1.0] {
                let signed = &matrix * sign;
                let expected = if sign < 0.0 && expected == Definiteness::Psd {
                    Definiteness::Nsd
                } else {
                    expected
                };
                for array in [
                    Array::Dense(signed.clone()),
                    Array::Sparse(dense_to_csc(&signed)),
                ] {
                    let analysis = analyze(&array).unwrap();
                    assert_eq!(analysis.kind, expected);
                    check_factor(&analysis);
                }
            }
        }
    }

    #[test]
    fn reject_invalid_matrices() {
        for matrix in [
            DMatrix::zeros(2, 3),
            DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, f64::NAN, 1.0]),
            DMatrix::from_element(1, 1, f64::INFINITY),
        ] {
            assert!(analyze(&Array::Sparse(dense_to_csc(&matrix))).is_none());
            assert!(analyze(&Array::Dense(matrix)).is_none());
        }
        for scalar in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(analyze(&Array::Scalar(scalar)).is_none());
        }
    }

    #[test]
    fn scalar_dense_and_sparse_classification_agree() {
        for (value, expected) in [
            (2.0, Definiteness::Psd),
            (-2.0, Definiteness::Nsd),
            (0.0, Definiteness::Zero),
        ] {
            let matrix = DMatrix::from_element(1, 1, value);
            for array in [
                Array::Scalar(value),
                Array::Dense(matrix.clone()),
                Array::Sparse(dense_to_csc(&matrix)),
            ] {
                let analysis = analyze(&array).unwrap();
                assert_eq!(analysis.kind, expected);
                check_factor(&analysis);
            }
        }
    }

    #[test]
    fn diagonal_tolerance_is_relative_and_corrects_only_wrong_signs() {
        for scale in [1e-200, 1.0, 1e200] {
            for sign in [1.0, -1.0] {
                for (negative, expected) in [
                    (
                        1e-15,
                        if sign > 0.0 {
                            Definiteness::Psd
                        } else {
                            Definiteness::Nsd
                        },
                    ),
                    (1e-12, Definiteness::Indefinite),
                ] {
                    let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
                        sign * scale,
                        -sign * negative * scale,
                    ]));
                    let analysis = analyze(&Array::Dense(matrix)).unwrap();
                    assert_eq!(analysis.kind, expected);
                    assert_eq!(analysis.matrix[(0, 0)], sign * scale);
                    if expected != Definiteness::Indefinite {
                        assert_eq!(analysis.matrix[(1, 1)], 0.0);
                    }
                    check_factor(&analysis);
                }
            }
        }
    }

    #[test]
    fn eigen_tolerance_is_relative_and_preserves_native_cone_agreement() {
        for scale in [1e-100, 1.0, 1e100] {
            for sign in [1.0, -1.0] {
                for negative in [1e-15, 1e-12] {
                    let matrix = DMatrix::from_row_slice(
                        2,
                        2,
                        &[
                            0.5 - negative / 2.0,
                            0.5 + negative / 2.0,
                            0.5 + negative / 2.0,
                            0.5 - negative / 2.0,
                        ],
                    ) * (scale * sign);
                    let analysis = analyze(&Array::Dense(matrix)).unwrap();
                    let expected = if negative == 1e-12 {
                        Definiteness::Indefinite
                    } else if sign > 0.0 {
                        Definiteness::Psd
                    } else {
                        Definiteness::Nsd
                    };
                    assert_eq!(analysis.kind, expected);
                    check_factor(&analysis);
                }
            }
        }
    }

    #[test]
    fn retain_small_positive_diagonal_directions() {
        for sign in [1.0, -1.0] {
            for diagonal in [vec![1.0, 1e-18, 0.0], vec![1e308, f64::from_bits(1), 0.0]] {
                let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(diagonal)) * sign;
                let analysis = analyze(&Array::Dense(matrix.clone())).unwrap();
                assert_eq!(analysis.matrix, matrix);
                let factor = analysis.factor.unwrap();
                assert!(factor[(1, 1)] > 0.0);
                assert_abs_diff_eq!(
                    factor[(1, 1)].powi(2) / matrix[(1, 1)].abs(),
                    1.0,
                    epsilon = 1e-15
                );
            }
            let matrix =
                DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 1e-18, 0.0])) * sign;
            let analysis = analyze(&Array::Dense(matrix)).unwrap();
            let x = nalgebra::DVector::from_vec(vec![0.0, 1e9, 0.0]);
            assert_abs_diff_eq!(
                (analysis.factor.unwrap() * x).norm_squared(),
                1.0,
                epsilon = 1e-15
            );
        }
    }

    #[test]
    fn factor_rank_deficient_gram_matrices() {
        for n in [2, 3, 5, 12] {
            let x = nalgebra::DVector::from_fn(n, |i, _| (i % 5) as f64 - 2.0);
            for rank in [1, n - 1, n] {
                let a = DMatrix::from_fn(rank, n, |i, j| {
                    if i == j {
                        2.0
                    } else {
                        ((((i + 1) * 3 + (j + 1) * 5) % 11) as f64 - 5.0) / (4 * n) as f64
                    }
                });
                let expected = (&a * &x).norm_squared();
                for scale in [1e-100, 1.0, 1e100] {
                    for sign in [1.0, -1.0] {
                        let original = a.transpose() * &a * (sign * scale);
                        let analysis = analyze(&Array::Dense(original.clone())).unwrap();
                        assert_eq!(
                            analysis.kind,
                            if sign > 0.0 {
                                Definiteness::Psd
                            } else {
                                Definiteness::Nsd
                            }
                        );
                        assert!((&analysis.matrix - &original).amax() / original.amax() < 1e-12);
                        check_factor(&analysis);
                        let value = sign * (analysis.factor.unwrap() * &x).norm_squared() / scale;
                        assert_abs_diff_eq!(value, sign * expected, epsilon = 1e-12 * expected);
                    }
                }
            }
        }
    }

    #[test]
    fn use_symmetric_part_for_classification_and_factors() {
        let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 1.0 + 9e-11, 1.0, 1.0 + 1e-11]);
        assert_eq!(
            analyze(&Array::Dense(indefinite)).unwrap().kind,
            Definiteness::Indefinite
        );
        let definite = DMatrix::from_row_slice(2, 2, &[2.0, 1.0 + 9e-11, 1.0, 2.0]);
        let analysis = analyze(&Array::Dense(definite.clone())).unwrap();
        assert_eq!(analysis.matrix, symmetric_part(&definite));
        check_factor(&analysis);
    }

    #[test]
    fn failed_eigendecomposition_returns_no_analysis() {
        let matrix =
            DMatrix::from_row_slice(3, 3, &[2.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 4.0]);
        assert!(analyze_eigen(matrix, 4.0, 1).is_none());
    }
}
