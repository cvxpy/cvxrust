/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Shared classification and factors for symmetric quadratic forms.

use nalgebra::{DMatrix, DVector};

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
        if let Some(factor) = balanced_psd_factor(&matrix, sign) {
            return Some(Analysis {
                kind,
                matrix,
                factor: Some(factor),
            });
        }
        if let Some(cholesky) = (&matrix * sign).cholesky() {
            let factor = cholesky.l().transpose();
            if factor_matches(&matrix, sign, &factor) {
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

fn diagonal_scales(matrix: &DMatrix<f64>, sign: f64) -> Option<DVector<f64>> {
    let n = matrix.nrows();
    let mut scales = DVector::zeros(n);
    for i in 0..n {
        let diagonal = sign * matrix[(i, i)];
        if !diagonal.is_finite() || diagonal < 0.0 {
            return None;
        }
        if diagonal == 0.0 && (0..n).any(|j| matrix[(i, j)] != 0.0) {
            return None;
        }
        scales[i] = diagonal.sqrt();
    }
    Some(scales)
}

fn equilibrate(matrix: &DMatrix<f64>, sign: f64, scales: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        if scales[i] == 0.0 || scales[j] == 0.0 {
            0.0
        } else {
            // Sequential division avoids overflow or underflow in the product.
            let (i, j) = (i.min(j), i.max(j));
            sign * matrix[(i, j)] / scales[i] / scales[j]
        }
    })
}

fn factor_matches(matrix: &DMatrix<f64>, sign: f64, factor: &DMatrix<f64>) -> bool {
    let Some(scales) = diagonal_scales(matrix, sign) else {
        return false;
    };
    if (0..matrix.ncols())
        .any(|j| scales[j] == 0.0 && factor.column(j).iter().any(|value| *value != 0.0))
    {
        return false;
    }
    let balanced = equilibrate(matrix, sign, &scales);
    let normalized = DMatrix::from_fn(factor.nrows(), factor.ncols(), |i, j| {
        if scales[j] == 0.0 {
            factor[(i, j)]
        } else {
            factor[(i, j)] / scales[j]
        }
    });
    let residual = normalized.transpose() * normalized - balanced;
    factor.iter().all(|value| value.is_finite())
        && residual.iter().all(|value| {
            value.is_finite() && value.abs() <= 4.0 * relative_tolerance(matrix.nrows())
        })
}

fn balanced_psd_factor(matrix: &DMatrix<f64>, sign: f64) -> Option<DMatrix<f64>> {
    let n = matrix.nrows();
    let scales = diagonal_scales(matrix, sign)?;
    let mut residual = equilibrate(matrix, sign, &scales);
    if !residual.iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut order: Vec<_> = (0..n).collect();
    let mut factor = DMatrix::zeros(n, n);
    for k in 0..n {
        let pivot_position = (k..n).max_by(|&i, &j| {
            residual[(order[i], order[i])].total_cmp(&residual[(order[j], order[j])])
        })?;
        order.swap(k, pivot_position);
        let pivot = order[k];
        let diagonal = residual[(pivot, pivot)];
        if diagonal < 0.0 {
            return None;
        }
        if diagonal == 0.0 {
            // A zero diagonal permits no nonzero entries in the remaining block.
            if order[k..]
                .iter()
                .any(|&i| order[k..].iter().any(|&j| residual[(i, j)] != 0.0))
            {
                return None;
            }
            break;
        }
        // Keep every positive pivot, including values below roundoff tolerance.
        let root = diagonal.sqrt();
        factor[(k, pivot)] = root;
        for &j in &order[(k + 1)..] {
            factor[(k, j)] = root * (residual[(pivot, j)] / diagonal);
        }
        // LDL^T updates avoid square-root roundoff in singular Schur blocks.
        for (position, &i) in order[(k + 1)..].iter().enumerate() {
            for &j in &order[(k + 1 + position)..] {
                let (larger, smaller) = if residual[(i, pivot)].abs() >= residual[(j, pivot)].abs()
                {
                    (residual[(i, pivot)], residual[(j, pivot)])
                } else {
                    (residual[(j, pivot)], residual[(i, pivot)])
                };
                let value = (-larger / diagonal).mul_add(smaller, residual[(i, j)]);
                residual[(i, j)] = value;
                residual[(j, i)] = value;
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            factor[(i, j)] *= scales[j];
        }
    }
    factor_matches(matrix, sign, &factor).then_some(factor)
}

struct SpectralBlock {
    indices: Vec<usize>,
    scale: f64,
    values: DVector<f64>,
    vectors: DMatrix<f64>,
}

fn spectral_blocks(matrix: &DMatrix<f64>, max_iterations: usize) -> Option<Vec<SpectralBlock>> {
    let n = matrix.nrows();
    let mut visited = vec![false; n];
    let mut blocks = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut indices = vec![start];
        visited[start] = true;
        let mut cursor = 0;
        while cursor < indices.len() {
            let i = indices[cursor];
            for j in 0..n {
                if !visited[j] && matrix[(i, j)] != 0.0 {
                    visited[j] = true;
                    indices.push(j);
                }
            }
            cursor += 1;
        }
        let block = DMatrix::from_fn(indices.len(), indices.len(), |i, j| {
            matrix[(indices[i], indices[j])]
        });
        let scale = block.amax();
        let (values, vectors) = if indices.len() == 1 {
            (
                DVector::from_element(
                    1,
                    if scale == 0.0 {
                        0.0
                    } else {
                        block[(0, 0)] / scale
                    },
                ),
                DMatrix::identity(1, 1),
            )
        } else {
            // Disconnected blocks can have very different coefficient scales.
            let eigen = block
                .map(|value| value / scale)
                .try_symmetric_eigen(f64::EPSILON, max_iterations)?;
            (eigen.eigenvalues, eigen.eigenvectors)
        };
        if !values
            .iter()
            .chain(vectors.iter())
            .all(|value| value.is_finite())
        {
            return None;
        }
        blocks.push(SpectralBlock {
            indices,
            scale,
            values,
            vectors,
        });
    }
    Some(blocks)
}

fn analyze_eigen(mut matrix: DMatrix<f64>, scale: f64, max_iterations: usize) -> Option<Analysis> {
    let n = matrix.nrows();
    let blocks = spectral_blocks(&matrix, max_iterations)?;
    // Classification uses the spectral radius of the whole matrix. Ratios avoid
    // overflowing eigenvalues when a block's spectral radius exceeds f64::MAX.
    let (min, max) = blocks
        .iter()
        .flat_map(|block| {
            block
                .values
                .iter()
                .map(|value| value * (block.scale / scale))
        })
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        });
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
    let mut factor = DMatrix::zeros(n, n);
    for block in blocks {
        let indices = &block.indices;
        let size = indices.len();
        if size == 1 {
            let i = indices[0];
            if sign * matrix[(i, i)] < 0.0 {
                matrix[(i, i)] = 0.0;
            }
            factor[(i, i)] = matrix[(i, i)].abs().sqrt();
            continue;
        }
        if block.values.iter().all(|value| sign * value <= 0.0) {
            for &i in indices {
                for &j in indices {
                    matrix[(i, j)] = 0.0;
                }
            }
            continue;
        }
        let root_scale = block.scale.sqrt();
        let mut block_factor = DMatrix::zeros(size, size);
        for k in 0..size {
            let value = block.values[k];
            if sign * value < 0.0 {
                // Correct only clipped eigenpairs, retaining all other coefficients.
                for i in 0..size {
                    for j in i..size {
                        let correction =
                            value * block.vectors[(i, k)] * block.vectors[(j, k)] * block.scale;
                        let corrected = matrix[(indices[i], indices[j])] - correction;
                        matrix[(indices[i], indices[j])] = corrected;
                        matrix[(indices[j], indices[i])] = corrected;
                    }
                }
            } else {
                let root = root_scale * value.abs().sqrt();
                for j in 0..size {
                    block_factor[(k, j)] = root * block.vectors[(j, k)];
                }
            }
        }
        let effective = DMatrix::from_fn(size, size, |i, j| matrix[(indices[i], indices[j])]);
        if let Some(balanced_factor) = balanced_psd_factor(&effective, sign) {
            block_factor = balanced_factor;
        }
        // Check in diagonal-scaled coordinates so small connected directions
        // receive the same relative accuracy as large ones.
        if !factor_matches(&effective, sign, &block_factor) {
            return None;
        }
        for i in 0..size {
            for j in 0..size {
                factor[(indices[i], indices[j])] = block_factor[(i, j)];
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
    fn strict_factor_rejects_negative_pivots_and_nonzero_null_rows() {
        for matrix in [
            DMatrix::from_element(1, 1, -1e-18),
            DMatrix::from_row_slice(2, 2, &[1.0, 1.0 + 1e-15, 1.0 + 1e-15, 1.0]),
            DMatrix::from_row_slice(
                3,
                3,
                &[1.0, 1.0 - 1e-15, 1.0, 1.0 - 1e-15, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            DMatrix::from_row_slice(2, 2, &[1.0, 1e-18, 1e-18, 0.0]),
        ] {
            for sign in [1.0, -1.0] {
                assert!(balanced_psd_factor(&(&matrix * sign), sign).is_none());
            }
        }
    }

    #[test]
    fn factor_reconstruction_checks_small_and_zero_diagonals() {
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1e-18, 0.0]));
        let valid = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1e-9, 0.0]));
        let missing = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.0, 0.0]));
        let nonzero_null = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1e-9, 1e-200]));
        for sign in [1.0, -1.0] {
            let signed = &matrix * sign;
            assert!(factor_matches(&signed, sign, &valid));
            assert!(!factor_matches(&signed, sign, &missing));
            assert!(!factor_matches(&signed, sign, &nonzero_null));
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
