/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Dense, sparse, and scalar storage for expression values.

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::semidefinite::{self, Definiteness};
use super::shape::Shape;

/// Form `(P + P^T) / 2` with stable averaging.
pub(crate) fn symmetric_part(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    debug_assert!(matrix.is_square());
    let mut symmetric = matrix.clone();
    for i in 0..matrix.nrows() {
        for j in (i + 1)..matrix.ncols() {
            let value = matrix[(i, j)].midpoint(matrix[(j, i)]);
            symmetric[(i, j)] = value;
            symmetric[(j, i)] = value;
        }
    }
    symmetric
}

/// Efficient array storage (dense or sparse).
#[derive(Debug, Clone)]
pub enum Array {
    /// Dense matrix storage.
    Dense(DMatrix<f64>),
    /// Sparse CSC matrix storage.
    Sparse(CscMatrix<f64>),
    /// Scalar value.
    Scalar(f64),
}

impl Array {
    /// Get the shape of the array.
    pub fn shape(&self) -> Shape {
        match self {
            Array::Dense(m) => Shape::matrix(m.nrows(), m.ncols()),
            Array::Sparse(m) => Shape::matrix(m.nrows(), m.ncols()),
            Array::Scalar(_) => Shape::scalar(),
        }
    }

    /// Get the total number of elements.
    pub fn size(&self) -> usize {
        match self {
            Array::Dense(m) => m.nrows() * m.ncols(),
            Array::Sparse(m) => m.nrows() * m.ncols(),
            Array::Scalar(_) => 1,
        }
    }

    /// Try to get as a scalar value.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Array::Scalar(v) => Some(*v),
            Array::Dense(m) if m.nrows() == 1 && m.ncols() == 1 => Some(m[(0, 0)]),
            _ => None,
        }
    }

    /// Check if all elements are non-negative.
    pub fn is_nonneg(&self) -> bool {
        match self {
            Array::Scalar(v) => *v >= 0.0,
            Array::Dense(m) => m.iter().all(|&v| v >= 0.0),
            Array::Sparse(m) => m.values().iter().all(|&v| v >= 0.0),
        }
    }

    /// Check if all elements are non-positive.
    pub fn is_nonpos(&self) -> bool {
        match self {
            Array::Scalar(v) => *v <= 0.0,
            Array::Dense(m) => m.iter().all(|&v| v <= 0.0),
            Array::Sparse(m) => {
                // For sparse, zeros are implicitly non-positive
                m.values().iter().all(|&v| v <= 0.0)
            }
        }
    }

    /// Check positive semidefiniteness within floating-point roundoff.
    ///
    /// Returns `None` for nonsquare, nonsymmetric, or nonfinite inputs, or if
    /// numerical analysis fails.
    pub fn is_psd(&self) -> Option<bool> {
        semidefinite::analyze(self)
            .map(|analysis| matches!(analysis.kind, Definiteness::Zero | Definiteness::Psd))
    }

    /// Create from a scalar.
    pub fn from_scalar(v: f64) -> Self {
        Array::Scalar(v)
    }

    /// Create from a vector.
    pub fn from_vec(v: Vec<f64>) -> Self {
        let n = v.len();
        Array::Dense(DMatrix::from_vec(n, 1, v))
    }

    /// Create from a dense matrix.
    pub fn from_matrix(m: DMatrix<f64>) -> Self {
        Array::Dense(m)
    }
}

impl From<f64> for Array {
    fn from(v: f64) -> Self {
        Array::Scalar(v)
    }
}

impl From<Vec<f64>> for Array {
    fn from(v: Vec<f64>) -> Self {
        Array::from_vec(v)
    }
}

impl From<DMatrix<f64>> for Array {
    fn from(m: DMatrix<f64>) -> Self {
        Array::Dense(m)
    }
}

impl std::ops::Index<(usize, usize)> for Array {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        match self {
            Array::Dense(m) => &m[(row, col)],
            Array::Scalar(v) => {
                assert!(row == 0 && col == 0, "scalar index out of bounds");
                v
            }
            Array::Sparse(_) => panic!("use Array::Dense for indexing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_scalar() {
        let arr = Array::Scalar(5.0);
        assert_eq!(arr.as_scalar(), Some(5.0));
        assert!(arr.is_nonneg());
        assert!(!arr.is_nonpos());
    }

    #[test]
    fn test_array_from_vec() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(arr.shape(), Shape::matrix(3, 1));
        assert!(arr.is_nonneg());
    }

    #[test]
    fn test_array_semidefinite_status() {
        use crate::dcp::PsdStatus;
        use crate::sparse::dense_to_csc;

        for (entries, psd, status) in [
            ([1.0, 0.0], Some(true), PsdStatus::Psd),
            ([-1.0, 0.0], Some(false), PsdStatus::Nsd),
            ([0.0, 0.0], Some(true), PsdStatus::Psd),
            ([1.0, -1.0], Some(false), PsdStatus::Neither),
            ([1.0, f64::NAN], None, PsdStatus::Neither),
        ] {
            let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&entries));
            for array in [Array::Sparse(dense_to_csc(&matrix)), Array::Dense(matrix)] {
                assert_eq!(array.is_psd(), psd);
                assert_eq!(PsdStatus::of_array(&array), status);
            }
        }
        assert_eq!(Array::Dense(DMatrix::zeros(2, 3)).is_psd(), None);
        assert_eq!(Array::Scalar(f64::INFINITY).is_psd(), None);
    }
}
