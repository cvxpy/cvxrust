/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Sparse matrix utilities.
//!
//! Helper functions for working with nalgebra-sparse matrices.

use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Create a CSC matrix from triplets (row, col, value).
///
/// Duplicates are summed together.
pub fn csc_from_triplets(
    nrows: usize,
    ncols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
) -> CscMatrix<f64> {
    if rows.is_empty() {
        return CscMatrix::zeros(nrows, ncols);
    }

    // Build COO matrix first
    let mut coo = CooMatrix::new(nrows, ncols);
    for ((row, col), val) in rows.into_iter().zip(cols).zip(vals) {
        if row < nrows && col < ncols {
            coo.push(row, col, val);
        }
    }

    // Convert to CSC
    CscMatrix::from(&coo)
}

/// Create a CSC identity matrix.
pub fn csc_identity(n: usize) -> CscMatrix<f64> {
    CscMatrix::identity(n)
}

/// Convert a dense matrix to CSC format.
pub fn dense_to_csc(dense: &DMatrix<f64>) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for j in 0..dense.ncols() {
        for i in 0..dense.nrows() {
            let v = dense[(i, j)];
            if v != 0.0 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
    }

    csc_from_triplets(dense.nrows(), dense.ncols(), rows, cols, vals)
}

/// Convert CSC to dense matrix.
pub fn csc_to_dense(sparse: &CscMatrix<f64>) -> DMatrix<f64> {
    let mut dense = DMatrix::zeros(sparse.nrows(), sparse.ncols());
    for (row, col, val) in sparse.triplet_iter() {
        dense[(row, col)] = *val;
    }
    dense
}

/// Stack two CSC matrices vertically.
pub fn csc_vstack(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    assert_eq!(
        a.ncols(),
        b.ncols(),
        "vstack requires matching column counts"
    );
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in a.triplet_iter() {
        rows.push(r);
        cols.push(c);
        vals.push(*v);
    }
    for (r, c, v) in b.triplet_iter() {
        rows.push(r + a.nrows());
        cols.push(c);
        vals.push(*v);
    }

    csc_from_triplets(a.nrows() + b.nrows(), a.ncols(), rows, cols, vals)
}

/// Add two CSC matrices.
/// Uses nalgebra_sparse's built-in addition which is more efficient.
pub fn csc_add(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    a + b
}

/// Negate a CSC matrix.
pub fn csc_neg(a: &CscMatrix<f64>) -> CscMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|v| -v).collect();
    let col_offsets: Vec<usize> = a.col_offsets().to_vec();
    let row_indices: Vec<usize> = a.row_indices().to_vec();
    CscMatrix::try_from_csc_data(a.nrows(), a.ncols(), col_offsets, row_indices, values)
        .expect("valid CSC structure should remain valid when negating values")
}

/// Scale a CSC matrix.
pub fn csc_scale(a: &CscMatrix<f64>, scalar: f64) -> CscMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|v| v * scalar).collect();
    let col_offsets: Vec<usize> = a.col_offsets().to_vec();
    let row_indices: Vec<usize> = a.row_indices().to_vec();
    CscMatrix::try_from_csc_data(a.nrows(), a.ncols(), col_offsets, row_indices, values)
        .expect("valid CSC structure should remain valid when scaling values")
}

/// Multiply a CSC matrix by a dense matrix on the right.
pub fn sparse_dense_matmul(a: &CscMatrix<f64>, b: &DMatrix<f64>) -> CscMatrix<f64> {
    // Convert sparse to dense, multiply, convert back
    // This is not optimal but correct
    let a_dense = csc_to_dense(a);
    let result = &a_dense * b;
    dense_to_csc(&result)
}

/// Stack two CSC matrices horizontally.
pub fn csc_hstack(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    assert_eq!(a.nrows(), b.nrows(), "hstack requires matching row counts");
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in a.triplet_iter() {
        rows.push(r);
        cols.push(c);
        vals.push(*v);
    }
    for (r, c, v) in b.triplet_iter() {
        rows.push(r);
        cols.push(c + a.ncols());
        vals.push(*v);
    }

    csc_from_triplets(a.nrows(), a.ncols() + b.ncols(), rows, cols, vals)
}

/// Multiply two CSC matrices: `A B`.
pub fn csc_matmul(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    // Use nalgebra-sparse's built-in multiplication
    a * b
}

/// Alias for csc_from_triplets with reference arguments.
pub fn triplets_to_csc(
    nrows: usize,
    ncols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
) -> CscMatrix<f64> {
    csc_from_triplets(nrows, ncols, rows.to_vec(), cols.to_vec(), vals.to_vec())
}

/// Repeat rows of a CSC matrix.
pub fn csc_repeat_rows(m: &CscMatrix<f64>, times: usize) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in m.triplet_iter() {
        for t in 0..times {
            rows.push(t * m.nrows() + r);
            cols.push(c);
            vals.push(*v);
        }
    }

    csc_from_triplets(m.nrows() * times, m.ncols(), rows, cols, vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_from_triplets() {
        let m = csc_from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn test_csc_identity() {
        let m = csc_identity(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn test_dense_to_csc() {
        let dense = DMatrix::identity(3, 3);
        let sparse = dense_to_csc(&dense);
        assert_eq!(sparse.nrows(), 3);
    }

    #[test]
    fn test_dense_to_csc_preserves_small_nonzero_entries() {
        let dense = DMatrix::from_row_slice(2, 3, &[0.0, 1e-16, -1e-20, 1e-15, -0.0, 2.0]);
        let sparse = dense_to_csc(&dense);
        assert_eq!(sparse.nnz(), 4);
        approx::assert_relative_eq!(
            csc_to_dense(&sparse),
            dense,
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_csc_add() {
        let a = CscMatrix::<f64>::identity(3);
        let b = CscMatrix::<f64>::identity(3);
        let c = csc_add(&a, &b);
        // Identity + Identity = 2*Identity, so diagonal values should be 2.0
        assert_eq!(c.values().len(), 3);
        for v in c.values() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_csc_neg_preserves_structure_and_negates_values() {
        let a = csc_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![2.0, -3.0]);
        let neg = csc_neg(&a);
        let dense = csc_to_dense(&neg);

        assert_eq!(neg.nrows(), 2);
        assert_eq!(neg.ncols(), 2);
        assert_eq!(dense[(0, 0)], -2.0);
        assert_eq!(dense[(1, 1)], 3.0);
    }

    #[test]
    fn test_csc_scale_preserves_structure_and_scales_values() {
        let a = csc_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![2.0, -3.0]);
        let scaled = csc_scale(&a, 4.0);
        let dense = csc_to_dense(&scaled);

        assert_eq!(scaled.nrows(), 2);
        assert_eq!(scaled.ncols(), 2);
        assert_eq!(dense[(0, 0)], 8.0);
        assert_eq!(dense[(1, 1)], -12.0);
    }

    #[test]
    fn test_csc_vstack_matching_columns() {
        let a = CscMatrix::<f64>::identity(2);
        let b = CscMatrix::<f64>::zeros(3, 2);
        let stacked = csc_vstack(&a, &b);

        assert_eq!(stacked.nrows(), 5);
        assert_eq!(stacked.ncols(), 2);
    }

    #[test]
    #[should_panic(expected = "vstack requires matching column counts")]
    fn test_csc_vstack_mismatched_columns_panics() {
        let a = CscMatrix::<f64>::zeros(2, 2);
        let b = CscMatrix::<f64>::zeros(3, 3);
        let _ = csc_vstack(&a, &b);
    }

    #[test]
    fn test_csc_hstack_matching_rows() {
        let a = CscMatrix::<f64>::identity(2);
        let b = CscMatrix::<f64>::identity(2);
        let stacked = csc_hstack(&a, &b);

        assert_eq!(stacked.nrows(), 2);
        assert_eq!(stacked.ncols(), 4);
    }

    #[test]
    #[should_panic(expected = "hstack requires matching row counts")]
    fn test_csc_hstack_mismatched_rows_panics() {
        let a = CscMatrix::<f64>::zeros(2, 2);
        let b = CscMatrix::<f64>::zeros(3, 2);
        let _ = csc_hstack(&a, &b);
    }
}
