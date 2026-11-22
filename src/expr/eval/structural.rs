/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Shape transformations, indexing, and stacking.

use nalgebra::DMatrix;

use super::Array;
use super::affine::arr_to_dense;
use crate::expr::{IndexSpec, Shape};

pub(super) fn eval_promote(a: Array, shape: &Shape) -> crate::Result<Array> {
    let scalar = a.as_scalar().ok_or_else(|| {
        crate::CvxError::InvalidProblem("Only scalar expressions can be promoted".into())
    })?;
    Ok(Array::Dense(DMatrix::from_element(
        shape.rows(),
        shape.cols(),
        scalar,
    )))
}

pub(super) fn eval_reshape(a: Array, shape: &Shape) -> crate::Result<Array> {
    let flat: Vec<f64> = match a {
        Array::Scalar(v) => vec![v],
        Array::Dense(m) => m.iter().cloned().collect(),
        Array::Sparse(m) => {
            let mut v = vec![0.0; m.nrows() * m.ncols()];
            for (r, c, val) in m.triplet_iter() {
                v[c * m.nrows() + r] = *val;
            }
            v
        }
    };
    let (rows, cols) = (shape.rows(), shape.cols());
    if flat.len() != rows * cols {
        return Err(crate::CvxError::InvalidProblem(
            "Reshape size mismatch".into(),
        ));
    }
    if shape.is_scalar() {
        Ok(Array::Scalar(flat[0]))
    } else {
        Ok(Array::Dense(DMatrix::from_vec(rows, cols, flat)))
    }
}

pub(super) fn eval_index(a: Array, spec: &IndexSpec) -> crate::Result<Array> {
    let m = arr_to_dense(a);
    let nrows = m.nrows();
    let ncols = m.ncols();

    match spec.ranges.as_slice() {
        [Some((start, stop, step))] => {
            if *stop > nrows * ncols {
                return Err(crate::CvxError::InvalidProblem(
                    "Index out of bounds in eval".into(),
                ));
            }
            let indices: Vec<usize> = (*start..*stop).step_by(*step).collect();
            // For column vectors, index rows; otherwise index flat (column-major)
            let data: Vec<f64> = if ncols == 1 {
                indices.iter().map(|&i| m[(i, 0)]).collect()
            } else {
                indices.iter().map(|&i| m.as_slice()[i]).collect()
            };
            if data.len() == 1 {
                Ok(Array::Scalar(data[0]))
            } else {
                Ok(Array::Dense(DMatrix::from_vec(data.len(), 1, data)))
            }
        }
        [None] => Ok(Array::Dense(m)),
        [row_spec, col_spec] => {
            let row_range: Vec<usize> = match row_spec {
                Some((s, e, step)) => (*s..*e).step_by(*step).collect(),
                None => (0..nrows).collect(),
            };
            let col_range: Vec<usize> = match col_spec {
                Some((s, e, step)) => (*s..*e).step_by(*step).collect(),
                None => (0..ncols).collect(),
            };
            if row_range.iter().any(|&i| i >= nrows) || col_range.iter().any(|&i| i >= ncols) {
                return Err(crate::CvxError::InvalidProblem(
                    "Index out of bounds in eval".into(),
                ));
            }
            let row_drop = spec.drop_axes.first().copied().unwrap_or(false);
            let col_drop = spec.drop_axes.get(1).copied().unwrap_or(false);
            if row_drop && !col_drop {
                let data: Vec<f64> = col_range.iter().map(|&j| m[(row_range[0], j)]).collect();
                return Ok(Array::Dense(DMatrix::from_vec(data.len(), 1, data)));
            }
            if !row_drop && col_drop {
                let data: Vec<f64> = row_range.iter().map(|&i| m[(i, col_range[0])]).collect();
                return Ok(Array::Dense(DMatrix::from_vec(data.len(), 1, data)));
            }
            let result = DMatrix::from_fn(row_range.len(), col_range.len(), |i, j| {
                m[(row_range[i], col_range[j])]
            });
            if result.nrows() == 1 && result.ncols() == 1 {
                Ok(Array::Scalar(result[(0, 0)]))
            } else {
                Ok(Array::Dense(result))
            }
        }
        _ => Err(crate::CvxError::InvalidProblem(
            "Unsupported index spec in eval".into(),
        )),
    }
}

pub(super) fn eval_vstack(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Ok(Array::Scalar(0.0));
    }
    let mats: Vec<DMatrix<f64>> = arrays.into_iter().map(arr_to_dense).collect();
    let ncols = mats[0].ncols();
    let total_rows: usize = mats.iter().map(|m| m.nrows()).sum();
    let mut result = DMatrix::zeros(total_rows, ncols);
    let mut row_offset = 0;
    for m in mats {
        let nrows = m.nrows();
        result.rows_mut(row_offset, nrows).copy_from(&m);
        row_offset += nrows;
    }
    Ok(Array::Dense(result))
}

pub(super) fn eval_hstack(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Ok(Array::Scalar(0.0));
    }
    let mats: Vec<DMatrix<f64>> = arrays.into_iter().map(arr_to_dense).collect();
    let nrows = mats[0].nrows();
    let total_cols: usize = mats.iter().map(|m| m.ncols()).sum();
    let mut result = DMatrix::zeros(nrows, total_cols);
    let mut col_offset = 0;
    for m in mats {
        let ncols = m.ncols();
        result.columns_mut(col_offset, ncols).copy_from(&m);
        col_offset += ncols;
    }
    Ok(Array::Dense(result))
}

pub(super) fn eval_transpose(a: Array) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(v),
        Array::Dense(m) => Array::Dense(m.transpose()),
        a @ Array::Sparse(_) => Array::Dense(arr_to_dense(a).transpose()),
    }
}

pub(super) fn eval_diag(a: Array) -> Array {
    let m = arr_to_dense(a);
    let (rows, cols) = (m.nrows(), m.ncols());
    if cols == 1 || rows == 1 {
        // Vector → diagonal matrix
        let n = rows.max(cols);
        let mut result = DMatrix::zeros(n, n);
        for i in 0..n {
            let v = if cols == 1 { m[(i, 0)] } else { m[(0, i)] };
            result[(i, i)] = v;
        }
        Array::Dense(result)
    } else {
        // Matrix → diagonal vector
        let n = rows.min(cols);
        let data: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
        Array::Dense(DMatrix::from_vec(n, 1, data))
    }
}
