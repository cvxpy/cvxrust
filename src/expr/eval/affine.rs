/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Array arithmetic, reductions, and matrix multiplication.

use nalgebra::DMatrix;

use super::Array;
use super::broadcast::broadcast_pair;
use crate::expr::Shape;

/// Convert Array to DMatrix (always 2D column-major).
pub(super) fn arr_to_dense(a: Array) -> DMatrix<f64> {
    match a {
        Array::Dense(m) => m,
        Array::Scalar(v) => DMatrix::from_element(1, 1, v),
        Array::Sparse(m) => {
            let mut dense = DMatrix::zeros(m.nrows(), m.ncols());
            for (row, col, val) in m.triplet_iter() {
                dense[(row, col)] = *val;
            }
            dense
        }
    }
}

pub(super) fn eval_add(
    a: Array,
    a_shape: &Shape,
    b: Array,
    b_shape: &Shape,
) -> crate::Result<Array> {
    let (a, b) = broadcast_pair(a, a_shape, b, b_shape)?;
    match (a, b) {
        (Array::Scalar(x), Array::Scalar(y)) => Ok(Array::Scalar(x + y)),
        (Array::Scalar(s), b) => Ok(Array::Dense(arr_to_dense(b).map(|x| x + s))),
        (a, Array::Scalar(s)) => Ok(Array::Dense(arr_to_dense(a).map(|x| x + s))),
        (a, b) => {
            let am = arr_to_dense(a);
            let bm = arr_to_dense(b);
            if am.nrows() != bm.nrows() || am.ncols() != bm.ncols() {
                return Err(crate::CvxError::InvalidProblem(
                    "Shape mismatch in addition".into(),
                ));
            }
            Ok(Array::Dense(am + bm))
        }
    }
}

pub(super) fn eval_neg(a: Array) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(-v),
        Array::Dense(m) => Array::Dense(-m),
        a @ Array::Sparse(_) => Array::Dense(-arr_to_dense(a)),
    }
}

pub(super) fn eval_mul(
    a: Array,
    a_shape: &Shape,
    b: Array,
    b_shape: &Shape,
) -> crate::Result<Array> {
    let (a, b) = broadcast_pair(a, a_shape, b, b_shape)?;
    match (a, b) {
        (Array::Scalar(x), Array::Scalar(y)) => Ok(Array::Scalar(x * y)),
        (Array::Scalar(s), b) => Ok(Array::Dense(arr_to_dense(b) * s)),
        (a, Array::Scalar(s)) => Ok(Array::Dense(arr_to_dense(a) * s)),
        (a, b) => {
            let am = arr_to_dense(a);
            let bm = arr_to_dense(b);
            if am.nrows() != bm.nrows() || am.ncols() != bm.ncols() {
                return Err(crate::CvxError::InvalidProblem(
                    "Shape mismatch in element-wise multiply".into(),
                ));
            }
            Ok(Array::Dense(am.component_mul(&bm)))
        }
    }
}

pub(super) fn eval_matmul(a: Array, b: Array) -> crate::Result<Array> {
    let am = arr_to_dense(a);
    let bm = arr_to_dense(b);
    if am.ncols() != bm.nrows() {
        return Err(crate::CvxError::InvalidProblem(
            "Shape mismatch in matrix multiply".into(),
        ));
    }
    Ok(Array::Dense(am * bm))
}

pub(super) fn eval_sum(a: Array, axis: Option<usize>) -> crate::Result<Array> {
    match axis {
        None => {
            let total: f64 = match &a {
                Array::Scalar(v) => *v,
                Array::Dense(m) => m.sum(),
                Array::Sparse(m) => m.values().iter().sum(),
            };
            Ok(Array::Scalar(total))
        }
        Some(0) => {
            // Summing rows produces a column vector of size ncols.
            let m = arr_to_dense(a);
            let result = DMatrix::from_fn(m.ncols(), 1, |j, _| m.column(j).iter().sum::<f64>());
            Ok(Array::Dense(result))
        }
        Some(1) => {
            // Summing columns produces a column vector of size nrows.
            let m = arr_to_dense(a);
            let result = DMatrix::from_fn(m.nrows(), 1, |i, _| m.row(i).iter().sum::<f64>());
            Ok(Array::Dense(result))
        }
        Some(ax) => Err(crate::CvxError::InvalidProblem(format!(
            "Invalid axis {} for sum",
            ax
        ))),
    }
}

pub(super) fn eval_trace(a: Array) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(v),
        a => {
            let m = arr_to_dense(a);
            Array::Scalar((0..m.nrows().min(m.ncols())).map(|i| m[(i, i)]).sum())
        }
    }
}

pub(super) fn eval_cumsum(a: Array, axis: Option<usize>) -> crate::Result<Array> {
    let mut m = arr_to_dense(a);
    match axis {
        None | Some(0) => {
            // Cumsum along rows
            for i in 1..m.nrows() {
                for j in 0..m.ncols() {
                    m[(i, j)] += m[(i - 1, j)];
                }
            }
        }
        Some(1) => {
            // Cumsum along columns
            for j in 1..m.ncols() {
                for i in 0..m.nrows() {
                    m[(i, j)] += m[(i, j - 1)];
                }
            }
        }
        Some(ax) => {
            return Err(crate::CvxError::InvalidProblem(format!(
                "Invalid axis {} for cumsum",
                ax
            )));
        }
    }
    Ok(Array::Dense(m))
}
