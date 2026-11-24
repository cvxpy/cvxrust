/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Numerical evaluation of nonlinear atoms.

use super::Array;
use super::affine::arr_to_dense;

pub(super) fn eval_norm1(a: &Array) -> f64 {
    match a {
        Array::Scalar(v) => v.abs(),
        Array::Dense(m) => m.iter().map(|x| x.abs()).sum(),
        Array::Sparse(m) => m.values().iter().map(|x| x.abs()).sum(),
    }
}

pub(super) fn eval_norm2(a: &Array) -> f64 {
    match a {
        Array::Scalar(v) => v.abs(),
        Array::Dense(m) => m.iter().map(|x| x * x).sum::<f64>().sqrt(),
        Array::Sparse(m) => m.values().iter().map(|x| x * x).sum::<f64>().sqrt(),
    }
}

pub(super) fn eval_norminf(a: &Array) -> f64 {
    match a {
        Array::Scalar(v) => v.abs(),
        Array::Dense(m) => m.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
        Array::Sparse(m) => m.values().iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
    }
}

pub(super) fn eval_elementwise(a: Array, f: impl Fn(f64) -> f64) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(f(v)),
        Array::Dense(m) => Array::Dense(m.map(f)),
        a @ Array::Sparse(_) => Array::Dense(arr_to_dense(a).map(f)),
    }
}

pub(super) fn eval_maximum(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Err(crate::CvxError::InvalidProblem(
            "maximum of empty list".into(),
        ));
    }
    let mut iter = arrays.into_iter();
    let first = arr_to_dense(iter.next().unwrap());
    let result = iter.try_fold(first, |acc, a| {
        let m = arr_to_dense(a);
        if acc.nrows() != m.nrows() || acc.ncols() != m.ncols() {
            return Err(crate::CvxError::InvalidProblem(
                "Shape mismatch in maximum".into(),
            ));
        }
        Ok(acc.zip_map(&m, f64::max))
    })?;
    Ok(Array::Dense(result))
}

pub(super) fn eval_minimum(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Err(crate::CvxError::InvalidProblem(
            "minimum of empty list".into(),
        ));
    }
    let mut iter = arrays.into_iter();
    let first = arr_to_dense(iter.next().unwrap());
    let result = iter.try_fold(first, |acc, a| {
        let m = arr_to_dense(a);
        if acc.nrows() != m.nrows() || acc.ncols() != m.ncols() {
            return Err(crate::CvxError::InvalidProblem(
                "Shape mismatch in minimum".into(),
            ));
        }
        Ok(acc.zip_map(&m, f64::min))
    })?;
    Ok(Array::Dense(result))
}

pub(super) fn eval_quad_form(x: Array, p: Array) -> crate::Result<Array> {
    let xm = arr_to_dense(x);
    let pm = arr_to_dense(p);
    if pm.nrows() != pm.ncols() || xm.nrows() != pm.nrows() {
        return Err(crate::CvxError::InvalidProblem(
            "Shape mismatch in quad_form".into(),
        ));
    }
    // x^T P x
    let result = (xm.transpose() * &pm * &xm)[(0, 0)];
    Ok(Array::Scalar(result))
}

pub(super) fn eval_quad_over_lin(x: Array, y: Array) -> crate::Result<Array> {
    let norm_sq = eval_norm2(&x).powi(2);
    let y_val = y.as_scalar().ok_or_else(|| {
        crate::CvxError::InvalidProblem("quad_over_lin: denominator must be scalar".into())
    })?;
    Ok(Array::Scalar(norm_sq / y_val))
}
