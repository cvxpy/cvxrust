/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Numerical broadcasting using declared expression shapes.

use nalgebra::DMatrix;

use super::Array;
use super::affine::arr_to_dense;
use crate::expr::Shape;

pub(super) fn broadcast_shape(a: &Shape, b: &Shape) -> crate::Result<Shape> {
    a.broadcast(b).ok_or_else(|| {
        crate::CvxError::InvalidProblem(format!("Cannot broadcast shapes {a} and {b}"))
    })
}

pub(super) fn broadcast_array(
    array: Array,
    source: &Shape,
    target: &Shape,
) -> crate::Result<Array> {
    let actual = array.shape();
    if actual.rows() != source.rows() || actual.cols() != source.cols() {
        return Err(crate::CvxError::InvalidProblem(format!(
            "Evaluated shape {actual} does not match expression shape {source}"
        )));
    }
    if source.rows() == target.rows() && source.cols() == target.cols() {
        return Ok(array);
    }
    if source.is_scalar_like() {
        let value = array.as_scalar().ok_or_else(|| {
            crate::CvxError::InvalidProblem("Cannot broadcast an empty array".into())
        })?;
        return Ok(Array::Dense(DMatrix::from_element(
            target.rows(),
            target.cols(),
            value,
        )));
    }
    if !source.is_matrix()
        || !target.is_matrix()
        || (source.rows() != target.rows() && source.rows() != 1)
        || (source.cols() != target.cols() && source.cols() != 1)
    {
        return Err(crate::CvxError::InvalidProblem(format!(
            "Cannot broadcast shape {source} to {target}"
        )));
    }

    let matrix = arr_to_dense(array);
    Ok(Array::Dense(DMatrix::from_fn(
        target.rows(),
        target.cols(),
        |row, col| {
            matrix[(
                if source.rows() == 1 { 0 } else { row },
                if source.cols() == 1 { 0 } else { col },
            )]
        },
    )))
}

pub(super) fn broadcast_pair(
    a: Array,
    a_shape: &Shape,
    b: Array,
    b_shape: &Shape,
) -> crate::Result<(Array, Array)> {
    let target = broadcast_shape(a_shape, b_shape)?;
    Ok((
        broadcast_array(a, a_shape, &target)?,
        broadcast_array(b, b_shape, &target)?,
    ))
}

pub(super) fn broadcast_arrays(arrays: Vec<(Array, Shape)>) -> crate::Result<Vec<Array>> {
    let Some((_, first_shape)) = arrays.first() else {
        return Ok(Vec::new());
    };
    let target = arrays
        .iter()
        .skip(1)
        .try_fold(first_shape.clone(), |shape, (_, next)| {
            broadcast_shape(&shape, next)
        })?;
    arrays
        .into_iter()
        .map(|(array, shape)| broadcast_array(array, &shape, &target))
        .collect()
}
