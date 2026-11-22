/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Axis selectors and index specifications.

use super::shape::Shape;

/// Selector for one axis of an indexing operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisIndex {
    /// Select a single index and drop this axis.
    Index(usize),
    /// Select a half-open range and preserve this axis.
    Slice(usize, usize),
    /// Select the entire axis and preserve it.
    All,
}

/// Specification for indexing operations.
#[derive(Debug, Clone)]
pub struct IndexSpec {
    /// Ranges for each dimension: (start, stop, step).
    /// None means take the whole dimension.
    pub ranges: Vec<Option<(usize, usize, usize)>>,
    /// Whether each indexed dimension was selected by an integer index.
    pub drop_axes: Vec<bool>,
}

impl IndexSpec {
    /// Create an index spec for a single element.
    pub fn element(indices: Vec<usize>) -> Self {
        let len = indices.len();
        IndexSpec {
            ranges: indices.into_iter().map(|i| Some((i, i + 1, 1))).collect(),
            drop_axes: vec![true; len],
        }
    }

    /// Create an index spec for a range.
    pub fn range(start: usize, stop: usize) -> Self {
        IndexSpec {
            ranges: vec![Some((start, stop, 1))],
            drop_axes: vec![false],
        }
    }

    /// Create an index spec that takes everything.
    pub fn all() -> Self {
        IndexSpec {
            ranges: vec![None],
            drop_axes: vec![false],
        }
    }

    /// Compute the output shape when this spec is applied to a base shape.
    pub fn output_shape(&self, base: &Shape) -> Shape {
        let mut new_dims = Vec::new();
        for i in 0..base.ndim() {
            let range = self.ranges.get(i).copied().flatten();
            let drop_axis = self.drop_axes.get(i).copied().unwrap_or(false);
            match range {
                Some((start, stop, step)) => {
                    let size = (stop - start).div_ceil(step);
                    if !drop_axis {
                        new_dims.push(size);
                    }
                }
                None => new_dims.push(base.dims()[i]),
            }
        }

        if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::from_dims(new_dims)
        }
    }
}
