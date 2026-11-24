/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Shape representation for expressions.
//!
//! Shapes follow NumPy conventions:
//! - `()` or `[]` is a scalar
//! - `(n,)` or `[n]` is a vector of length n
//! - `(m, n)` or `[m, n]` is an `m x n` matrix

use std::fmt;

/// Shape of an expression (row-major like NumPy).
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a scalar shape.
    pub fn scalar() -> Self {
        Shape(vec![])
    }

    /// Create a vector shape.
    pub fn vector(n: usize) -> Self {
        Shape(vec![n])
    }

    /// Create a matrix shape.
    pub fn matrix(m: usize, n: usize) -> Self {
        Shape(vec![m, n])
    }

    /// Create a shape from dimensions.
    pub fn from_dims(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.0.iter().product::<usize>().max(1)
    }

    /// Number of dimensions (0 for scalar, 1 for vector, 2 for matrix).
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get the dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Check if this is a scalar.
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if this shape has exactly one element.
    ///
    /// This is used for CVXPY-style scalar promotion in broadcasting while
    /// preserving `is_scalar()` as the strict `()` shape check.
    pub fn is_scalar_like(&self) -> bool {
        self.size() == 1
    }

    /// Check if this is a vector.
    pub fn is_vector(&self) -> bool {
        self.0.len() == 1
    }

    /// Check if this is a matrix.
    pub fn is_matrix(&self) -> bool {
        self.0.len() == 2
    }

    /// Number of rows (1 for scalar, n for vector, m for matrix).
    pub fn rows(&self) -> usize {
        match self.0.len() {
            0 => 1,
            1 => self.0[0],
            _ => self.0[0],
        }
    }

    /// Number of columns (1 for scalar, 1 for vector, n for matrix).
    pub fn cols(&self) -> usize {
        match self.0.len() {
            0 => 1,
            1 => 1,
            _ => self.0[1],
        }
    }

    /// Get the transposed shape.
    pub fn transpose(&self) -> Self {
        match self.0.len() {
            0 => Shape::scalar(),
            1 => Shape::matrix(1, self.0[0]),
            2 => Shape::matrix(self.0[1], self.0[0]),
            _ => {
                let mut dims = self.0.clone();
                dims.reverse();
                Shape(dims)
            }
        }
    }

    /// Return the result shape for cvxrust element-wise affine broadcasting.
    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        if self == other {
            return Some(self.clone());
        }

        if self.is_scalar_like() && !other.is_scalar_like() {
            return Some(other.clone());
        }
        if other.is_scalar_like() && !self.is_scalar_like() {
            return Some(self.clone());
        }

        if self.rows() == other.rows() && self.cols() == other.cols() {
            if self.is_vector() {
                return Some(self.clone());
            }
            if other.is_vector() {
                return Some(other.clone());
            }
            return Some(Shape::matrix(self.rows(), self.cols()));
        }

        if self.is_matrix() && other.is_matrix() {
            if self.rows() == 1 && self.cols() == other.cols() {
                return Some(other.clone());
            }
            if other.rows() == 1 && other.cols() == self.cols() {
                return Some(self.clone());
            }
            if self.cols() == 1 && self.rows() == other.rows() {
                return Some(other.clone());
            }
            if other.cols() == 1 && other.rows() == self.rows() {
                return Some(self.clone());
            }
            if self.rows() == 1 && other.cols() == 1 {
                return Some(Shape::matrix(other.rows(), self.cols()));
            }
            if other.rows() == 1 && self.cols() == 1 {
                return Some(Shape::matrix(self.rows(), other.cols()));
            }
        }

        None
    }

    /// Check if matrix multiplication is valid and return result shape.
    pub fn matmul(&self, other: &Shape) -> Option<Shape> {
        // Handle various cases
        match (self.ndim(), other.ndim()) {
            // Matrix-scalar product, treating the scalar as a 1 x 1 column
            (2, 0) if self.cols() == 1 => Some(Shape::vector(self.rows())),
            // Matrix-matrix product
            (2, 2) if self.cols() == other.rows() => Some(Shape::matrix(self.rows(), other.cols())),
            // Matrix-vector product
            (2, 1) if self.cols() == other.rows() => Some(Shape::vector(self.rows())),
            // Vector-matrix product (vector treated as a row)
            (1, 2) if self.rows() == other.rows() => Some(Shape::vector(other.cols())),
            // Vector dot product
            (1, 1) if self.rows() == other.rows() => Some(Shape::scalar()),
            _ => None,
        }
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, "()")
        } else if self.0.len() == 1 {
            write!(f, "({},)", self.0[0])
        } else {
            write!(f, "({}, {})", self.0[0], self.0[1])
        }
    }
}

// Conversion traits
impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape::scalar()
    }
}

impl From<usize> for Shape {
    fn from(n: usize) -> Self {
        Shape::vector(n)
    }
}

impl From<(usize,)> for Shape {
    fn from((n,): (usize,)) -> Self {
        Shape::vector(n)
    }
}

impl From<(usize, usize)> for Shape {
    fn from((m, n): (usize, usize)) -> Self {
        Shape::matrix(m, n)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape(dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.size(), 1);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.rows(), 1);
        assert_eq!(s.cols(), 1);
    }

    #[test]
    fn test_vector() {
        let s = Shape::vector(5);
        assert!(s.is_vector());
        assert_eq!(s.size(), 5);
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.rows(), 5);
        assert_eq!(s.cols(), 1);
    }

    #[test]
    fn test_matrix() {
        let s = Shape::matrix(3, 4);
        assert!(s.is_matrix());
        assert_eq!(s.size(), 12);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.rows(), 3);
        assert_eq!(s.cols(), 4);
    }

    #[test]
    fn test_transpose() {
        assert_eq!(Shape::scalar().transpose(), Shape::scalar());
        assert_eq!(Shape::vector(3).transpose(), Shape::matrix(1, 3));
        assert_eq!(Shape::matrix(3, 4).transpose(), Shape::matrix(4, 3));
    }

    #[test]
    fn test_broadcast() {
        // Same shapes
        assert_eq!(
            Shape::vector(3).broadcast(&Shape::vector(3)),
            Some(Shape::vector(3))
        );

        // Scalar broadcasts to anything
        assert_eq!(
            Shape::scalar().broadcast(&Shape::matrix(3, 4)),
            Some(Shape::matrix(3, 4))
        );

        // Row and column matrices broadcast over a matching matrix.
        assert_eq!(
            Shape::matrix(1, 4).broadcast(&Shape::matrix(3, 4)),
            Some(Shape::matrix(3, 4))
        );
        assert_eq!(
            Shape::matrix(3, 1).broadcast(&Shape::matrix(3, 4)),
            Some(Shape::matrix(3, 4))
        );

        // Mutual row/column broadcast.
        assert_eq!(
            Shape::matrix(1, 4).broadcast(&Shape::matrix(3, 1)),
            Some(Shape::matrix(3, 4))
        );

        // Incompatible
        assert_eq!(Shape::vector(3).broadcast(&Shape::vector(4)), None);
    }

    #[test]
    fn test_matmul() {
        // Matrix-matrix product
        assert_eq!(
            Shape::matrix(3, 4).matmul(&Shape::matrix(4, 5)),
            Some(Shape::matrix(3, 5))
        );

        // Matrix-vector product
        assert_eq!(
            Shape::matrix(3, 4).matmul(&Shape::vector(4)),
            Some(Shape::vector(3))
        );

        // Vector dot product
        assert_eq!(
            Shape::vector(3).matmul(&Shape::vector(3)),
            Some(Shape::scalar())
        );

        // Incompatible
        assert_eq!(Shape::matrix(3, 4).matmul(&Shape::vector(3)), None);
    }

    #[test]
    fn test_conversions() {
        let _: Shape = ().into();
        let _: Shape = 5.into();
        let _: Shape = (5,).into();
        let _: Shape = (3, 4).into();
    }
}
