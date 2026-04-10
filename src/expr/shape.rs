//! Shape representation for expressions.
//!
//! Shapes follow NumPy conventions:
//! - `()` or `[]` is a scalar
//! - `(n,)` or `[n]` is a vector of length n
//! - `(m, n)` or `[m, n]` is an m x n matrix

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

    /// Check if shapes are broadcastable and return the result shape.
    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result = Vec::with_capacity(max_ndim);

        // Pad shapes with 1s on the left
        let self_padded: Vec<usize> = std::iter::repeat_n(1, max_ndim - self.ndim())
            .chain(self.0.iter().copied())
            .collect();
        let other_padded: Vec<usize> = std::iter::repeat_n(1, max_ndim - other.ndim())
            .chain(other.0.iter().copied())
            .collect();

        for (a, b) in self_padded.iter().zip(other_padded.iter()) {
            if *a == *b {
                result.push(*a);
            } else if *a == 1 {
                result.push(*b);
            } else if *b == 1 {
                result.push(*a);
            } else {
                return None; // Not broadcastable
            }
        }

        Some(Shape(result))
    }

    /// Check if matrix multiplication is valid and return result shape.
    pub fn matmul(&self, other: &Shape) -> Option<Shape> {
        // Handle various cases
        match (self.ndim(), other.ndim()) {
            // matrix @ matrix
            (2, 2) if self.cols() == other.rows() => Some(Shape::matrix(self.rows(), other.cols())),
            // matrix @ vector
            (2, 1) if self.cols() == other.rows() => Some(Shape::vector(self.rows())),
            // vector @ matrix (treated as row vector)
            (1, 2) if self.rows() == other.rows() => Some(Shape::vector(other.cols())),
            // vector @ vector (dot product)
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

        // Vector broadcasts with matrix
        assert_eq!(
            Shape::vector(4).broadcast(&Shape::matrix(3, 4)),
            Some(Shape::matrix(3, 4))
        );

        // Incompatible
        assert_eq!(Shape::vector(3).broadcast(&Shape::vector(4)), None);
    }

    #[test]
    fn test_matmul() {
        // matrix @ matrix
        assert_eq!(
            Shape::matrix(3, 4).matmul(&Shape::matrix(4, 5)),
            Some(Shape::matrix(3, 5))
        );

        // matrix @ vector
        assert_eq!(
            Shape::matrix(3, 4).matmul(&Shape::vector(4)),
            Some(Shape::vector(3))
        );

        // vector @ vector (dot product)
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
