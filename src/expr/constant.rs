/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Constant expression creation.

use std::sync::Arc;

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::expression::{Array, ConstantData, Expr, ExprId};
use super::shape::Shape;

/// Create a constant expression from a scalar.
pub fn constant(value: f64) -> Expr {
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value: Array::Scalar(value),
    })
}

/// Create a constant expression from a vector.
pub fn constant_vec(values: Vec<f64>) -> Expr {
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value: Array::from_vec(values),
    })
}

/// Create a constant expression from a dense matrix.
pub fn constant_matrix(values: Vec<f64>, rows: usize, cols: usize) -> Expr {
    let matrix = DMatrix::from_vec(rows, cols, values);
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value: Array::Dense(matrix),
    })
}

/// Create a constant expression from a nalgebra DMatrix.
pub fn constant_dmatrix(matrix: DMatrix<f64>) -> Expr {
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value: Array::Dense(matrix),
    })
}

/// Create a constant expression from a sparse CSC matrix.
pub fn constant_sparse(matrix: CscMatrix<f64>) -> Expr {
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value: Array::Sparse(matrix),
    })
}

/// Create a zero constant with the given shape.
pub fn zeros(shape: impl Into<Shape>) -> Expr {
    let shape = shape.into();
    let value = if shape.is_scalar() {
        Array::Scalar(0.0)
    } else {
        Array::Dense(DMatrix::zeros(shape.rows(), shape.cols()))
    };
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value,
    })
}

/// Create a ones constant with the given shape.
pub fn ones(shape: impl Into<Shape>) -> Expr {
    let shape = shape.into();
    let value = if shape.is_scalar() {
        Array::Scalar(1.0)
    } else {
        Array::Dense(DMatrix::from_element(shape.rows(), shape.cols(), 1.0))
    };
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value,
    })
}

/// Create an identity matrix constant.
pub fn eye(n: usize) -> Expr {
    Expr::Constant(ConstantData {
        id: ExprId::new(),
        value: Array::Dense(DMatrix::identity(n, n)),
    })
}

/// Extension trait for creating constants from various types.
pub trait IntoConstant {
    fn into_constant(self) -> Expr;
}

impl IntoConstant for f64 {
    fn into_constant(self) -> Expr {
        constant(self)
    }
}

impl IntoConstant for i32 {
    fn into_constant(self) -> Expr {
        constant(self as f64)
    }
}

impl IntoConstant for Vec<f64> {
    fn into_constant(self) -> Expr {
        constant_vec(self)
    }
}

impl IntoConstant for &[f64] {
    fn into_constant(self) -> Expr {
        constant_vec(self.to_vec())
    }
}

impl IntoConstant for DMatrix<f64> {
    fn into_constant(self) -> Expr {
        constant_dmatrix(self)
    }
}

impl IntoConstant for CscMatrix<f64> {
    fn into_constant(self) -> Expr {
        constant_sparse(self)
    }
}

/// Arc-wrapped constant for use in expressions.
pub fn const_arc(value: f64) -> Arc<Expr> {
    Arc::new(constant(value))
}

/// Arc-wrapped vector constant for use in expressions.
pub fn const_vec_arc(values: Vec<f64>) -> Arc<Expr> {
    Arc::new(constant_vec(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scalar() {
        let c = constant(5.0);
        if let Expr::Constant(data) = &c {
            assert_eq!(data.value.as_scalar(), Some(5.0));
        } else {
            panic!("Expected Constant");
        }
    }

    #[test]
    fn test_constant_vec() {
        let c = constant_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(c.shape(), Shape::matrix(3, 1));
    }

    #[test]
    fn test_zeros() {
        let z = zeros((3, 4));
        assert_eq!(z.shape(), Shape::matrix(3, 4));
        if let Expr::Constant(data) = &z {
            assert!(data.value.is_nonneg());
            assert!(data.value.is_nonpos());
        } else {
            panic!("Expected Constant");
        }
    }

    #[test]
    fn test_ones() {
        let o = ones(5);
        assert_eq!(o.shape(), Shape::matrix(5, 1));
    }

    #[test]
    fn test_eye() {
        let e = eye(3);
        assert_eq!(e.shape(), Shape::matrix(3, 3));
    }

    #[test]
    fn test_into_constant() {
        let _: Expr = 5.0.into_constant();
        let _: Expr = vec![1.0, 2.0].into_constant();
    }
}
