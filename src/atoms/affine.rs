//! Affine atoms and operator overloading.
//!
//! Affine atoms are both convex and concave. They include:
//! - Addition, subtraction, negation
//! - Scalar and matrix multiplication
//! - Sum, reshape, index, stack operations
//! - Transpose and trace

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::expr::{Expr, Shape, constant};

// ============================================================================
// Operator overloading for Expr
// ============================================================================

impl Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::Neg(Arc::new(self))
    }
}

impl Neg for &Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::Neg(Arc::new(self.clone()))
    }
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(rhs))
    }
}

impl Add for &Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Expr {
        Expr::Add(Arc::new(self.clone()), Arc::new(rhs.clone()))
    }
}

impl Add<&Expr> for Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(rhs.clone()))
    }
}

impl Add<Expr> for &Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(self.clone()), Arc::new(rhs))
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(Expr::Neg(Arc::new(rhs))))
    }
}

impl Sub for &Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Expr {
        Expr::Add(
            Arc::new(self.clone()),
            Arc::new(Expr::Neg(Arc::new(rhs.clone()))),
        )
    }
}

impl Sub<&Expr> for Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(Expr::Neg(Arc::new(rhs.clone()))))
    }
}

impl Sub<Expr> for &Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(self.clone()), Arc::new(Expr::Neg(Arc::new(rhs))))
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Arc::new(self), Arc::new(rhs))
    }
}

impl Mul for &Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        Expr::Mul(Arc::new(self.clone()), Arc::new(rhs.clone()))
    }
}

impl Mul<&Expr> for Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        Expr::Mul(Arc::new(self), Arc::new(rhs.clone()))
    }
}

impl Mul<Expr> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Arc::new(self.clone()), Arc::new(rhs))
    }
}

// Scalar multiplication
impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Expr {
        Expr::Mul(Arc::new(constant(rhs)), Arc::new(self))
    }
}

impl Mul<f64> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Expr {
        Expr::Mul(Arc::new(constant(rhs)), Arc::new(self.clone()))
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Arc::new(constant(self)), Arc::new(rhs))
    }
}

impl Mul<&Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        Expr::Mul(Arc::new(constant(self)), Arc::new(rhs.clone()))
    }
}

// Division by scalar
impl Div<f64> for Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Expr {
        Expr::Mul(Arc::new(constant(1.0 / rhs)), Arc::new(self))
    }
}

impl Div<f64> for &Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Expr {
        Expr::Mul(Arc::new(constant(1.0 / rhs)), Arc::new(self.clone()))
    }
}

// ============================================================================
// Affine atom functions
// ============================================================================

/// Sum of all elements, or along an axis.
pub fn sum(expr: &Expr) -> Expr {
    Expr::Sum(Arc::new(expr.clone()), None)
}

/// Sum along a specific axis.
pub fn sum_axis(expr: &Expr, axis: usize) -> Expr {
    Expr::Sum(Arc::new(expr.clone()), Some(axis))
}

/// Reshape an expression to a new shape.
pub fn reshape(expr: &Expr, shape: impl Into<Shape>) -> Expr {
    Expr::Reshape(Arc::new(expr.clone()), shape.into())
}

/// Flatten an expression to a vector.
pub fn flatten(expr: &Expr) -> Expr {
    let size = expr.shape().size();
    Expr::Reshape(Arc::new(expr.clone()), Shape::vector(size))
}

/// Transpose an expression.
pub fn transpose(expr: &Expr) -> Expr {
    Expr::Transpose(Arc::new(expr.clone()))
}

/// Matrix trace.
pub fn trace(expr: &Expr) -> Expr {
    Expr::Trace(Arc::new(expr.clone()))
}

/// Vertical stack (row-wise concatenation).
pub fn vstack(exprs: Vec<Expr>) -> Expr {
    Expr::VStack(exprs.into_iter().map(Arc::new).collect())
}

/// Horizontal stack (column-wise concatenation).
pub fn hstack(exprs: Vec<Expr>) -> Expr {
    Expr::HStack(exprs.into_iter().map(Arc::new).collect())
}

/// Matrix-vector or matrix-matrix multiplication.
pub fn matmul(a: &Expr, b: &Expr) -> Expr {
    Expr::MatMul(Arc::new(a.clone()), Arc::new(b.clone()))
}

/// Dot product (inner product) of two vectors.
pub fn dot(a: &Expr, b: &Expr) -> Expr {
    // dot(a, b) = sum(a * b) for element-wise product
    // or a'b for vector dot product
    Expr::MatMul(
        Arc::new(Expr::Transpose(Arc::new(a.clone()))),
        Arc::new(b.clone()),
    )
}

/// Index into an expression.
pub fn index(expr: &Expr, idx: usize) -> Expr {
    use crate::expr::IndexSpec;
    Expr::Index(Arc::new(expr.clone()), IndexSpec::element(vec![idx]))
}

/// Slice a range from an expression.
pub fn slice(expr: &Expr, start: usize, stop: usize) -> Expr {
    use crate::expr::IndexSpec;
    Expr::Index(Arc::new(expr.clone()), IndexSpec::range(start, stop))
}

/// Cumulative sum along an axis.
///
/// Returns cumsum([x1, x2, x3]) = [x1, x1+x2, x1+x2+x3]
pub fn cumsum(expr: &Expr) -> Expr {
    Expr::Cumsum(Arc::new(expr.clone()), None)
}

/// Diagonal matrix from vector, or diagonal of matrix.
///
/// - Vector input: Creates diagonal matrix with vector on diagonal
/// - Matrix input: Extracts diagonal as vector (v1.0: returns input as fallback)
pub fn diag(expr: &Expr) -> Expr {
    Expr::Diag(Arc::new(expr.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, variable};

    #[test]
    fn test_add() {
        let x = variable(5);
        let y = variable(5);
        let z = &x + &y;
        assert_eq!(z.shape(), Shape::vector(5));
    }

    #[test]
    fn test_sub() {
        let x = variable(5);
        let y = variable(5);
        let z = &x - &y;
        assert_eq!(z.shape(), Shape::vector(5));
    }

    #[test]
    fn test_neg() {
        let x = variable(5);
        let z = -&x;
        assert_eq!(z.shape(), Shape::vector(5));
    }

    #[test]
    fn test_scalar_mul() {
        let x = variable(5);
        let z = 2.0 * &x;
        assert_eq!(z.shape(), Shape::vector(5));

        let z = &x * 2.0;
        assert_eq!(z.shape(), Shape::vector(5));
    }

    #[test]
    fn test_sum() {
        let x = variable((3, 4));
        let s = sum(&x);
        assert_eq!(s.shape(), Shape::scalar());
    }

    #[test]
    fn test_transpose() {
        let x = variable((3, 4));
        let t = transpose(&x);
        assert_eq!(t.shape(), Shape::matrix(4, 3));
    }

    #[test]
    fn test_matmul() {
        let a = variable((3, 4));
        let x = variable(4);
        let b = matmul(&a, &x);
        assert_eq!(b.shape(), Shape::vector(3));
    }

    #[test]
    fn test_vstack() {
        let x = variable((2, 3));
        let y = variable((3, 3));
        let z = vstack(vec![x, y]);
        assert_eq!(z.shape(), Shape::matrix(5, 3));
    }

    #[test]
    fn test_affine_is_affine() {
        let x = variable(5);
        let y = variable(5);
        let _c = constant(2.0);

        // x + y is affine
        let z = &x + &y;
        assert!(z.is_affine());

        // 2*x is affine
        let z = 2.0 * &x;
        assert!(z.is_affine());

        // sum(x) is affine
        let s = sum(&x);
        assert!(s.is_affine());
    }
}
