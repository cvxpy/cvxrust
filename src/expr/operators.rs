//! Arithmetic operator implementations for expressions.

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::atoms::broadcast::broadcast_exprs;

use super::{Expr, constant};

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
        add_exprs(self, rhs)
    }
}

impl Add for &Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Expr {
        add_exprs(self.clone(), rhs.clone())
    }
}

impl Add<&Expr> for Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Expr {
        add_exprs(self, rhs.clone())
    }
}

impl Add<Expr> for &Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        add_exprs(self.clone(), rhs)
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        sub_exprs(self, rhs)
    }
}

impl Sub for &Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Expr {
        sub_exprs(self.clone(), rhs.clone())
    }
}

impl Sub<&Expr> for Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Expr {
        sub_exprs(self, rhs.clone())
    }
}

impl Sub<Expr> for &Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        sub_exprs(self.clone(), rhs)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        mul_exprs(self, rhs)
    }
}

impl Mul for &Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        mul_exprs(self.clone(), rhs.clone())
    }
}

impl Mul<&Expr> for Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        mul_exprs(self, rhs.clone())
    }
}

impl Mul<Expr> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        mul_exprs(self.clone(), rhs)
    }
}

// Scalar multiplication
impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Expr {
        mul_exprs(constant(rhs), self)
    }
}

impl Mul<f64> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Expr {
        mul_exprs(constant(rhs), self.clone())
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        mul_exprs(constant(self), rhs)
    }
}

impl Mul<&Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        mul_exprs(constant(self), rhs.clone())
    }
}

// Division by scalar
impl Div<f64> for Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Expr {
        mul_exprs(constant(1.0 / rhs), self)
    }
}

impl Div<f64> for &Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Expr {
        mul_exprs(constant(1.0 / rhs), self.clone())
    }
}

fn add_exprs(lhs: Expr, rhs: Expr) -> Expr {
    let (lhs, rhs) = broadcast_exprs(lhs, rhs);
    Expr::Add(Arc::new(lhs), Arc::new(rhs))
}

fn sub_exprs(lhs: Expr, rhs: Expr) -> Expr {
    let (lhs, rhs) = broadcast_exprs(lhs, rhs);
    Expr::Add(Arc::new(lhs), Arc::new(Expr::Neg(Arc::new(rhs))))
}

fn mul_exprs(lhs: Expr, rhs: Expr) -> Expr {
    let (lhs, rhs) = broadcast_exprs(lhs, rhs);
    Expr::Mul(Arc::new(lhs), Arc::new(rhs))
}

#[cfg(test)]
mod tests {
    use crate::expr::{Shape, constant_matrix, constant_vec, variable};

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
    fn test_scalar_like_broadcast_shapes() {
        let x = variable((2, 2));
        let scalar = variable(());
        let row_scalar = constant_vec(vec![1.0]);
        let matrix_scalar = constant_matrix(vec![1.0], 1, 1);

        assert_eq!((&scalar + &x).shape(), Shape::matrix(2, 2));
        assert_eq!((&x - &row_scalar).shape(), Shape::matrix(2, 2));
        assert_eq!((&matrix_scalar * &x).shape(), Shape::matrix(2, 2));
    }

    #[test]
    fn test_row_and_column_broadcast_shapes() {
        let m = variable((2, 3));
        let row = variable((1, 3));
        let col = variable((2, 1));

        assert_eq!((&row + &m).shape(), Shape::matrix(2, 3));
        assert_eq!((&m - &row).shape(), Shape::matrix(2, 3));
        assert_eq!((&col * &m).shape(), Shape::matrix(2, 3));
        assert_eq!((&m + &col).shape(), Shape::matrix(2, 3));
    }

    #[test]
    fn test_mutual_row_and_column_broadcast_shape() {
        let row = variable((1, 3));
        let col = variable((2, 1));

        assert_eq!((&row + &col).shape(), Shape::matrix(2, 3));
        assert_eq!((&col * &row).shape(), Shape::matrix(2, 3));
    }

    #[test]
    #[should_panic(expected = "cannot broadcast shapes (2, 2) and (3, 3)")]
    fn test_incompatible_broadcast_panics_at_construction() {
        let c = constant_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let x = variable((3, 3));

        let _ = c * x;
    }
}
