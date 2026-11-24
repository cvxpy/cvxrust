/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Nonlinear atoms for convex optimization.
//!
//! These atoms have specific curvature properties (convex or concave)
//! and require DCP composition rules to be applied correctly.

use std::sync::Arc;

use crate::atoms::broadcast::broadcast_elementwise_exprs;
use crate::expr::Expr;

// ============================================================================
// Norms (all convex)
// ============================================================================

/// L1 norm: ||x||_1 = sum(|x_i|).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn norm1(x: &Expr) -> Expr {
    Expr::Norm1(Arc::new(x.clone()))
}

/// L2 norm: ||x||_2 = sqrt(sum(x_i^2)).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn norm2(x: &Expr) -> Expr {
    Expr::Norm2(Arc::new(x.clone()))
}

/// Infinity norm: ||x||_inf = max(|x_i|).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn norm_inf(x: &Expr) -> Expr {
    Expr::NormInf(Arc::new(x.clone()))
}

/// General p-norm: ||x||_p = (sum(|x_i|^p))^(1/p).
///
/// For p >= 1, this is convex. We support:
/// - p = 1: L1 norm
/// - p = 2: L2 norm
/// - p = f64::INFINITY: Infinity norm
///
/// General p-norm.
///
/// Currently supports p = 1, 2, or infinity.
///
/// # Panics
///
/// Panics if p is not 1, 2, or infinity. Use `try_norm()` for explicit error handling.
///
/// # Example
///
/// ```
/// use cvxrust::prelude::*;
///
/// let x = variable(5);
/// let n = norm(&x, 2.0);  // Same as norm2(&x)
/// ```
pub fn norm(x: &Expr, p: f64) -> Expr {
    try_norm(x, p).expect("unsupported norm p-value")
}

/// General p-norm, returning an error for unsupported p values.
///
/// This is the fallible version of `norm()`. Use this when you need
/// explicit error handling for the p parameter.
///
/// # Errors
///
/// Returns an error if p is not 1, 2, or infinity.
pub fn try_norm(x: &Expr, p: f64) -> crate::Result<Expr> {
    if p == 1.0 {
        Ok(norm1(x))
    } else if p == 2.0 {
        Ok(norm2(x))
    } else if p.is_infinite() {
        Ok(norm_inf(x))
    } else {
        Err(crate::CvxError::InvalidProblem(format!(
            "norm p={} is not supported; use p=1, 2, or inf",
            p
        )))
    }
}

// ============================================================================
// Element-wise atoms
// ============================================================================

/// Absolute value: |x| (element-wise).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn abs(x: &Expr) -> Expr {
    Expr::Abs(Arc::new(x.clone()))
}

/// Positive part: max(x, 0) (element-wise).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing
pub fn pos(x: &Expr) -> Expr {
    Expr::Pos(Arc::new(x.clone()))
}

/// Negative part: max(-x, 0) (element-wise).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Decreasing
pub fn neg_part(x: &Expr) -> Expr {
    Expr::NegPart(Arc::new(x.clone()))
}

// ============================================================================
// Maximum and minimum
// ============================================================================

/// Maximum of expressions (element-wise if same shape).
///
/// Properties:
/// - Curvature: Convex (when all arguments are convex)
/// - Sign: Depends on arguments
/// - Monotonicity: Increasing in all arguments
pub fn maximum(exprs: Vec<Expr>) -> Expr {
    if exprs.is_empty() {
        panic!("maximum requires at least one expression");
    }
    if exprs.len() == 1 {
        return exprs.into_iter().next().unwrap();
    }

    let (_, exprs) = broadcast_elementwise_exprs(exprs);
    Expr::Maximum(exprs.into_iter().map(Arc::new).collect())
}

/// Maximum of two expressions.
pub fn max2(a: &Expr, b: &Expr) -> Expr {
    maximum(vec![a.clone(), b.clone()])
}

/// Minimum of expressions (element-wise if same shape).
///
/// Properties:
/// - Curvature: Concave (when all arguments are concave)
/// - Sign: Depends on arguments
/// - Monotonicity: Increasing in all arguments
pub fn minimum(exprs: Vec<Expr>) -> Expr {
    if exprs.is_empty() {
        panic!("minimum requires at least one expression");
    }
    if exprs.len() == 1 {
        return exprs.into_iter().next().unwrap();
    }

    let (_, exprs) = broadcast_elementwise_exprs(exprs);
    Expr::Minimum(exprs.into_iter().map(Arc::new).collect())
}

/// Minimum of two expressions.
pub fn min2(a: &Expr, b: &Expr) -> Expr {
    minimum(vec![a.clone(), b.clone()])
}

// ============================================================================
// Quadratic atoms
// ============================================================================

/// Quadratic form xᵀPx.
///
/// `x` must be affine and scalar, a vector, or a column matrix. `P` must be a
/// matching constant symmetric matrix; scalar `P` denotes a 1×1 matrix.
/// Convex and nonnegative for PSD `P`; concave and nonpositive for NSD `P`.
///
/// DCP checking supports scalar constants and definite dense matrices.
pub fn quad_form(x: &Expr, p: &Expr) -> Expr {
    Expr::QuadForm(Arc::new(x.clone()), Arc::new(p.clone()))
}

/// Sum of squares: ||x||_2^2 = x' x.
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
///
/// This is equivalent to quad_form(x, I) where I is identity.
pub fn sum_squares(x: &Expr) -> Expr {
    Expr::SumSquares(Arc::new(x.clone()))
}

/// Quadratic over linear: ||x||_2^2 / y.
///
/// Properties:
/// - Curvature: Convex (when x is affine and y is concave and positive)
/// - Sign: Non-negative
/// - Domain: y > 0
///
/// This is a perspective function and is jointly convex in (x, y).
pub fn quad_over_lin(x: &Expr, y: &Expr) -> Expr {
    Expr::QuadOverLin(Arc::new(x.clone()), Arc::new(y.clone()))
}

/// Exponential function (elementwise): exp(x)
///
/// Convex when x is affine.
pub fn exp(x: &Expr) -> Expr {
    Expr::Exp(Arc::new(x.clone()))
}

/// Natural logarithm (elementwise): log(x)
///
/// Concave when x is concave (and positive).
pub fn log(x: &Expr) -> Expr {
    Expr::Log(Arc::new(x.clone()))
}

/// Entropy (elementwise): -x * log(x)
///
/// Concave when x is affine (and positive).
/// Note: v1.0 implementation is simplified for scalar/small vectors.
pub fn entropy(x: &Expr) -> Expr {
    Expr::Entropy(Arc::new(x.clone()))
}

/// Power function (elementwise): x^p
///
/// - p > 1 or p < 0: Convex when x is affine and nonnegative
/// - 0 < p < 1: Concave when x is affine and nonnegative
/// - p = 1: Affine
/// - p = 2: Same as sum_squares (but less efficient)
///
/// Uses native power cones for better performance than exp/log reformulation.
pub fn power(x: &Expr, p: f64) -> Expr {
    Expr::Power(Arc::new(x.clone()), p)
}

/// Square root: sqrt(x) = x^0.5
///
/// Concave when x is affine and nonnegative.
pub fn sqrt(x: &Expr) -> Expr {
    power(x, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcp::Curvature;
    use crate::expr::variable;

    #[test]
    fn test_norm2_convex() {
        let x = variable(5);
        let n = norm2(&x);
        assert_eq!(n.curvature(), Curvature::Convex);
        assert!(n.is_nonneg());
    }

    #[test]
    fn test_norm1_convex() {
        let x = variable(5);
        let n = norm1(&x);
        assert_eq!(n.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_abs_convex() {
        let x = variable(5);
        let a = abs(&x);
        assert_eq!(a.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_pos_convex() {
        let x = variable(5);
        let p = pos(&x);
        assert_eq!(p.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_maximum_convex() {
        let x = variable(5);
        let y = variable(5);
        let m = maximum(vec![x, y]);
        assert_eq!(m.curvature(), Curvature::Convex);
    }

    #[test]
    #[should_panic(expected = "maximum requires at least one expression")]
    fn test_maximum_empty_panics() {
        let _ = maximum(Vec::new());
    }

    #[test]
    fn test_maximum_single_expression_is_identity() {
        let x = variable(5);
        let m = maximum(vec![x.clone()]);
        assert_eq!(m.shape(), x.shape());
        assert_eq!(m.variables(), x.variables());
    }

    #[test]
    fn test_maximum_broadcasts_scalar_to_vector() {
        let x = variable(5);
        let y = variable(());
        let m = maximum(vec![x, y]);
        assert_eq!(m.shape(), crate::expr::Shape::vector(5));
    }

    #[test]
    fn test_maximum_three_args_stays_flat() {
        let x = variable(5);
        let y = variable(5);
        let z = variable(());
        let m = maximum(vec![x, y, z]);

        match m {
            Expr::Maximum(args) => {
                assert_eq!(args.len(), 3);
                assert!(
                    args.iter()
                        .all(|arg| arg.shape() == crate::expr::Shape::vector(5))
                );
                assert!(!matches!(&*args[0], Expr::Maximum(_)));
            }
            _ => panic!("expected maximum expression"),
        }
    }

    #[test]
    #[should_panic(expected = "cannot broadcast shapes (2,) and (3,)")]
    fn test_maximum_incompatible_shapes_panic() {
        let x = variable(2);
        let y = variable(3);
        let _ = maximum(vec![x, y]);
    }

    #[test]
    fn test_minimum_concave() {
        let x = variable(5);
        let y = variable(5);
        let m = minimum(vec![x, y]);
        assert_eq!(m.curvature(), Curvature::Concave);
    }

    #[test]
    #[should_panic(expected = "minimum requires at least one expression")]
    fn test_minimum_empty_panics() {
        let _ = minimum(Vec::new());
    }

    #[test]
    fn test_minimum_single_expression_is_identity() {
        let x = variable(5);
        let m = minimum(vec![x.clone()]);
        assert_eq!(m.shape(), x.shape());
        assert_eq!(m.variables(), x.variables());
    }

    #[test]
    fn test_minimum_broadcasts_scalar_to_vector() {
        let x = variable(5);
        let y = variable(());
        let m = minimum(vec![x, y]);
        assert_eq!(m.shape(), crate::expr::Shape::vector(5));
    }

    #[test]
    fn test_minimum_three_args_stays_flat() {
        let x = variable(5);
        let y = variable(5);
        let z = variable(());
        let m = minimum(vec![x, y, z]);

        match m {
            Expr::Minimum(args) => {
                assert_eq!(args.len(), 3);
                assert!(
                    args.iter()
                        .all(|arg| arg.shape() == crate::expr::Shape::vector(5))
                );
                assert!(!matches!(&*args[0], Expr::Minimum(_)));
            }
            _ => panic!("expected minimum expression"),
        }
    }

    #[test]
    #[should_panic(expected = "cannot broadcast shapes (2,) and (3,)")]
    fn test_minimum_incompatible_shapes_panic() {
        let x = variable(2);
        let y = variable(3);
        let _ = minimum(vec![x, y]);
    }

    #[test]
    fn test_sum_squares_convex() {
        let x = variable(5);
        let s = sum_squares(&x);
        assert_eq!(s.curvature(), Curvature::Convex);
        assert!(s.is_nonneg());
    }

    #[test]
    fn test_quad_form_psd() {
        use nalgebra::DMatrix;
        let x = variable(2);
        // Identity matrix is PSD
        let p = crate::expr::constant_dmatrix(DMatrix::identity(2, 2));
        let q = quad_form(&x, &p);
        assert_eq!(q.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_norm_of_affine_is_convex() {
        let x = variable(5);
        let y = variable(5);
        // x + y is affine
        let z = &x + &y;
        let n = norm2(&z);
        // norm2 of affine is convex
        assert_eq!(n.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_norm_of_convex_is_unknown() {
        let x = variable(5);
        let n1 = norm2(&x);
        // norm2(norm2(x)) - norm of convex is unknown (not DCP)
        let n2 = norm2(&n1);
        assert_eq!(n2.curvature(), Curvature::Unknown);
    }
}
