/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Curvature tracking for DCP (Disciplined Convex Programming).
//!
//! This module implements the curvature rules that determine whether an
//! expression is convex, concave, affine, or unknown.

use std::sync::Arc;

use crate::expr::{Array, Expr};

/// Curvature of an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Curvature {
    /// Constant value (most restrictive).
    Constant,
    /// Affine function (both convex and concave).
    Affine,
    /// Convex function.
    Convex,
    /// Concave function.
    Concave,
    /// Unknown curvature (not DCP-compliant).
    Unknown,
}

impl Curvature {
    /// Check if the curvature is convex (constant, affine, or convex).
    pub fn is_convex(self) -> bool {
        matches!(
            self,
            Curvature::Constant | Curvature::Affine | Curvature::Convex
        )
    }

    /// Check if the curvature is concave (constant, affine, or concave).
    pub fn is_concave(self) -> bool {
        matches!(
            self,
            Curvature::Constant | Curvature::Affine | Curvature::Concave
        )
    }

    /// Check if the curvature is affine (constant or affine).
    pub fn is_affine(self) -> bool {
        matches!(self, Curvature::Constant | Curvature::Affine)
    }

    /// Check if this is a constant.
    pub fn is_constant(self) -> bool {
        matches!(self, Curvature::Constant)
    }

    /// Negate the curvature (convex <-> concave).
    pub fn negate(self) -> Self {
        match self {
            Curvature::Convex => Curvature::Concave,
            Curvature::Concave => Curvature::Convex,
            other => other,
        }
    }
}

/// Combine curvatures for addition: a + b.
pub fn add_curvature(a: Curvature, b: Curvature) -> Curvature {
    use Curvature::*;
    match (a, b) {
        // Constants don't affect curvature
        (Constant, x) | (x, Constant) => x,
        // Affine + Affine = Affine
        (Affine, Affine) => Affine,
        // Affine doesn't affect non-constant curvature
        (Affine, x) | (x, Affine) => x,
        // Convex + Convex = Convex
        (Convex, Convex) => Convex,
        // Concave + Concave = Concave
        (Concave, Concave) => Concave,
        // Convex + Concave = Unknown
        (Convex, Concave) | (Concave, Convex) => Unknown,
        // Unknown propagates
        (Unknown, _) | (_, Unknown) => Unknown,
    }
}

/// Combine curvatures for scalar multiplication: scalar * expr.
///
/// If scalar > 0: preserves curvature
/// If scalar < 0: negates curvature
/// If scalar == 0: constant
pub fn scalar_mul_curvature(scalar: f64, expr_curv: Curvature) -> Curvature {
    if scalar == 0.0 {
        Curvature::Constant
    } else if scalar > 0.0 {
        expr_curv
    } else {
        expr_curv.negate()
    }
}

/// Determine if a matrix is PSD, NSD, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsdStatus {
    Psd,     // Positive semi-definite
    Nsd,     // Negative semi-definite
    Neither, // Indefinite or non-symmetric
}

impl PsdStatus {
    /// Determine PSD status of an array.
    pub fn of_array(arr: &Array) -> Self {
        match arr.is_psd() {
            Some(true) => PsdStatus::Psd,
            Some(false) => {
                // Check if negative of it is PSD
                match arr {
                    Array::Scalar(v) => {
                        if *v <= 0.0 {
                            PsdStatus::Nsd
                        } else {
                            PsdStatus::Neither
                        }
                    }
                    Array::Dense(m) => {
                        // Check if -M is PSD
                        let neg = -m.clone();
                        if neg.cholesky().is_some() {
                            PsdStatus::Nsd
                        } else {
                            PsdStatus::Neither
                        }
                    }
                    _ => PsdStatus::Neither,
                }
            }
            None => PsdStatus::Neither,
        }
    }
}

impl Expr {
    /// Get the curvature of this expression.
    pub fn curvature(&self) -> Curvature {
        match self {
            // Leaves
            Expr::Variable(_) => Curvature::Affine,
            Expr::Constant(_) => Curvature::Constant,

            // Affine operations preserve/combine curvatures
            Expr::Add(a, b) => add_curvature(a.curvature(), b.curvature()),
            Expr::Neg(a) => a.curvature().negate(),
            Expr::Mul(a, b) => mul_curvature(a, b),
            Expr::MatMul(a, b) => matmul_curvature(a, b),
            Expr::Promote(a, _) => a.curvature(),
            Expr::Sum(a, _) => a.curvature(),
            Expr::Reshape(a, _) => a.curvature(),
            Expr::Index(a, _) => a.curvature(),
            Expr::VStack(exprs) => combine_all_curvatures(exprs),
            Expr::HStack(exprs) => combine_all_curvatures(exprs),
            Expr::Transpose(a) => a.curvature(),
            Expr::Trace(a) => a.curvature(),

            // Nonlinear convex atoms
            Expr::Norm1(x) | Expr::Norm2(x) | Expr::NormInf(x) => {
                // Convex when argument is affine
                if x.curvature().is_affine() {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Abs(x) => {
                // Convex when argument is affine
                if x.curvature().is_affine() {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Pos(x) => {
                // pos(x) = max(x, 0) is convex
                // Convex when x is affine OR when x is convex
                if x.curvature().is_convex() {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::NegPart(x) => {
                // neg(x) = max(-x, 0) is convex
                // Convex when x is affine OR when x is concave
                if x.curvature().is_concave() {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Maximum(exprs) => {
                // max(x1, ..., xn) is convex if all xi are convex
                if exprs.iter().all(|e| e.curvature().is_convex()) {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Minimum(exprs) => {
                // min(x1, ..., xn) is concave if all xi are concave
                if exprs.iter().all(|e| e.curvature().is_concave()) {
                    Curvature::Concave
                } else {
                    Curvature::Unknown
                }
            }
            Expr::QuadForm(x, p) => {
                // x'Px: convex if P is PSD and x is affine
                //       concave if P is NSD and x is affine
                if !x.curvature().is_affine() {
                    return Curvature::Unknown;
                }
                if let Some(p_val) = p.constant_value() {
                    match PsdStatus::of_array(p_val) {
                        PsdStatus::Psd => Curvature::Convex,
                        PsdStatus::Nsd => Curvature::Concave,
                        PsdStatus::Neither => Curvature::Unknown,
                    }
                } else {
                    Curvature::Unknown
                }
            }
            Expr::SumSquares(x) => {
                // ||x||_2^2 is convex when x is affine
                if x.curvature().is_affine() {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::QuadOverLin(x, y) => {
                // ||x||_2^2 / y is convex when:
                // - x is affine
                // - y is concave and positive
                if x.curvature().is_affine() && y.curvature().is_concave() {
                    // Note: We'd need to check y > 0 at runtime
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }

            // Exponential cone atoms
            Expr::Exp(x) => {
                // exp(x) is convex when x is affine
                if x.curvature().is_affine() {
                    Curvature::Convex
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Log(x) => {
                // log(x) is concave when x is concave (and positive)
                if x.curvature().is_concave() {
                    Curvature::Concave
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Entropy(x) => {
                // -x*log(x) is concave when x is affine (and positive)
                if x.curvature().is_affine() {
                    Curvature::Concave
                } else {
                    Curvature::Unknown
                }
            }
            Expr::Power(x, p) => {
                // x^p curvature depends on p and sign of x
                if *p == 0.0 {
                    // x^0 = 1 is constant
                    Curvature::Constant
                } else if *p == 1.0 {
                    // x^1 = x is affine
                    x.curvature()
                } else if *p == 2.0 {
                    // x^2 is convex when x is affine (same as sum_squares)
                    if x.curvature().is_affine() {
                        Curvature::Convex
                    } else {
                        Curvature::Unknown
                    }
                } else if *p > 1.0 || *p < 0.0 {
                    // x^p is convex for p > 1 or p < 0 when x is concave and nonneg
                    // For DCP we require x to be affine for simplicity
                    if x.curvature().is_affine() {
                        Curvature::Convex
                    } else {
                        Curvature::Unknown
                    }
                } else if *p > 0.0 && *p < 1.0 {
                    // x^p is concave for 0 < p < 1 when x is convex and nonneg
                    // For DCP we require x to be affine
                    if x.curvature().is_affine() {
                        Curvature::Concave
                    } else {
                        Curvature::Unknown
                    }
                } else {
                    // Other edge cases (NaN, etc.)
                    Curvature::Unknown
                }
            }

            // Additional affine atoms
            Expr::Cumsum(x, _) => x.curvature(), // Affine operation
            Expr::Diag(x) => x.curvature(),      // Affine operation
        }
    }

    /// Check if this expression is convex.
    pub fn is_convex(&self) -> bool {
        self.curvature().is_convex()
    }

    /// Check if this expression is concave.
    pub fn is_concave(&self) -> bool {
        self.curvature().is_concave()
    }

    /// Check if this expression is affine.
    pub fn is_affine(&self) -> bool {
        self.curvature().is_affine()
    }
}

/// Handle multiplication curvature.
fn mul_curvature(a: &Expr, b: &Expr) -> Curvature {
    let ac = a.curvature();
    let bc = b.curvature();

    // If both are constant, result is constant
    if ac.is_constant() && bc.is_constant() {
        return Curvature::Constant;
    }

    // If one is constant, the other determines curvature (scaled)
    if ac.is_constant() {
        if let Some(arr) = a.constant_value() {
            if let Some(scalar) = arr.as_scalar() {
                return scalar_mul_curvature(scalar, bc);
            }
        }
        // Non-scalar constant * expression: only affine if expression is affine
        if bc.is_affine() {
            return Curvature::Affine;
        }
        return Curvature::Unknown;
    }

    if bc.is_constant() {
        if let Some(arr) = b.constant_value() {
            if let Some(scalar) = arr.as_scalar() {
                return scalar_mul_curvature(scalar, ac);
            }
        }
        // Expression * non-scalar constant: only affine if expression is affine
        if ac.is_affine() {
            return Curvature::Affine;
        }
        return Curvature::Unknown;
    }

    // Both non-constant: only affine * affine gives meaningful result
    // But affine * affine is quadratic (not affine unless one is scalar constant)
    Curvature::Unknown
}

/// Handle matrix multiplication curvature.
fn matmul_curvature(a: &Expr, b: &Expr) -> Curvature {
    let ac = a.curvature();
    let bc = b.curvature();

    // Constant @ anything = preserves curvature
    if ac.is_constant() {
        return bc;
    }
    // Anything @ constant = preserves curvature
    if bc.is_constant() {
        return ac;
    }

    // affine @ affine = quadratic (not DCP unless one side is constant)
    Curvature::Unknown
}

/// Combine curvatures for stacking operations.
fn combine_all_curvatures(exprs: &[Arc<Expr>]) -> Curvature {
    if exprs.is_empty() {
        return Curvature::Constant;
    }

    let mut result = Curvature::Constant;
    for e in exprs {
        result = add_curvature(result, e.curvature());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, variable};

    #[test]
    fn test_curvature_basics() {
        assert!(Curvature::Constant.is_convex());
        assert!(Curvature::Constant.is_concave());
        assert!(Curvature::Constant.is_affine());

        assert!(Curvature::Affine.is_convex());
        assert!(Curvature::Affine.is_concave());
        assert!(Curvature::Affine.is_affine());

        assert!(Curvature::Convex.is_convex());
        assert!(!Curvature::Convex.is_concave());
        assert!(!Curvature::Convex.is_affine());

        assert!(!Curvature::Concave.is_convex());
        assert!(Curvature::Concave.is_concave());
        assert!(!Curvature::Concave.is_affine());
    }

    #[test]
    fn test_negate_curvature() {
        assert_eq!(Curvature::Convex.negate(), Curvature::Concave);
        assert_eq!(Curvature::Concave.negate(), Curvature::Convex);
        assert_eq!(Curvature::Affine.negate(), Curvature::Affine);
        assert_eq!(Curvature::Constant.negate(), Curvature::Constant);
    }

    #[test]
    fn test_add_curvature() {
        use Curvature::*;
        assert_eq!(add_curvature(Convex, Convex), Convex);
        assert_eq!(add_curvature(Concave, Concave), Concave);
        assert_eq!(add_curvature(Affine, Affine), Affine);
        assert_eq!(add_curvature(Convex, Affine), Convex);
        assert_eq!(add_curvature(Concave, Affine), Concave);
        assert_eq!(add_curvature(Convex, Concave), Unknown);
    }

    #[test]
    fn test_variable_is_affine() {
        let x = variable(5);
        assert!(x.is_affine());
        assert!(x.is_convex());
        assert!(x.is_concave());
    }

    #[test]
    fn test_constant_is_constant() {
        let c = constant(5.0);
        assert_eq!(c.curvature(), Curvature::Constant);
    }

    #[test]
    fn test_norm_is_convex() {
        let x = variable(5);
        let n = Expr::Norm2(Arc::new(x));
        assert_eq!(n.curvature(), Curvature::Convex);
        assert!(n.is_convex());
        assert!(!n.is_concave());
    }

    #[test]
    fn test_neg_flips_curvature() {
        let x = variable(5);
        let n = Expr::Norm2(Arc::new(x));
        let neg_n = Expr::Neg(Arc::new(n));
        assert_eq!(neg_n.curvature(), Curvature::Concave);
    }
}
