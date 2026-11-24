/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Sign tracking for DCP (Disciplined Convex Programming).
//!
//! This module tracks whether expressions are non-negative, non-positive,
//! or have unknown sign. Sign information is used in DCP composition rules.

use std::sync::Arc;

use crate::expr::Expr;
use crate::expr::semidefinite::{self, Definiteness};

/// Sign of an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Expression is always `>= 0`.
    Nonnegative,
    /// Expression is always `<= 0`.
    Nonpositive,
    /// Expression is always `== 0`.
    Zero,
    /// Sign is unknown.
    Unknown,
}

impl Sign {
    /// Check if the sign is non-negative (`>= 0`).
    pub fn is_nonneg(self) -> bool {
        matches!(self, Sign::Nonnegative | Sign::Zero)
    }

    /// Check if the sign is non-positive (`<= 0`).
    pub fn is_nonpos(self) -> bool {
        matches!(self, Sign::Nonpositive | Sign::Zero)
    }

    /// Check if the sign is zero.
    pub fn is_zero(self) -> bool {
        matches!(self, Sign::Zero)
    }

    /// Negate the sign.
    pub fn negate(self) -> Self {
        match self {
            Sign::Nonnegative => Sign::Nonpositive,
            Sign::Nonpositive => Sign::Nonnegative,
            Sign::Zero => Sign::Zero,
            Sign::Unknown => Sign::Unknown,
        }
    }
}

/// Combine signs for addition: `a + b`.
pub fn add_sign(a: Sign, b: Sign) -> Sign {
    use Sign::*;
    match (a, b) {
        // Zero doesn't change sign
        (Zero, x) | (x, Zero) => x,
        // Same signs combine
        (Nonnegative, Nonnegative) => Nonnegative,
        (Nonpositive, Nonpositive) => Nonpositive,
        // Different signs -> unknown
        (Nonnegative, Nonpositive) | (Nonpositive, Nonnegative) => Unknown,
        // Unknown propagates
        (Unknown, _) | (_, Unknown) => Unknown,
    }
}

/// Combine signs for multiplication: `a * b`.
pub fn mul_sign(a: Sign, b: Sign) -> Sign {
    use Sign::*;
    match (a, b) {
        // Zero times anything is zero
        (Zero, _) | (_, Zero) => Zero,
        // Same sign -> nonnegative
        (Nonnegative, Nonnegative) | (Nonpositive, Nonpositive) => Nonnegative,
        // Different signs -> nonpositive
        (Nonnegative, Nonpositive) | (Nonpositive, Nonnegative) => Nonpositive,
        // Unknown propagates
        (Unknown, _) | (_, Unknown) => Unknown,
    }
}

impl Expr {
    /// Get the sign of this expression.
    pub fn sign(&self) -> Sign {
        match self {
            // Leaves
            Expr::Variable(v) => {
                if v.nonneg {
                    Sign::Nonnegative
                } else if v.nonpos {
                    Sign::Nonpositive
                } else {
                    Sign::Unknown
                }
            }
            Expr::Constant(c) => {
                if c.value.is_nonneg() && c.value.is_nonpos() {
                    Sign::Zero
                } else if c.value.is_nonneg() {
                    Sign::Nonnegative
                } else if c.value.is_nonpos() {
                    Sign::Nonpositive
                } else {
                    Sign::Unknown
                }
            }

            // Affine operations
            Expr::Add(a, b) => add_sign(a.sign(), b.sign()),
            Expr::Neg(a) => a.sign().negate(),
            Expr::Mul(a, b) => mul_sign(a.sign(), b.sign()),
            Expr::MatMul(a, b) => {
                // Matrix multiplication sign is complex; be conservative
                let as_ = a.sign();
                let bs = b.sign();
                if as_.is_zero() || bs.is_zero() {
                    Sign::Zero
                } else if (as_.is_nonneg() && bs.is_nonneg()) || (as_.is_nonpos() && bs.is_nonpos())
                {
                    Sign::Nonnegative
                } else {
                    Sign::Unknown
                }
            }
            Expr::Sum(a, _) => a.sign(),
            Expr::Promote(a, _) => a.sign(),
            Expr::Reshape(a, _) => a.sign(),
            Expr::Index(a, _) => a.sign(),
            Expr::VStack(exprs) => combine_signs(exprs),
            Expr::HStack(exprs) => combine_signs(exprs),
            Expr::Transpose(a) => a.sign(),
            Expr::Trace(a) => a.sign(),

            // Nonlinear atoms with known signs
            Expr::Norm1(_) | Expr::Norm2(_) | Expr::NormInf(_) => Sign::Nonnegative,
            Expr::Abs(_) => Sign::Nonnegative,
            Expr::Pos(_) => Sign::Nonnegative,
            Expr::NegPart(_) => Sign::Nonnegative,
            Expr::Maximum(exprs) => {
                // max is nonneg if any arg is nonneg
                if exprs.iter().any(|e| e.sign().is_nonneg()) {
                    Sign::Nonnegative
                } else if exprs.iter().all(|e| e.sign().is_nonpos()) {
                    Sign::Nonpositive
                } else {
                    Sign::Unknown
                }
            }
            Expr::Minimum(exprs) => {
                // min is nonpos if any arg is nonpos
                if exprs.iter().any(|e| e.sign().is_nonpos()) {
                    Sign::Nonpositive
                } else if exprs.iter().all(|e| e.sign().is_nonneg()) {
                    Sign::Nonnegative
                } else {
                    Sign::Unknown
                }
            }
            Expr::QuadForm(_, p) => {
                // x^T P x is nonneg if P is PSD, nonpos if P is NSD
                match p
                    .constant_value()
                    .and_then(semidefinite::analyze)
                    .map(|a| a.kind)
                {
                    Some(Definiteness::Zero) => Sign::Zero,
                    Some(Definiteness::Psd) => Sign::Nonnegative,
                    Some(Definiteness::Nsd) => Sign::Nonpositive,
                    _ => Sign::Unknown,
                }
            }
            Expr::SumSquares(_) => Sign::Nonnegative,
            Expr::QuadOverLin(_, _) => Sign::Nonnegative,

            // Exponential cone atoms
            Expr::Exp(_) => Sign::Nonnegative, // exp(x) > 0 always
            Expr::Log(_x) => {
                // log(x) can be positive or negative depending on whether x > 1 or x < 1
                // Conservative: Unknown
                // Could check if x is a constant > 1 or < 1
                Sign::Unknown
            }
            Expr::Entropy(_) => {
                // -x*log(x) for x in [0,1] is nonneg, for x > 1 could be neg
                // Conservative: Unknown
                Sign::Unknown
            }
            Expr::Power(x, p) => {
                // x^p sign depends on p and x
                if *p > 0.0 {
                    // x^p for p > 0 is nonneg when x is nonneg
                    if x.sign().is_nonneg() {
                        Sign::Nonnegative
                    } else {
                        Sign::Unknown
                    }
                } else if *p < 0.0 {
                    // x^p for p < 0 is nonneg when x is nonneg (and x > 0)
                    if x.sign().is_nonneg() {
                        Sign::Nonnegative
                    } else {
                        Sign::Unknown
                    }
                } else {
                    // p = 0 means x^0 = 1
                    Sign::Nonnegative
                }
            }

            // Additional affine atoms
            Expr::Cumsum(x, _) => x.sign(), // Cumsum preserves sign
            Expr::Diag(x) => x.sign(),      // Diag preserves sign
        }
    }

    /// Check if this expression is non-negative.
    pub fn is_nonneg(&self) -> bool {
        self.sign().is_nonneg()
    }

    /// Check if this expression is non-positive.
    pub fn is_nonpos(&self) -> bool {
        self.sign().is_nonpos()
    }
}

/// Combine signs for stacking/concatenation.
fn combine_signs(exprs: &[Arc<Expr>]) -> Sign {
    if exprs.is_empty() {
        return Sign::Zero;
    }

    let mut all_nonneg = true;
    let mut all_nonpos = true;
    let mut all_zero = true;

    for e in exprs {
        let s = e.sign();
        if !s.is_nonneg() {
            all_nonneg = false;
        }
        if !s.is_nonpos() {
            all_nonpos = false;
        }
        if !s.is_zero() {
            all_zero = false;
        }
    }

    if all_zero {
        Sign::Zero
    } else if all_nonneg {
        Sign::Nonnegative
    } else if all_nonpos {
        Sign::Nonpositive
    } else {
        Sign::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, nonneg_variable, variable};

    #[test]
    fn test_sign_basics() {
        assert!(Sign::Nonnegative.is_nonneg());
        assert!(!Sign::Nonnegative.is_nonpos());

        assert!(!Sign::Nonpositive.is_nonneg());
        assert!(Sign::Nonpositive.is_nonpos());

        assert!(Sign::Zero.is_nonneg());
        assert!(Sign::Zero.is_nonpos());
        assert!(Sign::Zero.is_zero());
    }

    #[test]
    fn test_negate_sign() {
        assert_eq!(Sign::Nonnegative.negate(), Sign::Nonpositive);
        assert_eq!(Sign::Nonpositive.negate(), Sign::Nonnegative);
        assert_eq!(Sign::Zero.negate(), Sign::Zero);
    }

    #[test]
    fn test_add_sign() {
        use Sign::*;
        assert_eq!(add_sign(Nonnegative, Nonnegative), Nonnegative);
        assert_eq!(add_sign(Nonpositive, Nonpositive), Nonpositive);
        assert_eq!(add_sign(Nonnegative, Nonpositive), Unknown);
        assert_eq!(add_sign(Zero, Nonnegative), Nonnegative);
    }

    #[test]
    fn test_mul_sign() {
        use Sign::*;
        assert_eq!(mul_sign(Nonnegative, Nonnegative), Nonnegative);
        assert_eq!(mul_sign(Nonpositive, Nonpositive), Nonnegative);
        assert_eq!(mul_sign(Nonnegative, Nonpositive), Nonpositive);
        assert_eq!(mul_sign(Zero, Unknown), Zero);
    }

    #[test]
    fn test_variable_sign() {
        let x = variable(5);
        assert_eq!(x.sign(), Sign::Unknown);

        let y = nonneg_variable(5);
        assert_eq!(y.sign(), Sign::Nonnegative);
    }

    #[test]
    fn test_constant_sign() {
        let c = constant(5.0);
        assert_eq!(c.sign(), Sign::Nonnegative);

        let c = constant(-5.0);
        assert_eq!(c.sign(), Sign::Nonpositive);

        let c = constant(0.0);
        assert_eq!(c.sign(), Sign::Zero);
    }

    #[test]
    fn test_norm_sign() {
        let x = variable(5);
        let n = Expr::Norm2(Arc::new(x));
        assert_eq!(n.sign(), Sign::Nonnegative);
    }
}
