/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Constraint types for optimization problems.
//!
//! Constraints map to cone constraints in the solver:
//! - Zero: `A x + b = 0` (zero cone / equality)
//! - NonNeg: `A x + b >= 0` (nonnegative orthant)
//! - SOC: `||x||_2 <= t` (second-order cone)

use std::sync::Arc;

use crate::expr::Expr;

/// A constraint in an optimization problem.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Equality constraint: `expr == 0`.
    /// Maps to the zero cone.
    Zero(Arc<Expr>),

    /// Inequality constraint: `expr >= 0`.
    /// Maps to the nonnegative orthant cone.
    NonNeg(Arc<Expr>),

    /// Second-order cone constraint: `||x||_2 <= t`.
    /// The `t` argument must be scalar; `x` can be a vector.
    SOC {
        /// The scalar upper bound.
        t: Arc<Expr>,
        /// The vector argument.
        x: Arc<Expr>,
    },
}

impl Constraint {
    /// Create an equality constraint: `lhs == rhs` (with broadcasting).
    pub fn eq(lhs: Expr, rhs: Expr) -> Self {
        let (lhs, rhs) = Self::broadcast_if_needed(lhs, rhs);
        Constraint::Zero(Arc::new(Expr::Add(
            Arc::new(lhs),
            Arc::new(Expr::Neg(Arc::new(rhs))),
        )))
    }

    /// Create an inequality constraint: `lhs <= rhs` (with broadcasting).
    pub fn leq(lhs: Expr, rhs: Expr) -> Self {
        let (lhs, rhs) = Self::broadcast_if_needed(lhs, rhs);
        // lhs <= rhs  <=>  rhs - lhs >= 0
        Constraint::NonNeg(Arc::new(Expr::Add(
            Arc::new(rhs),
            Arc::new(Expr::Neg(Arc::new(lhs))),
        )))
    }

    /// Create an inequality constraint: `lhs >= rhs` (with broadcasting).
    pub fn geq(lhs: Expr, rhs: Expr) -> Self {
        let (lhs, rhs) = Self::broadcast_if_needed(lhs, rhs);
        // lhs >= rhs  <=>  lhs - rhs >= 0
        Constraint::NonNeg(Arc::new(Expr::Add(
            Arc::new(lhs),
            Arc::new(Expr::Neg(Arc::new(rhs))),
        )))
    }

    /// Broadcast scalars to match shapes if needed.
    fn broadcast_if_needed(lhs: Expr, rhs: Expr) -> (Expr, Expr) {
        crate::atoms::broadcast::broadcast_exprs(lhs, rhs)
    }

    /// Create a SOC constraint: `||x||_2 <= t`.
    pub fn soc(t: Expr, x: Expr) -> Self {
        Constraint::SOC {
            t: Arc::new(t),
            x: Arc::new(x),
        }
    }

    /// Check if this constraint is DCP-compliant.
    ///
    /// DCP rules for constraints:
    /// - Zero: expression must be affine (equality of affine expressions)
    /// - NonNeg: expression must be concave (`concave >= 0`)
    /// - SOC: both t and x must be affine
    pub fn is_dcp(&self) -> bool {
        match self {
            Constraint::Zero(expr) => expr.is_affine(),
            Constraint::NonNeg(expr) => expr.is_concave(),
            Constraint::SOC { t, x } => t.is_affine() && x.is_affine(),
        }
    }

    /// Get all expressions in this constraint.
    pub fn expressions(&self) -> Vec<&Expr> {
        match self {
            Constraint::Zero(e) => vec![e.as_ref()],
            Constraint::NonNeg(e) => vec![e.as_ref()],
            Constraint::SOC { t, x } => vec![t.as_ref(), x.as_ref()],
        }
    }

    /// Get all variable IDs in this constraint.
    pub fn variables(&self) -> Vec<crate::expr::ExprId> {
        let mut vars = Vec::new();
        for expr in self.expressions() {
            vars.extend(expr.variables());
        }
        vars.sort_by_key(|id| id.raw());
        vars.dedup();
        vars
    }
}

/// Extension trait for creating constraints from expressions.
pub trait ConstraintExt {
    /// Create equality constraint: `self == rhs`.
    ///
    /// The rhs can be any type that converts to Expr (f64, Vec, etc.).
    fn eq<E: Into<Expr>>(&self, rhs: E) -> Constraint;

    /// Create inequality constraint: `self <= rhs`.
    ///
    /// The rhs can be any type that converts to Expr (f64, Vec, etc.).
    fn le<E: Into<Expr>>(&self, rhs: E) -> Constraint;

    /// Create inequality constraint: `self >= rhs`.
    ///
    /// The rhs can be any type that converts to Expr (f64, Vec, etc.).
    fn ge<E: Into<Expr>>(&self, rhs: E) -> Constraint;
}

impl ConstraintExt for Expr {
    fn eq<E: Into<Expr>>(&self, rhs: E) -> Constraint {
        Constraint::eq(self.clone(), rhs.into())
    }

    fn le<E: Into<Expr>>(&self, rhs: E) -> Constraint {
        Constraint::leq(self.clone(), rhs.into())
    }

    fn ge<E: Into<Expr>>(&self, rhs: E) -> Constraint {
        Constraint::geq(self.clone(), rhs.into())
    }
}

/// Macro for creating constraints with operator syntax.
///
/// # Examples
///
/// ```
/// use cvxrust::prelude::*;
///
/// let x = variable(5);
/// let c1 = constraint!(x >= 1.0);
/// let c2 = constraint!(x <= 10.0);
/// let c3 = constraint!(x == 5.0);
/// ```
#[macro_export]
macro_rules! constraint {
    ($lhs:tt >= $rhs:tt) => {
        $crate::constraints::ConstraintExt::ge(&$lhs, $rhs)
    };
    ($lhs:tt <= $rhs:tt) => {
        $crate::constraints::ConstraintExt::le(&$lhs, $rhs)
    };
    ($lhs:tt == $rhs:tt) => {
        $crate::constraints::ConstraintExt::eq(&$lhs, $rhs)
    };
    (($($lhs:tt)+) >= $rhs:tt) => {
        $crate::constraints::ConstraintExt::ge(&($($lhs)+), $rhs)
    };
    (($($lhs:tt)+) <= $rhs:tt) => {
        $crate::constraints::ConstraintExt::le(&($($lhs)+), $rhs)
    };
    (($($lhs:tt)+) == $rhs:tt) => {
        $crate::constraints::ConstraintExt::eq(&($($lhs)+), $rhs)
    };
    ($lhs:tt >= ($($rhs:tt)+)) => {
        $crate::constraints::ConstraintExt::ge(&$lhs, ($($rhs)+))
    };
    ($lhs:tt <= ($($rhs:tt)+)) => {
        $crate::constraints::ConstraintExt::le(&$lhs, ($($rhs)+))
    };
    ($lhs:tt == ($($rhs:tt)+)) => {
        $crate::constraints::ConstraintExt::eq(&$lhs, ($($rhs)+))
    };
    (($($lhs:tt)+) >= ($($rhs:tt)+)) => {
        $crate::constraints::ConstraintExt::ge(&($($lhs)+), ($($rhs)+))
    };
    (($($lhs:tt)+) <= ($($rhs:tt)+)) => {
        $crate::constraints::ConstraintExt::le(&($($lhs)+), ($($rhs)+))
    };
    (($($lhs:tt)+) == ($($rhs:tt)+)) => {
        $crate::constraints::ConstraintExt::eq(&($($lhs)+), ($($rhs)+))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, variable};

    #[test]
    fn test_equality_constraint() {
        let x = variable(5);
        let c = constant(1.0);
        let constr = Constraint::eq(x, c);

        assert!(constr.is_dcp());
        if let Constraint::Zero(_) = constr {
            // OK
        } else {
            panic!("Expected Zero constraint");
        }
    }

    #[test]
    fn test_inequality_constraint() {
        let x = variable(5);
        let c = constant(0.0);
        let constr = Constraint::geq(x, c);

        assert!(constr.is_dcp());
        if let Constraint::NonNeg(_) = constr {
            // OK
        } else {
            panic!("Expected NonNeg constraint");
        }
    }

    #[test]
    fn test_soc_constraint() {
        let t = variable(());
        let x = variable(5);
        let constr = Constraint::soc(t, x);

        assert!(constr.is_dcp());
    }

    #[test]
    fn test_non_dcp_constraint() {
        let x = variable(5);
        // norm(x) >= 1 is NOT DCP (convex >= constant)
        let norm_x = Expr::Norm2(Arc::new(x));
        let c = constant(1.0);
        let constr = Constraint::geq(norm_x, c);

        // norm_x is convex, so norm_x - 1 is convex, not concave
        assert!(!constr.is_dcp());
    }

    #[test]
    fn test_constraint_ext() {
        let x = variable(5);

        let eq_constr = x.eq(1.0);
        assert!(eq_constr.is_dcp());

        let le_constr = x.le(2.0);
        assert!(le_constr.is_dcp());

        let ge_constr = x.ge(0.0);
        assert!(ge_constr.is_dcp());
    }

    #[test]
    fn test_broadcasting_scalar_to_vector() {
        // Test that x >= 0.0 broadcasts 0.0 to match x's shape
        let x = variable(5);
        let constr = x.ge(0.0);
        assert!(constr.is_dcp());

        // Verify it's a NonNeg constraint
        if let Constraint::NonNeg(_) = constr {
            // OK
        } else {
            panic!("Expected NonNeg constraint");
        }
    }

    #[test]
    fn test_broadcasting_with_macro() {
        let x = variable(3);

        // Test >= with broadcasting
        let c1 = constraint!(x >= 0.0);
        assert!(c1.is_dcp());

        // Test <= with broadcasting
        let c2 = constraint!(x <= 10.0);
        assert!(c2.is_dcp());

        // Test == with broadcasting
        let c3 = constraint!(x == 5.0);
        assert!(c3.is_dcp());
    }

    #[test]
    fn test_no_broadcasting_when_shapes_match() {
        use crate::expr::zeros;

        // When shapes already match, no broadcasting needed
        let x = variable(4);
        let z = zeros(4);
        let constr = x.ge(z);
        assert!(constr.is_dcp());
    }

    #[test]
    fn test_new_short_methods() {
        let x = variable(5);

        // Test new .ge(), .le(), .eq() methods with scalar auto-conversion
        let c1 = x.ge(1.0);
        assert!(c1.is_dcp());

        let x2 = variable(5);
        let c2 = x2.le(10.0);
        assert!(c2.is_dcp());

        let x3 = variable(5);
        let c3 = x3.eq(5.0);
        assert!(c3.is_dcp());
    }
}
