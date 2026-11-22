/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Variable occurrence traversal shared by expression and problem queries.

use super::{Expr, VariableData};

impl Expr {
    /// Visit each variable occurrence in left-to-right expression order.
    ///
    /// Shared subexpressions and clones with the same ID are visited repeatedly
    /// so callers can observe all occurrence-specific metadata.
    pub(crate) fn visit_variables(&self, visit: &mut impl FnMut(&VariableData)) {
        match self {
            Expr::Variable(v) => visit(v),
            Expr::Constant(_) => {}

            Expr::Add(a, b)
            | Expr::Mul(a, b)
            | Expr::MatMul(a, b)
            | Expr::QuadForm(a, b)
            | Expr::QuadOverLin(a, b) => {
                a.visit_variables(visit);
                b.visit_variables(visit);
            }
            Expr::Neg(a)
            | Expr::Promote(a, _)
            | Expr::Sum(a, _)
            | Expr::Reshape(a, _)
            | Expr::Index(a, _)
            | Expr::Transpose(a)
            | Expr::Trace(a)
            | Expr::Norm1(a)
            | Expr::Norm2(a)
            | Expr::NormInf(a)
            | Expr::Abs(a)
            | Expr::Pos(a)
            | Expr::NegPart(a)
            | Expr::SumSquares(a)
            | Expr::Exp(a)
            | Expr::Log(a)
            | Expr::Entropy(a)
            | Expr::Power(a, _)
            | Expr::Cumsum(a, _)
            | Expr::Diag(a) => {
                a.visit_variables(visit);
            }
            Expr::VStack(exprs)
            | Expr::HStack(exprs)
            | Expr::Maximum(exprs)
            | Expr::Minimum(exprs) => {
                for e in exprs {
                    e.visit_variables(visit);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::expr::{VariableExt, constant, variable};

    #[test]
    fn test_shared_variable_occurrences_preserve_order_and_metadata() {
        let x = variable(());
        let y = variable(());
        let x_id = x.variable_id().unwrap();
        let y_id = y.variable_id().unwrap();
        let shared = Arc::new(&y + &x);
        let expr = Expr::Maximum(vec![
            shared.clone(),
            Arc::new(-x.clone().nonneg()),
            Arc::new(constant(0.0)),
            Arc::new(x.nonpos()),
            shared,
        ]);

        let mut occurrences = Vec::new();
        expr.visit_variables(&mut |variable| {
            occurrences.push((variable.id, variable.nonneg, variable.nonpos));
        });

        assert_eq!(
            occurrences,
            vec![
                (y_id, false, false),
                (x_id, false, false),
                (x_id, true, false),
                (x_id, false, true),
                (y_id, false, false),
                (x_id, false, false),
            ]
        );
        assert_eq!(expr.variables(), vec![x_id, y_id]);
    }
}
