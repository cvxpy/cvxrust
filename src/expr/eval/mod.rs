/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Expression evaluation given a solution.
//!
//! This module provides the `Evaluable` trait and `Expr::eval()` method,
//! allowing users to compute the value of any expression (not just variables)
//! after solving a problem.
//!
//! # Example
//!
//! ```
//! use cvxrust::prelude::*;
//!
//! let x = variable(5);
//! let obj = norm2(&x);
//!
//! let solution = Problem::minimize(obj.clone())
//!     .subject_to([x.ge(1.0)])
//!     .solve()
//!     .unwrap();
//!
//! // Evaluate any expression, not just variables
//! let norm_val = obj.value(&solution);
//! let x_vals = x.eval(&solution).unwrap();
//! ```

use super::expression::{Array, Expr, ExprId};

mod affine;
mod broadcast;
mod nonlinear;
mod structural;

#[cfg(test)]
mod tests;

use affine::{eval_add, eval_cumsum, eval_matmul, eval_mul, eval_neg, eval_sum, eval_trace};
use nonlinear::{
    eval_elementwise, eval_maximum, eval_minimum, eval_norm1, eval_norm2, eval_norminf,
    eval_quad_form, eval_quad_over_lin,
};
use structural::{
    eval_diag, eval_hstack, eval_index, eval_promote, eval_reshape, eval_transpose, eval_vstack,
};

/// Trait for types that provide variable values (used for expression evaluation).
///
/// Implement this trait to allow `Expr::eval()` to look up variable values.
/// The `Solution` type in `cvxrust::solver` implements this trait.
pub trait Evaluable {
    /// Get the value of a variable by its ID.
    fn get_variable_value(&self, id: ExprId) -> Option<&Array>;
}

impl Expr {
    /// Evaluate this expression given variable values from a solution.
    ///
    /// Evaluate any expression after solving a problem, as with CVXPY's
    /// `expr.value` attribute.
    ///
    /// # Example
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(3);
    /// let expr = norm2(&x);
    ///
    /// let solution = Problem::minimize(expr.clone())
    ///     .subject_to([x.ge(1.0)])
    ///     .solve()
    ///     .unwrap();
    ///
    /// let arr = expr.eval(&solution).unwrap();
    /// ```
    /// Evaluate this expression, returning an `Array`.
    ///
    /// Panics if a variable in the expression is not present in `ctx`.
    /// Use `eval()` for explicit error handling.
    ///
    /// The returned `Array` supports `[(row, col)]` indexing, matching the
    /// ergonomics of the old `&solution[&x]` API:
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(2);
    /// let solution = Problem::minimize(sum(&x))
    ///     .subject_to([x.ge(1.0)])
    ///     .solve()
    ///     .unwrap();
    ///
    /// let vals = x.value(&solution);
    /// println!("{}", vals[(0, 0)]);
    /// ```
    pub fn value<E: Evaluable>(&self, ctx: &E) -> Array {
        self.eval(ctx).expect("failed to evaluate expression")
    }

    /// Evaluate this expression, returning `Result<Array>`.
    pub fn eval<E: Evaluable>(&self, ctx: &E) -> crate::Result<Array> {
        match self {
            Expr::Variable(v) => ctx
                .get_variable_value(v.id)
                .cloned()
                .ok_or_else(|| crate::CvxError::InvalidProblem("Variable not in solution".into())),
            Expr::Constant(c) => Ok(c.value.clone()),

            Expr::Add(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                eval_add(av, &a.shape(), bv, &b.shape())
            }
            Expr::Neg(a) => Ok(eval_neg(a.eval(ctx)?)),
            Expr::Mul(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                eval_mul(av, &a.shape(), bv, &b.shape())
            }
            Expr::Promote(a, shape) => eval_promote(a.eval(ctx)?, shape),
            Expr::Sum(a, axis) => eval_sum(a.eval(ctx)?, *axis),
            Expr::Reshape(a, shape) => eval_reshape(a.eval(ctx)?, shape),
            Expr::Index(a, spec) => eval_index(a.eval(ctx)?, spec),
            Expr::VStack(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs.iter().map(|e| e.eval(ctx)).collect();
                eval_vstack(arrs?)
            }
            Expr::HStack(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs.iter().map(|e| e.eval(ctx)).collect();
                eval_hstack(arrs?)
            }
            Expr::Transpose(a) => Ok(eval_transpose(a.eval(ctx)?)),
            Expr::Trace(a) => Ok(eval_trace(a.eval(ctx)?)),
            Expr::MatMul(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                eval_matmul(av, bv)
            }
            Expr::Norm1(a) => Ok(Array::Scalar(eval_norm1(&a.eval(ctx)?))),
            Expr::Norm2(a) => Ok(Array::Scalar(eval_norm2(&a.eval(ctx)?))),
            Expr::NormInf(a) => Ok(Array::Scalar(eval_norminf(&a.eval(ctx)?))),
            Expr::Abs(a) => Ok(eval_elementwise(a.eval(ctx)?, f64::abs)),
            Expr::Pos(a) => Ok(eval_elementwise(a.eval(ctx)?, |x| x.max(0.0))),
            Expr::NegPart(a) => Ok(eval_elementwise(a.eval(ctx)?, |x| (-x).max(0.0))),
            Expr::Maximum(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs
                    .iter()
                    .map(|e| e.eval(ctx).map(|array| (array, e.shape())))
                    .collect();
                eval_maximum(arrs?)
            }
            Expr::Minimum(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs
                    .iter()
                    .map(|e| e.eval(ctx).map(|array| (array, e.shape())))
                    .collect();
                eval_minimum(arrs?)
            }
            Expr::QuadForm(x, p) => {
                let xv = x.eval(ctx)?;
                let pv = p.eval(ctx)?;
                eval_quad_form(xv, pv)
            }
            Expr::SumSquares(a) => {
                let av = a.eval(ctx)?;
                Ok(Array::Scalar(eval_norm2(&av).powi(2)))
            }
            Expr::QuadOverLin(x, y) => {
                let xv = x.eval(ctx)?;
                let yv = y.eval(ctx)?;
                eval_quad_over_lin(xv, yv)
            }
            Expr::Exp(a) => Ok(eval_elementwise(a.eval(ctx)?, f64::exp)),
            Expr::Log(a) => Ok(eval_elementwise(a.eval(ctx)?, f64::ln)),
            Expr::Entropy(a) => Ok(eval_elementwise(a.eval(ctx)?, |x| {
                if x <= 0.0 { 0.0 } else { -x * x.ln() }
            })),
            Expr::Power(a, p) => {
                let p = *p;
                Ok(eval_elementwise(a.eval(ctx)?, move |x| x.powf(p)))
            }
            Expr::Cumsum(a, axis) => eval_cumsum(a.eval(ctx)?, *axis),
            Expr::Diag(a) => Ok(eval_diag(a.eval(ctx)?, &a.shape())),
        }
    }
}
