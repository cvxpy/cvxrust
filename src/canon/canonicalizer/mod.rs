/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Expression canonicalization.
//!
//! Canonicalization transforms arbitrary DCP expressions into standard form:
//! - Affine expressions become LinExpr
//! - Quadratic objectives become QuadExpr (with P matrix for native QP)
//! - Nonlinear atoms are reformulated as affine + cone constraints

use std::collections::HashSet;

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::lin_expr::{LinExpr, QuadExpr};
use crate::expr::{Expr, ExprId, Shape, VariableBuilder};
use crate::sparse::{csc_repeat_rows, csc_to_dense, dense_to_csc};

mod affine;
mod nonlinear;
mod quadratic;
mod stacking;
mod structural;

#[cfg(test)]
mod tests;

/// A cone constraint in standard form: Ax + b in K.
#[derive(Debug, Clone)]
pub enum ConeConstraint {
    /// Zero cone: Ax + b = 0 (equality).
    Zero { a: LinExpr },
    /// Nonnegative cone: Ax + b >= 0.
    NonNeg { a: LinExpr },
    /// Second-order cone: ||x||_2 <= t.
    /// Represented as [t; x] in K_soc.
    SOC {
        /// The scalar t expression.
        t: LinExpr,
        /// The vector x expression.
        x: LinExpr,
    },
    /// Exponential cone: {(x, y, z) | y ≥ 0, y*exp(x/y) ≤ z} ∪ {(x,y,z) | x ≤ 0, y = 0, z ≥ 0}
    /// Variable order is (x, y, z).
    ExpCone {
        /// The x expression.
        x: LinExpr,
        /// The y expression.
        y: LinExpr,
        /// The z expression.
        z: LinExpr,
    },
    /// Power cone: {(x, y, z) : x^α * y^(1-α) ≥ |z|, (x,y) ≥ 0} with α ∈ (0,1)
    /// Variable order is (x, y, z).
    PowerCone {
        /// The x expression.
        x: LinExpr,
        /// The y expression.
        y: LinExpr,
        /// The z expression.
        z: LinExpr,
        /// The power α ∈ (0,1).
        alpha: f64,
    },
}

/// Result of canonicalizing an expression.
#[derive(Debug)]
pub struct CanonResult {
    /// The canonicalized expression (affine or quadratic).
    pub expr: CanonExpr,
    /// Additional cone constraints introduced during canonicalization.
    pub constraints: Vec<ConeConstraint>,
    /// Auxiliary variables introduced during canonicalization.
    pub aux_vars: Vec<(ExprId, Shape)>,
}

/// The type of canonicalized expression.
#[derive(Debug)]
pub enum CanonExpr {
    /// Linear expression.
    Linear(LinExpr),
    /// Quadratic expression (for objectives only).
    Quadratic(QuadExpr),
}

impl CanonExpr {
    /// Get as linear expression, panicking if quadratic.
    pub fn as_linear(&self) -> &LinExpr {
        match self {
            CanonExpr::Linear(l) => l,
            CanonExpr::Quadratic(_) => panic!("Expected linear expression, got quadratic"),
        }
    }

    /// Get as quadratic expression, converting linear if needed.
    pub fn into_quadratic(self) -> QuadExpr {
        match self {
            CanonExpr::Linear(l) => QuadExpr::from_linear(l),
            CanonExpr::Quadratic(q) => q,
        }
    }
}

/// Implicit variable bounds already emitted during canonicalization.
///
/// Create fresh state with [`Self::default`] for each independent problem.
/// Reuse it across expressions in the same problem, retaining the constraints
/// returned by every call to [`canonicalize`]. The state tracks emitted bounds
/// but does not store the constraints themselves.
#[derive(Default)]
pub struct VariableDomains {
    nonneg: HashSet<ExprId>,
    nonpos: HashSet<ExprId>,
}

/// Canonicalize an expression using explicit variable domain state.
///
/// This converts the expression tree into affine form plus cone constraints.
/// For objectives, quadratic expressions are preserved for native QP support.
/// Variable sign declarations are included as elementwise bounds. Declarations
/// on clones with the same variable ID combine, so opposite signs require zero.
/// Only variables referenced by the expression contribute bounds.
///
/// Reuse `domains` across expressions whose results form one problem. Each call
/// returns only newly emitted variable bounds, so combine the constraints from
/// all calls sharing that state. For an independent result, use fresh state.
///
/// ```
/// use cvxrust::canon::{VariableDomains, canonicalize};
/// use cvxrust::prelude::*;
///
/// let x = variable(3).nonneg();
/// let mut domains = VariableDomains::default();
/// let result = canonicalize(&x, false, &mut domains);
/// assert_eq!(result.constraints.len(), 1);
/// ```
pub fn canonicalize(
    expr: &Expr,
    for_objective: bool,
    domains: &mut VariableDomains,
) -> CanonResult {
    let mut ctx = CanonContext::new(domains);
    let canon_expr = ctx.canonicalize_expr(expr, for_objective);
    CanonResult {
        expr: canon_expr,
        constraints: ctx.constraints,
        aux_vars: ctx.aux_vars,
    }
}

/// Context for canonicalization, tracking auxiliary variables and constraints.
struct CanonContext<'a> {
    constraints: Vec<ConeConstraint>,
    aux_vars: Vec<(ExprId, Shape)>,
    domains: &'a mut VariableDomains,
}

impl<'a> CanonContext<'a> {
    fn new(domains: &'a mut VariableDomains) -> Self {
        CanonContext {
            constraints: Vec::new(),
            aux_vars: Vec::new(),
            domains,
        }
    }

    /// Create a new auxiliary variable.
    fn new_aux_var(&mut self, shape: Shape) -> (ExprId, LinExpr) {
        let var = VariableBuilder::new(shape.clone()).build();
        let var_id = var.variable_id().unwrap();
        self.aux_vars.push((var_id, shape.clone()));
        (var_id, LinExpr::variable(var_id, shape))
    }

    /// Create a new non-negative auxiliary variable.
    fn new_nonneg_aux_var(&mut self, shape: Shape) -> (ExprId, LinExpr) {
        let var = VariableBuilder::new(shape.clone()).nonneg().build();
        let var_id = var.variable_id().unwrap();
        self.aux_vars.push((var_id, shape.clone()));
        let lin_var = LinExpr::variable(var_id, shape);
        // Add t >= 0 constraint
        self.constraints
            .push(ConeConstraint::NonNeg { a: lin_var.clone() });
        (var_id, lin_var)
    }

    /// Canonicalize an expression.
    fn canonicalize_expr(&mut self, expr: &Expr, for_objective: bool) -> CanonExpr {
        match expr {
            // Leaves
            Expr::Variable(v) => {
                let lin = LinExpr::variable(v.id, v.shape.clone());
                if v.nonneg && self.domains.nonneg.insert(v.id) {
                    self.constraints
                        .push(ConeConstraint::NonNeg { a: lin.clone() });
                }
                if v.nonpos && self.domains.nonpos.insert(v.id) {
                    self.constraints
                        .push(ConeConstraint::NonNeg { a: lin.neg() });
                }
                CanonExpr::Linear(lin)
            }
            Expr::Constant(c) => CanonExpr::Linear(self.canonicalize_constant(&c.value)),

            // Affine operations
            Expr::Add(a, b) => {
                let ca = self.canonicalize_expr(a, for_objective);
                let cb = self.canonicalize_expr(b, for_objective);
                match (ca, cb) {
                    (CanonExpr::Linear(la), CanonExpr::Linear(lb)) => {
                        CanonExpr::Linear(la.add(&lb))
                    }
                    (CanonExpr::Quadratic(qa), CanonExpr::Linear(lb)) => {
                        let qb = QuadExpr::from_linear(lb);
                        CanonExpr::Quadratic(qa.add(&qb))
                    }
                    (CanonExpr::Linear(la), CanonExpr::Quadratic(qb)) => {
                        let qa = QuadExpr::from_linear(la);
                        CanonExpr::Quadratic(qa.add(&qb))
                    }
                    (CanonExpr::Quadratic(qa), CanonExpr::Quadratic(qb)) => {
                        CanonExpr::Quadratic(qa.add(&qb))
                    }
                }
            }
            Expr::Neg(a) => {
                let ca = self.canonicalize_expr(a, for_objective);
                match ca {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.neg()),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(-1.0)),
                }
            }
            Expr::Mul(a, b) => self.canonicalize_mul(a, b, for_objective),
            Expr::Promote(a, shape) => self.canonicalize_promote(a, shape),
            Expr::MatMul(a, b) => self.canonicalize_matmul(a, b),
            Expr::Sum(a, axis) => self.canonicalize_sum(a, *axis),
            Expr::Reshape(a, shape) => self.canonicalize_reshape(a, shape),
            Expr::Index(a, spec) => self.canonicalize_index(a, spec),
            Expr::VStack(exprs) => self.canonicalize_vstack(exprs),
            Expr::HStack(exprs) => self.canonicalize_hstack(exprs),
            Expr::Transpose(a) => self.canonicalize_transpose(a),
            Expr::Trace(a) => self.canonicalize_trace(a),

            // Nonlinear atoms - introduce auxiliary variables and cone constraints
            Expr::Norm1(x) => self.canonicalize_norm1(x),
            Expr::Norm2(x) => self.canonicalize_norm2(x),
            Expr::NormInf(x) => self.canonicalize_norm_inf(x),
            Expr::Abs(x) => self.canonicalize_abs(x),
            Expr::Pos(x) => self.canonicalize_pos(x),
            Expr::NegPart(x) => self.canonicalize_neg_part(x),
            Expr::Maximum(exprs) => self.canonicalize_maximum(exprs),
            Expr::Minimum(exprs) => self.canonicalize_minimum(exprs),
            Expr::QuadForm(x, p) => self.canonicalize_quad_form(x, p, for_objective),
            Expr::SumSquares(x) => self.canonicalize_sum_squares(x, for_objective),
            Expr::QuadOverLin(x, y) => self.canonicalize_quad_over_lin(x, y),

            // Exponential cone atoms
            Expr::Exp(x) => self.canonicalize_exp(x),
            Expr::Log(x) => self.canonicalize_log(x),
            Expr::Entropy(x) => self.canonicalize_entropy(x),
            // Power cone atoms
            Expr::Power(x, p) => self.canonicalize_power(x, *p),

            // Additional affine atoms
            Expr::Cumsum(x, axis) => self.canonicalize_cumsum(x, *axis),
            Expr::Diag(x) => self.canonicalize_diag(x),
        }
    }

    fn flatten_lin(&self, x: &LinExpr) -> LinExpr {
        let size = x.size();
        LinExpr {
            coeffs: x.coeffs.clone(),
            constant: x
                .constant
                .clone()
                .reshape_generic(nalgebra::Dyn(size), nalgebra::Dyn(1)),
            shape: Shape::vector(size),
        }
    }

    fn expand_scalar(&self, scalar: &LinExpr, size: usize) -> LinExpr {
        // Expand a scalar to a vector by repeating
        let ones = DMatrix::from_element(size, 1, 1.0);
        let new_const = &ones * &scalar.constant;

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &scalar.coeffs {
            // Repeat the scalar coefficient
            let expanded = repeat_rows_csc(coeff, size);
            new_coeffs.insert(*var_id, expanded);
        }

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::vector(size),
        }
    }

    /// Extract a single element from a linear expression
    fn extract_element(&self, expr: &LinExpr, idx: usize) -> LinExpr {
        let mut new_coeffs = std::collections::HashMap::new();

        for (var_id, coeff) in &expr.coeffs {
            // Create a sparse matrix that selects element idx
            let coeff_dense = csc_to_dense(coeff);
            let var_size = coeff_dense.ncols();
            let mut new_coeff = DMatrix::zeros(1, var_size);

            for j in 0..var_size {
                new_coeff[(0, j)] = coeff_dense[(idx, j)];
            }

            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
        }

        // Extract constant element
        let const_val = if expr.shape.is_vector() {
            expr.constant[(idx, 0)]
        } else {
            // For matrix, compute flat index
            let row = idx % expr.shape.rows();
            let col = idx / expr.shape.rows();
            expr.constant[(row, col)]
        };

        LinExpr {
            coeffs: new_coeffs,
            constant: DMatrix::from_element(1, 1, const_val),
            shape: Shape::scalar(),
        }
    }
}

fn repeat_rows_csc(m: &CscMatrix<f64>, times: usize) -> CscMatrix<f64> {
    csc_repeat_rows(m, times)
}
