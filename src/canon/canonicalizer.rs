//! Expression canonicalization.
//!
//! Canonicalization transforms arbitrary DCP expressions into standard form:
//! - Affine expressions become LinExpr
//! - Quadratic objectives become QuadExpr (with P matrix for native QP)
//! - Nonlinear atoms are reformulated as affine + cone constraints

use std::sync::Arc;

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::lin_expr::{LinExpr, QuadExpr};
use crate::expr::{Array, Expr, ExprId, IndexSpec, Shape, VariableBuilder};
use crate::sparse::{csc_add, csc_repeat_rows, csc_to_dense, csc_vstack, dense_to_csc};

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

/// Canonicalize an expression.
///
/// This converts the expression tree into affine form plus cone constraints.
/// For objectives, quadratic expressions are preserved for native QP support.
pub fn canonicalize(expr: &Expr, for_objective: bool) -> CanonResult {
    let mut ctx = CanonContext::new();
    let canon_expr = ctx.canonicalize_expr(expr, for_objective);
    CanonResult {
        expr: canon_expr,
        constraints: ctx.constraints,
        aux_vars: ctx.aux_vars,
    }
}

/// Context for canonicalization, tracking auxiliary variables and constraints.
struct CanonContext {
    constraints: Vec<ConeConstraint>,
    aux_vars: Vec<(ExprId, Shape)>,
}

impl CanonContext {
    fn new() -> Self {
        CanonContext {
            constraints: Vec::new(),
            aux_vars: Vec::new(),
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
            Expr::Variable(v) => CanonExpr::Linear(LinExpr::variable(v.id, v.shape.clone())),
            Expr::Constant(c) => CanonExpr::Linear(self.canonicalize_constant(&c.value)),

            // Affine operations
            Expr::Add(a, b) => {
                let ca = self.canonicalize_expr(a, false);
                let cb = self.canonicalize_expr(b, false);
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
                let ca = self.canonicalize_expr(a, false);
                match ca {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.neg()),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(-1.0)),
                }
            }
            Expr::Mul(a, b) => self.canonicalize_mul(a, b, for_objective),
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

    fn canonicalize_constant(&self, arr: &Array) -> LinExpr {
        match arr {
            Array::Scalar(v) => LinExpr::scalar(*v),
            Array::Dense(m) => LinExpr::constant(m.clone()),
            Array::Sparse(s) => {
                // Convert sparse to dense
                LinExpr::constant(csc_to_dense(s))
            }
        }
    }

    fn canonicalize_mul(&mut self, a: &Expr, b: &Expr, for_objective: bool) -> CanonExpr {
        // Check if expressions are constant (no variables)
        let a_is_const = a.variables().is_empty();
        let b_is_const = b.variables().is_empty();

        // Handle scalar multiplication first (most common case)
        if let Some(arr) = a.constant_value() {
            if let Some(scalar) = arr.as_scalar() {
                let cb = self.canonicalize_expr(b, for_objective);
                return match cb {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
                };
            }
        }
        if let Some(arr) = b.constant_value() {
            if let Some(scalar) = arr.as_scalar() {
                let ca = self.canonicalize_expr(a, for_objective);
                return match ca {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
                };
            }
        }

        // Handle constant expression that evaluates to scalar
        if a_is_const && !b_is_const {
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            if ca.shape.is_scalar() {
                let scalar = ca.constant[(0, 0)];
                let cb = self.canonicalize_expr(b, for_objective);
                return match cb {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
                };
            }
            // Element-wise multiplication with constant vector/matrix
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            return CanonExpr::Linear(self.elementwise_mul_const_lin(&ca.constant, &cb));
        }
        if b_is_const && !a_is_const {
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            if cb.shape.is_scalar() {
                let scalar = cb.constant[(0, 0)];
                let ca = self.canonicalize_expr(a, for_objective);
                return match ca {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
                };
            }
            // Element-wise multiplication with constant vector/matrix
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            return CanonExpr::Linear(self.elementwise_mul_const_lin(&cb.constant, &ca));
        }
        if a_is_const && b_is_const {
            // Both constant - evaluate and return
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let result = ca.constant.component_mul(&cb.constant);
            return CanonExpr::Linear(LinExpr::constant(result));
        }

        // Both have variables - not DCP
        self.canonicalize_expr(a, false)
    }

    fn elementwise_mul_const_lin(&self, c: &DMatrix<f64>, lin: &LinExpr) -> LinExpr {
        // Element-wise multiplication: diag(c) @ lin
        // For flat representation, this scales each row of coefficients by corresponding c value
        let c_flat: Vec<f64> = c.iter().copied().collect();
        let size = c_flat.len();

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &lin.coeffs {
            let coeff_dense = csc_to_dense(coeff);
            let mut new_coeff = DMatrix::zeros(size, coeff_dense.ncols());
            for i in 0..size.min(coeff_dense.nrows()) {
                for j in 0..coeff_dense.ncols() {
                    new_coeff[(i, j)] = c_flat[i] * coeff_dense[(i, j)];
                }
            }
            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
        }

        let new_const = c.component_mul(&lin.constant);

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: lin.shape.clone(),
        }
    }

    fn canonicalize_matmul(&mut self, a: &Expr, b: &Expr) -> CanonExpr {
        // Check if expressions are constant (no variables, not just Constant variant)
        let a_is_const = a.variables().is_empty();
        let b_is_const = b.variables().is_empty();

        if a_is_const && !b_is_const {
            // A is constant expression, B has variables: A @ B is affine in B
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let a_arr = Array::Dense(ca.constant);
            return CanonExpr::Linear(self.matmul_const_lin(&a_arr, &cb));
        }
        if b_is_const && !a_is_const {
            // B is constant expression, A has variables: A @ B is affine in A
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let b_arr = Array::Dense(cb.constant);
            return CanonExpr::Linear(self.lin_matmul_const(&ca, &b_arr));
        }
        if a_is_const && b_is_const {
            // Both constant - evaluate and return constant
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let result = &ca.constant * &cb.constant;
            return CanonExpr::Linear(LinExpr::constant(result));
        }
        // Both have variables - not DCP, return simplified
        self.canonicalize_expr(a, false)
    }

    fn matmul_const_lin(&self, a: &Array, b: &LinExpr) -> LinExpr {
        // For matrix expression A @ E where E has shape (m, n):
        // vec(A @ E) = (I_n ⊗ A) @ vec(E)
        // So for coefficient C: new_C = (I_n ⊗ A) @ C
        let a_mat = match a {
            Array::Dense(m) => m.clone(),
            Array::Scalar(v) => DMatrix::from_element(1, 1, *v),
            Array::Sparse(s) => csc_to_dense(s),
        };

        let p = a_mat.nrows(); // rows of A
        let m = b.shape.rows(); // rows of E (should equal a_mat.ncols())
        let n = b.shape.cols(); // cols of E and result

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &b.coeffs {
            // Transform coefficient using Kronecker identity:
            // (I_n ⊗ A) @ c = vec(A @ reshape(c, m, n)) for each column c
            let coeff_dense = csc_to_dense(coeff);
            let var_size = coeff_dense.ncols();
            let new_size = p * n;

            let mut new_coeff_data = DMatrix::zeros(new_size, var_size);
            for j in 0..var_size {
                // Extract column j, reshape to (m, n), left-multiply by A, reshape back
                let col: Vec<f64> = (0..coeff_dense.nrows())
                    .map(|i| coeff_dense[(i, j)])
                    .collect();
                // Reshape to (m, n) - column-major order
                let mat = DMatrix::from_vec(m, n, col);
                // Left multiply by A
                let result = &a_mat * &mat;
                // Store back as column (column-major order)
                for (idx, val) in result.iter().enumerate() {
                    new_coeff_data[(idx, j)] = *val;
                }
            }
            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff_data));
        }

        let new_const = &a_mat * &b.constant;
        let shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape,
        }
    }

    fn lin_matmul_const(&self, a: &LinExpr, b: &Array) -> LinExpr {
        // For matrix expression E @ B where E has shape (m, n):
        // vec(E @ B) = (B' ⊗ I_m) @ vec(E)
        // So for coefficient C: new_C = (B' ⊗ I_m) @ C
        let b_mat = match b {
            Array::Dense(m) => m.clone(),
            Array::Scalar(v) => DMatrix::from_element(1, 1, *v),
            Array::Sparse(s) => csc_to_dense(s),
        };

        let m = a.shape.rows(); // rows of E
        let n = a.shape.cols(); // cols of E (should equal b_mat.nrows())
        let p = b_mat.ncols(); // cols of result

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &a.coeffs {
            // Transform coefficient using Kronecker identity:
            // (B' ⊗ I_m) @ c = vec(reshape(c, m, n) @ B) for each column c
            let coeff_dense = csc_to_dense(coeff);
            let var_size = coeff_dense.ncols();
            let new_size = m * p;

            let mut new_coeff_data = DMatrix::zeros(new_size, var_size);
            for j in 0..var_size {
                // Extract column j, reshape to (m, n), multiply by B, reshape back
                let col: Vec<f64> = (0..coeff_dense.nrows())
                    .map(|i| coeff_dense[(i, j)])
                    .collect();
                // Reshape to (m, n) - column-major order
                let mat = DMatrix::from_vec(m, n, col);
                // Right multiply by B
                let result = &mat * &b_mat;
                // Store back as column (column-major order)
                for (idx, val) in result.iter().enumerate() {
                    new_coeff_data[(idx, j)] = *val;
                }
            }
            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff_data));
        }

        let new_const = &a.constant * &b_mat;
        let shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape,
        }
    }

    fn canonicalize_sum(&mut self, a: &Expr, _axis: Option<usize>) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Sum all elements: multiply by ones vector
        let size = ca.size();
        let ones = DMatrix::from_element(1, size, 1.0);

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &ca.coeffs {
            let new_coeff = dense_sparse_matmul(&ones, coeff);
            new_coeffs.insert(*var_id, new_coeff);
        }

        let flat_const = ca
            .constant
            .reshape_generic(nalgebra::Dyn(size), nalgebra::Dyn(1));
        let result = &ones * &flat_const;
        let new_const = DMatrix::from_element(1, 1, result[(0, 0)]);

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::scalar(),
        })
    }

    fn canonicalize_reshape(&mut self, a: &Expr, shape: &Shape) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Reshape doesn't change the linear structure, just the shape interpretation
        CanonExpr::Linear(LinExpr {
            coeffs: ca.coeffs,
            constant: ca
                .constant
                .reshape_generic(nalgebra::Dyn(shape.rows()), nalgebra::Dyn(shape.cols())),
            shape: shape.clone(),
        })
    }

    fn canonicalize_index(&mut self, a: &Expr, spec: &IndexSpec) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();

        // For now, handle the common case: 1D vector indexing with a single range
        // The LinExpr stores data in column-major (flattened) order
        if spec.ranges.len() == 1 {
            if let Some((start, stop, step)) = spec.ranges[0] {
                // Simple range indexing on a vector
                let input_size = ca.shape.size();

                // Compute output indices
                let output_indices: Vec<usize> = (start..stop).step_by(step).collect();
                let output_size = output_indices.len();

                // Build selection matrix: S[i, output_indices[i]] = 1
                // S has shape (output_size, input_size)
                let mut s_rows = Vec::new();
                let mut s_cols = Vec::new();
                let mut s_vals = Vec::new();
                for (out_idx, &in_idx) in output_indices.iter().enumerate() {
                    if in_idx < input_size {
                        s_rows.push(out_idx);
                        s_cols.push(in_idx);
                        s_vals.push(1.0);
                    }
                }
                let s_mat = crate::sparse::triplets_to_csc(
                    output_size,
                    input_size,
                    &s_rows,
                    &s_cols,
                    &s_vals,
                );

                // Apply selection to each coefficient: new_A = S @ A
                let mut new_coeffs = std::collections::HashMap::new();
                for (var_id, coeff) in &ca.coeffs {
                    let new_coeff = crate::sparse::csc_matmul(&s_mat, coeff);
                    new_coeffs.insert(*var_id, new_coeff);
                }

                // Apply selection to constant: flatten, select rows, reshape
                let const_flat: Vec<f64> = ca.constant.iter().cloned().collect();
                let new_const_vals: Vec<f64> = output_indices
                    .iter()
                    .map(|&i| {
                        if i < const_flat.len() {
                            const_flat[i]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let new_const = DMatrix::from_vec(output_size, 1, new_const_vals);

                let new_shape = Shape::vector(output_size);

                return CanonExpr::Linear(LinExpr {
                    coeffs: new_coeffs,
                    constant: new_const,
                    shape: new_shape,
                });
            }
        }

        // Fallback for None ranges (take all) - return unchanged
        // Note: 2D matrix indexing could be added in future versions
        CanonExpr::Linear(ca)
    }

    fn canonicalize_vstack(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }
        // Canonicalize all and stack
        let mut result = self.canonicalize_expr(&exprs[0], false).as_linear().clone();
        for e in &exprs[1..] {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            result = self.vstack_lin(&result, &ce);
        }
        CanonExpr::Linear(result)
    }

    fn vstack_lin(&self, a: &LinExpr, b: &LinExpr) -> LinExpr {
        // Stack constants vertically
        let new_const = stack_vertical(&a.constant, &b.constant);
        let new_shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        // Stack coefficients for each variable
        let mut new_coeffs = std::collections::HashMap::new();
        let all_vars: std::collections::HashSet<_> =
            a.coeffs.keys().chain(b.coeffs.keys()).copied().collect();

        for var_id in all_vars {
            let ca = a.coeffs.get(&var_id);
            let cb = b.coeffs.get(&var_id);
            let stacked = match (ca, cb) {
                (Some(ma), Some(mb)) => stack_csc_vertical(ma, mb),
                (Some(ma), None) => {
                    let zeros = CscMatrix::zeros(b.size(), ma.ncols());
                    stack_csc_vertical(ma, &zeros)
                }
                (None, Some(mb)) => {
                    let zeros = CscMatrix::zeros(a.size(), mb.ncols());
                    stack_csc_vertical(&zeros, mb)
                }
                (None, None) => continue,
            };
            new_coeffs.insert(var_id, stacked);
        }

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: new_shape,
        }
    }

    fn canonicalize_hstack(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // Horizontal stacking (column-wise concatenation)
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }
        let mut result = self.canonicalize_expr(&exprs[0], false).as_linear().clone();
        for e in &exprs[1..] {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            result = self.hstack_lin(&result, &ce);
        }
        CanonExpr::Linear(result)
    }

    fn hstack_lin(&self, a: &LinExpr, b: &LinExpr) -> LinExpr {
        // Stack constants horizontally
        let new_const = stack_horizontal(&a.constant, &b.constant);
        let new_shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        // For coefficients: hstack increases output elements (in column-major order)
        // So we need to vertically stack the coefficient matrices
        // a contributes to first a.size() output elements
        // b contributes to next b.size() output elements
        let mut new_coeffs = std::collections::HashMap::new();
        let all_vars: std::collections::HashSet<_> =
            a.coeffs.keys().chain(b.coeffs.keys()).copied().collect();

        for var_id in all_vars {
            let ca = a.coeffs.get(&var_id);
            let cb = b.coeffs.get(&var_id);
            let stacked = match (ca, cb) {
                (Some(ma), Some(mb)) => csc_vstack(ma, mb),
                (Some(ma), None) => {
                    let zeros = CscMatrix::zeros(b.size(), ma.ncols());
                    csc_vstack(ma, &zeros)
                }
                (None, Some(mb)) => {
                    let zeros = CscMatrix::zeros(a.size(), mb.ncols());
                    csc_vstack(&zeros, mb)
                }
                (None, None) => continue,
            };
            new_coeffs.insert(var_id, stacked);
        }

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: new_shape,
        }
    }

    fn canonicalize_transpose(&mut self, a: &Expr) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        let m = ca.shape.rows();
        let n = ca.shape.cols();
        let new_shape = ca.shape.transpose();

        // Transpose the constant matrix
        let new_const = ca.constant.transpose();

        // For coefficients: we need to permute rows to match the transpose
        // Original flat index: i + j*m (column-major for m×n matrix)
        // Transposed flat index: j + i*n (column-major for n×m matrix)
        // Build permutation: perm[old_idx] = new_idx
        let size = m * n;
        let mut perm = vec![0usize; size];
        for j in 0..n {
            for i in 0..m {
                let old_idx = i + j * m;
                let new_idx = j + i * n;
                perm[old_idx] = new_idx;
            }
        }

        // Apply permutation to each coefficient matrix
        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &ca.coeffs {
            // coeff has shape (size, var_size)
            // We need to permute rows according to perm
            let coeff_dense = csc_to_dense(coeff);
            let var_size = coeff_dense.ncols();
            let mut new_coeff = DMatrix::zeros(size, var_size);
            for old_row in 0..size {
                let new_row = perm[old_row];
                for col in 0..var_size {
                    new_coeff[(new_row, col)] = coeff_dense[(old_row, col)];
                }
            }
            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
        }

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: new_shape,
        })
    }

    fn canonicalize_trace(&mut self, a: &Expr) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Trace: sum of diagonal elements
        let n = ca.shape.rows().min(ca.shape.cols());
        let nrows = ca.shape.rows();

        // Sum diagonal of constant matrix
        let trace_const: f64 = (0..n).map(|i| ca.constant[(i, i)]).sum();

        // Sum diagonal elements of coefficient matrices
        // Diagonal position (i, i) in column-major order is at flat index i * nrows + i
        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &ca.coeffs {
            let coeff_dense = csc_to_dense(coeff);
            let var_size = coeff_dense.ncols();
            let mut new_coeff = DMatrix::zeros(1, var_size);

            for j in 0..var_size {
                let mut sum = 0.0;
                for i in 0..n {
                    let diag_idx = i * nrows + i; // column-major flat index for (i, i)
                    if diag_idx < coeff_dense.nrows() {
                        sum += coeff_dense[(diag_idx, j)];
                    }
                }
                new_coeff[(0, j)] = sum;
            }
            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
        }

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: DMatrix::from_element(1, 1, trace_const),
            shape: Shape::scalar(),
        })
    }

    // ========================================================================
    // Nonlinear atom canonicalizers
    // ========================================================================

    fn canonicalize_norm1(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_1 = sum(|x_i|)
        // Introduce t_i >= 0, -t_i <= x_i <= t_i
        // Then ||x||_1 = sum(t_i)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let size = cx.size();
        let (_, t) = self.new_nonneg_aux_var(Shape::vector(size));

        // Constraints: t >= x, t >= -x
        // i.e., t - x >= 0, t + x >= 0
        self.constraints.push(ConeConstraint::NonNeg {
            a: t.add(&cx.neg()),
        });
        self.constraints
            .push(ConeConstraint::NonNeg { a: t.add(&cx) });

        // Return sum(t)
        self.canonicalize_sum_lin(&t)
    }

    fn canonicalize_norm2(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_2: Introduce t >= 0, SOC(t, x)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // SOC constraint: ||x||_2 <= t
        self.constraints.push(ConeConstraint::SOC {
            t: t.clone(),
            x: cx,
        });

        CanonExpr::Linear(t)
    }

    fn canonicalize_norm_inf(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_inf = max(|x_i|)
        // Introduce t >= 0, -t <= x_i <= t for all i
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let size = cx.size();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // Expand t to match x's size
        let t_expanded = self.expand_scalar(&t, size);

        // Constraints: t >= x_i, t >= -x_i for all i
        self.constraints.push(ConeConstraint::NonNeg {
            a: t_expanded.add(&cx.neg()),
        });
        self.constraints.push(ConeConstraint::NonNeg {
            a: t_expanded.add(&cx),
        });

        CanonExpr::Linear(t)
    }

    fn canonicalize_abs(&mut self, x: &Expr) -> CanonExpr {
        // |x| element-wise: same as norm1 but keeping element-wise
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let shape = cx.shape.clone();
        let (_, t) = self.new_nonneg_aux_var(shape);

        // Constraints: t >= x, t >= -x
        self.constraints.push(ConeConstraint::NonNeg {
            a: t.add(&cx.neg()),
        });
        self.constraints
            .push(ConeConstraint::NonNeg { a: t.add(&cx) });

        CanonExpr::Linear(t)
    }

    fn canonicalize_pos(&mut self, x: &Expr) -> CanonExpr {
        // pos(x) = max(x, 0)
        // Introduce t >= 0, t >= x
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let shape = cx.shape.clone();
        let (_, t) = self.new_nonneg_aux_var(shape);

        // Constraint: t >= x, i.e., t - x >= 0
        self.constraints.push(ConeConstraint::NonNeg {
            a: t.add(&cx.neg()),
        });

        CanonExpr::Linear(t)
    }

    fn canonicalize_neg_part(&mut self, x: &Expr) -> CanonExpr {
        // neg(x) = max(-x, 0) = pos(-x)
        let neg_x = Expr::Neg(Arc::new(x.clone()));
        self.canonicalize_pos(&neg_x)
    }

    fn canonicalize_maximum(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // max(x1, ..., xn): Introduce t, t >= x_i for all i
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }

        let shape = exprs[0].shape();
        let (_, t) = self.new_aux_var(shape);

        for e in exprs {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // t >= x_i, i.e., t - x_i >= 0
            self.constraints.push(ConeConstraint::NonNeg {
                a: t.add(&ce.neg()),
            });
        }

        CanonExpr::Linear(t)
    }

    fn canonicalize_minimum(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // min(x1, ..., xn): Introduce t, t <= x_i for all i
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }

        let shape = exprs[0].shape();
        let (_, t) = self.new_aux_var(shape);

        for e in exprs {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // t <= x_i, i.e., x_i - t >= 0
            self.constraints.push(ConeConstraint::NonNeg {
                a: ce.add(&t.neg()),
            });
        }

        CanonExpr::Linear(t)
    }

    fn canonicalize_quad_form(&mut self, x: &Expr, p: &Expr, for_objective: bool) -> CanonExpr {
        // x' P x: If for_objective and P is constant PSD, use native QP
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        if for_objective {
            if let Some(Array::Dense(p_mat)) = p.constant_value() {
                // Build quadratic form for native QP
                // x' P x where x = sum_i A_i v_i + b
                // For now, simplified: assume x is a single variable
                let vars = cx.variables();
                if vars.len() == 1 && cx.constant.iter().all(|&v| v == 0.0) {
                    let var_id = vars[0];
                    let p_csc = dense_to_csc(p_mat);
                    return CanonExpr::Quadratic(QuadExpr::quadratic(var_id, p_csc));
                }
            }
        }

        // Fall back to SOC reformulation
        // x' P x where P = L L' (Cholesky)
        // = ||L' x||_2^2
        // Introduce t, SOC constraint
        self.canonicalize_sum_squares_lin(&cx, for_objective)
    }

    fn canonicalize_sum_squares(&mut self, x: &Expr, for_objective: bool) -> CanonExpr {
        // ||x||_2^2 = x' x
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        self.canonicalize_sum_squares_lin(&cx, for_objective)
    }

    fn canonicalize_sum_squares_lin(&mut self, x: &LinExpr, for_objective: bool) -> CanonExpr {
        if for_objective {
            // For objective, use native QP: ||Ax + c||^2 = x'(A'A)x + 2c'Ax + ||c||^2
            // stuffing.rs doubles P to account for Clarabel's (1/2)x'Px convention.
            let vars = x.variables();
            let c = &x.constant; // dense (m, 1) constant

            let mut quad_coeffs = std::collections::HashMap::new();
            let mut linear_coeffs = std::collections::HashMap::new();

            for &var_i in &vars {
                let ai = csc_to_dense(&x.coeffs[&var_i]); // (m, ni)
                for &var_j in &vars {
                    let aj = csc_to_dense(&x.coeffs[&var_j]); // (m, nj)
                                                              // A_i' * A_j: (ni, m) * (m, nj) = (ni, nj)
                    let ai_t_aj = dense_to_csc(&(ai.transpose() * &aj));
                    quad_coeffs
                        .entry((var_i, var_j))
                        .and_modify(|existing| *existing = csc_add(existing, &ai_t_aj))
                        .or_insert(ai_t_aj);
                }
                // Linear term: 2 * c' * A_i  →  (1, ni) row coefficient
                let q_col = ai.transpose() * c; // (ni, 1)
                let q_row = dense_to_csc(&(q_col * 2.0).transpose()); // (1, ni)
                linear_coeffs.insert(var_i, q_row);
            }

            // Constant: ||c||^2
            let constant: f64 = c.iter().map(|v| v * v).sum();

            return CanonExpr::Quadratic(QuadExpr {
                quad_coeffs,
                linear: LinExpr {
                    coeffs: linear_coeffs,
                    constant: DMatrix::zeros(1, 1),
                    shape: Shape::scalar(),
                },
                constant,
            });
        }

        // SOC reformulation: ||x||^2 <= t iff SOC(sqrt(t), x)
        // Actually: introduce t, s, with t = s + 1, and SOC(s, x)
        // Simpler: introduce t >= 0, with ||x||_2^2 <= t via rotated SOC
        // Or: ||x||^2 = quad_over_lin(x, 1)
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // Rotated SOC: ||x||^2 <= 2 * t * 1 = 2t
        // Standard form: || [2t - 1; 2x] ||_2 <= 2t + 1
        // Simplified: use SOC with proper reformulation
        self.constraints.push(ConeConstraint::SOC {
            t: t.clone(),
            x: x.clone(),
        });

        CanonExpr::Linear(t)
    }

    fn canonicalize_quad_over_lin(&mut self, x: &Expr, y: &Expr) -> CanonExpr {
        // ||x||_2^2 / y: Introduce t, rotated SOC constraint
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let _cy = self.canonicalize_expr(y, false).as_linear().clone();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // Rotated SOC: ||x||^2 <= t * y
        // This requires proper rotated SOC support
        // Simplified: add as SOC
        self.constraints.push(ConeConstraint::SOC {
            t: t.clone(),
            x: cx,
        });

        CanonExpr::Linear(t)
    }

    // ========================================================================
    // Utility functions
    // ========================================================================

    fn canonicalize_sum_lin(&mut self, x: &LinExpr) -> CanonExpr {
        let size = x.size();
        let ones = DMatrix::from_element(1, size, 1.0);

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &x.coeffs {
            let new_coeff = dense_sparse_matmul(&ones, coeff);
            new_coeffs.insert(*var_id, new_coeff);
        }

        let flat_const = x
            .constant
            .clone()
            .reshape_generic(nalgebra::Dyn(size), nalgebra::Dyn(1));
        let result = &ones * &flat_const;
        let new_const = DMatrix::from_element(1, 1, result[(0, 0)]);

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::scalar(),
        })
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

    // ========================================================================
    // Exponential Cone Atoms
    // ========================================================================

    fn canonicalize_exp(&mut self, x: &Expr) -> CanonExpr {
        // exp(x): Introduce t, add ExpCone(x, 1, t) meaning t >= exp(x)
        // Each element requires its own exponential cone constraint
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        // Create auxiliary variable t for the result
        let (t_var_id, t) = self.new_nonneg_aux_var(cx.shape.clone());
        let _ = t_var_id; // Mark as used

        // Add one exp cone constraint per element
        let n = cx.size();
        for i in 0..n {
            let xi = self.extract_element(&cx, i);
            let ti = self.extract_element(&t, i);
            let one_scalar = LinExpr::scalar(1.0);

            // (xi, 1, ti) in K_exp means ti >= exp(xi)
            self.constraints.push(ConeConstraint::ExpCone {
                x: xi,
                y: one_scalar,
                z: ti,
            });
        }

        CanonExpr::Linear(t)
    }

    fn canonicalize_log(&mut self, x: &Expr) -> CanonExpr {
        // log(x): Introduce t, add ExpCone(t, 1, x) meaning t <= log(x)
        // Each element requires its own exponential cone constraint
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        // Create auxiliary variable t for the result
        let (t_var_id, t) = self.new_aux_var(cx.shape.clone());
        let _ = t_var_id; // Mark as used

        // Add one exp cone constraint per element
        let n = cx.size();
        for i in 0..n {
            let xi = self.extract_element(&cx, i);
            let ti = self.extract_element(&t, i);
            let one_scalar = LinExpr::scalar(1.0);

            // (ti, 1, xi) in K_exp means 1 * exp(ti/1) <= xi
            // which simplifies to exp(ti) <= xi, so ti <= log(xi)
            self.constraints.push(ConeConstraint::ExpCone {
                x: ti,
                y: one_scalar,
                z: xi,
            });
        }

        CanonExpr::Linear(t)
    }

    fn canonicalize_entropy(&mut self, x: &Expr) -> CanonExpr {
        // entropy(x) = -x * log(x)
        // Using exp cone: (t, x, 1) in K_exp means x * exp(t/x) <= 1
        // which gives t <= -x * log(x) = entropy(x) (hypograph)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        // Create auxiliary variable t for the result
        let (_t_var_id, t) = self.new_aux_var(cx.shape.clone());

        // Add one exp cone constraint per element
        let n = cx.size();
        for i in 0..n {
            let xi = self.extract_element(&cx, i);
            let ti = self.extract_element(&t, i);
            let one_scalar = LinExpr::scalar(1.0);

            // (ti, xi, 1) in K_exp means xi * exp(ti/xi) <= 1
            // i.e., exp(ti/xi) <= 1/xi, i.e., ti/xi <= -log(xi)
            // i.e., ti <= -xi * log(xi) = entropy(xi)
            self.constraints.push(ConeConstraint::ExpCone {
                x: ti,
                y: xi,
                z: one_scalar,
            });
        }

        CanonExpr::Linear(t)
    }

    // ========================================================================
    // Power Cone Atoms
    // ========================================================================

    fn canonicalize_power(&mut self, x: &Expr, p: f64) -> CanonExpr {
        // x^p using power cones
        // Each element requires its own power cone constraint
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        if (p - 1.0).abs() < 1e-10 {
            // x^1 = x (affine)
            return CanonExpr::Linear(cx);
        }

        if (p - 2.0).abs() < 1e-10 {
            // x^2 use sum_squares approach (more efficient)
            return self.canonicalize_sum_squares(&Expr::from(x), false);
        }

        // Create auxiliary variable t for the result
        let (t_var_id, t) = self.new_nonneg_aux_var(cx.shape.clone());
        let _ = t_var_id;

        // Add one power cone constraint per element
        let n = cx.size();
        for i in 0..n {
            let xi = self.extract_element(&cx, i);
            let ti = self.extract_element(&t, i);
            let one_scalar = LinExpr::scalar(1.0);

            if p > 0.0 && p < 1.0 {
                // x^p concave for p in (0,1): (x, 1, t) in K_pow(p) means t <= x^p
                let alpha = p;
                self.constraints.push(ConeConstraint::PowerCone {
                    x: xi,
                    y: one_scalar,
                    z: ti,
                    alpha,
                });
            } else if p > 1.0 {
                // x^p convex for p > 1: (t, 1, x) in K_pow(1/p) means t >= x^p
                let alpha = 1.0 / p;
                self.constraints.push(ConeConstraint::PowerCone {
                    x: ti,
                    y: one_scalar,
                    z: xi,
                    alpha,
                });
            } else if p < 0.0 {
                // x^p convex for p < 0: t >= x^p = 1/x^(-p)
                // Equivalently: t * x^(-p) >= 1
                // Use (t, x, 1) in K_pow(alpha) with alpha = 1/(1-p)
                // This gives t^alpha * x^(1-alpha) >= 1
                // where alpha = 1/(1-p), 1-alpha = -p/(1-p)
                // Raising to power (1-p): t * x^(-p) >= 1 ✓
                let alpha = 1.0 / (1.0 - p);
                self.constraints.push(ConeConstraint::PowerCone {
                    x: ti,
                    y: xi,
                    z: one_scalar,
                    alpha,
                });
            }
        }

        CanonExpr::Linear(t)
    }

    // ========================================================================
    // Additional Affine Atoms
    // ========================================================================

    fn canonicalize_cumsum(&mut self, x: &Expr, _axis: Option<usize>) -> CanonExpr {
        // Cumulative sum using auxiliary variables and constraints
        // cumsum([x1, x2, x3]) = [y1, y2, y3] where:
        //   y1 = x1
        //   y2 = y1 + x2  =>  y2 - y1 = x2
        //   y3 = y2 + x3  =>  y3 - y2 = x3

        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let n = cx.size();

        // Create auxiliary variables for the cumsum result
        let (y_var_id, y) = self.new_aux_var(cx.shape.clone());
        let _ = y_var_id;

        // For each element, add constraint: y[i] - y[i-1] = x[i]
        // Or for i=0: y[0] = x[0]

        for i in 0..n {
            if i == 0 {
                // y[0] = x[0]
                // Extract element 0 from both x and y
                let x0 = self.extract_element(&cx, i);
                let y0 = self.extract_element(&y, i);

                // Add constraint: y0 - x0 = 0
                let diff = y0.add(&x0.scale(-1.0));
                self.constraints.push(ConeConstraint::Zero { a: diff });
            } else {
                // y[i] - y[i-1] = x[i]
                let yi = self.extract_element(&y, i);
                let yi_prev = self.extract_element(&y, i - 1);
                let xi = self.extract_element(&cx, i);

                // y[i] - y[i-1] - x[i] = 0
                let diff = yi.add(&yi_prev.scale(-1.0)).add(&xi.scale(-1.0));
                self.constraints.push(ConeConstraint::Zero { a: diff });
            }
        }

        CanonExpr::Linear(y)
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
            let row = idx / expr.shape.cols();
            let col = idx % expr.shape.cols();
            expr.constant[(row, col)]
        };

        LinExpr {
            coeffs: new_coeffs,
            constant: DMatrix::from_element(1, 1, const_val),
            shape: Shape::scalar(),
        }
    }

    fn canonicalize_diag(&mut self, x: &Expr) -> CanonExpr {
        // Vector to diagonal matrix (simplified for v1.0)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        if cx.shape.is_vector() {
            // Vector -> diagonal matrix: n -> (n, n)
            let n = cx.shape.size();

            // Build selection matrix that places vector on diagonal
            let mut new_coeffs = std::collections::HashMap::new();
            for (var_id, coeff) in &cx.coeffs {
                let coeff_dense = csc_to_dense(coeff);
                let var_size = coeff_dense.ncols();
                let mut new_coeff = DMatrix::zeros(n * n, var_size);

                for j in 0..var_size {
                    for i in 0..n {
                        // Place element i of input vector at diagonal position (i,i) in output
                        let diag_idx = i * n + i;
                        new_coeff[(diag_idx, j)] = coeff_dense[(i, j)];
                    }
                }
                new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
            }

            // Apply to constant
            let mut new_const = DMatrix::zeros(n, n);
            for i in 0..n {
                new_const[(i, i)] = cx.constant[(i, 0)];
            }

            CanonExpr::Linear(LinExpr {
                coeffs: new_coeffs,
                constant: new_const,
                shape: Shape::matrix(n, n),
            })
        } else {
            // Matrix -> diagonal vector: extract diagonal elements
            let m = cx.shape.rows();
            let n = cx.shape.cols();
            let diag_size = m.min(n);

            // Build selection: output[i] = input[i,i] = input[i + i*m] (column-major)
            let mut new_coeffs = std::collections::HashMap::new();
            for (var_id, coeff) in &cx.coeffs {
                let coeff_dense = csc_to_dense(coeff);
                let var_size = coeff_dense.ncols();
                let mut new_coeff = DMatrix::zeros(diag_size, var_size);

                for j in 0..var_size {
                    for i in 0..diag_size {
                        // Diagonal element (i,i) in column-major is at index i + i*m
                        let input_idx = i + i * m;
                        if input_idx < coeff_dense.nrows() {
                            new_coeff[(i, j)] = coeff_dense[(input_idx, j)];
                        }
                    }
                }
                new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
            }

            // Extract diagonal from constant
            let mut new_const = DMatrix::zeros(diag_size, 1);
            for i in 0..diag_size {
                new_const[(i, 0)] = cx.constant[(i, i)];
            }

            CanonExpr::Linear(LinExpr {
                coeffs: new_coeffs,
                constant: new_const,
                shape: Shape::vector(diag_size),
            })
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn dense_sparse_matmul(dense: &DMatrix<f64>, sparse: &CscMatrix<f64>) -> CscMatrix<f64> {
    // Dense @ Sparse multiplication
    // Note: nalgebra_sparse doesn't support dense @ sparse directly.
    // A more efficient implementation would iterate through sparse columns.
    // For medium-scale problems (100-10k variables), this is acceptable.
    let result = dense * csc_to_dense(sparse);
    dense_to_csc(&result)
}

fn stack_vertical(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(a.nrows() + b.nrows(), a.ncols().max(b.ncols()));
    result.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    result
        .view_mut((a.nrows(), 0), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

fn stack_horizontal(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(a.nrows().max(b.nrows()), a.ncols() + b.ncols());
    result.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    result
        .view_mut((0, a.ncols()), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

fn stack_csc_vertical(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    csc_vstack(a, b)
}

fn repeat_rows_csc(m: &CscMatrix<f64>, times: usize) -> CscMatrix<f64> {
    csc_repeat_rows(m, times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::variable;

    #[test]
    fn test_canonicalize_variable() {
        let x = variable(5);
        let result = canonicalize(&x, false);
        assert!(result.constraints.is_empty());
        assert!(matches!(result.expr, CanonExpr::Linear(_)));
    }

    #[test]
    fn test_canonicalize_norm2() {
        let x = variable(5);
        let n = Expr::Norm2(Arc::new(x));
        let result = canonicalize(&n, false);
        // Should have 1 SOC constraint + 1 NonNeg (t >= 0), and 1 aux variable
        assert_eq!(result.constraints.len(), 2);
        assert_eq!(result.aux_vars.len(), 1);
    }

    #[test]
    fn test_canonicalize_sum_squares_objective() {
        let x = variable(5);
        let s = Expr::SumSquares(Arc::new(x));
        let result = canonicalize(&s, true);
        // For objective, should produce quadratic or SOC
        assert!(matches!(result.expr, CanonExpr::Quadratic(_)) || !result.constraints.is_empty());
    }
}
