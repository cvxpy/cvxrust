/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Linear and quadratic expression representations for canonicalization.
//!
//! After canonicalization, expressions are represented in standard form:
//! - Linear: sum_i(A_i * x_i) + b
//! - Quadratic: (1/2) x' P x + q' x + r

use std::collections::HashMap;

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use crate::expr::{ExprId, Shape};
use crate::sparse::{csc_add, csc_neg, csc_scale};

/// A linear expression in standard form: sum_i(A_i * x_i) + b
///
/// Each term is a sparse coefficient matrix multiplied by a variable.
/// The constant term `b` is a dense vector.
#[derive(Debug, Clone)]
pub struct LinExpr {
    /// Coefficient matrices for each variable: var_id -> coefficient matrix.
    /// The coefficient matrix A_i has shape (output_size, var_size).
    pub coeffs: HashMap<ExprId, CscMatrix<f64>>,
    /// Constant term (offset).
    pub constant: DMatrix<f64>,
    /// Output shape of this expression.
    pub shape: Shape,
}

impl LinExpr {
    /// Create a zero linear expression with the given shape.
    pub fn zeros(shape: Shape) -> Self {
        let rows = shape.rows();
        let cols = shape.cols();
        LinExpr {
            coeffs: HashMap::new(),
            constant: DMatrix::zeros(rows, cols),
            shape,
        }
    }

    /// Create a linear expression for a single variable (identity coefficient).
    pub fn variable(var_id: ExprId, shape: Shape) -> Self {
        let size = shape.size();
        // Identity matrix: var maps to itself
        let identity = CscMatrix::identity(size);
        let mut coeffs = HashMap::new();
        coeffs.insert(var_id, identity);
        LinExpr {
            coeffs,
            constant: DMatrix::zeros(shape.rows(), shape.cols()),
            shape,
        }
    }

    /// Create a constant linear expression.
    pub fn constant(value: DMatrix<f64>) -> Self {
        let shape = Shape::matrix(value.nrows(), value.ncols());
        LinExpr {
            coeffs: HashMap::new(),
            constant: value,
            shape,
        }
    }

    /// Create a scalar constant.
    pub fn scalar(value: f64) -> Self {
        LinExpr {
            coeffs: HashMap::new(),
            constant: DMatrix::from_element(1, 1, value),
            shape: Shape::scalar(),
        }
    }

    /// Check if this is a constant (no variables).
    pub fn is_constant(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Get the output size (flattened).
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Add two linear expressions.
    pub fn add(&self, other: &LinExpr) -> LinExpr {
        let new_shape =
            if self.shape.rows() == other.shape.rows() && self.shape.cols() == other.shape.cols() {
                if self.shape.is_vector() {
                    self.shape.clone()
                } else if other.shape.is_vector() {
                    other.shape.clone()
                } else {
                    self.shape.clone()
                }
            } else if self.shape.is_scalar_like() && !other.shape.is_scalar_like() {
                other.shape.clone()
            } else if other.shape.is_scalar_like() && !self.shape.is_scalar_like() {
                self.shape.clone()
            } else if self.shape.size() == other.shape.size() {
                Shape::vector(self.shape.size())
            } else {
                panic!(
                    "cannot add linear expressions with shapes {} and {}",
                    self.shape, other.shape
                )
            };
        let new_size = new_shape.size();

        // Optimization: if self has no coefficients, just clone other's
        let coeffs = if self.coeffs.is_empty() {
            other
                .coeffs
                .iter()
                .map(|(var_id, coeff)| (*var_id, coeff_for_size(coeff, other.size(), new_size)))
                .collect()
        } else if other.coeffs.is_empty() {
            self.coeffs
                .iter()
                .map(|(var_id, coeff)| (*var_id, coeff_for_size(coeff, self.size(), new_size)))
                .collect()
        } else {
            // Both have coefficients - clone the larger one and merge smaller into it
            let mut coeffs = HashMap::new();
            for (var_id, coeff) in &self.coeffs {
                coeffs.insert(*var_id, coeff_for_size(coeff, self.size(), new_size));
            }
            coeffs.reserve(other.coeffs.len());
            for (var_id, coeff) in &other.coeffs {
                let coeff = coeff_for_size(coeff, other.size(), new_size);
                coeffs
                    .entry(*var_id)
                    .and_modify(|c| *c = csc_add(c, &coeff))
                    .or_insert(coeff);
            }
            coeffs
        };

        // Handle broadcasting for constants
        let new_constant = if self.constant.nrows() == other.constant.nrows()
            && self.constant.ncols() == other.constant.ncols()
        {
            &self.constant + &other.constant
        } else if other.constant.nrows() == 1 && other.constant.ncols() == 1 {
            // Broadcast scalar other to match self's shape
            let scalar = other.constant[(0, 0)];
            self.constant.map(|v| v + scalar)
        } else if self.constant.nrows() == 1 && self.constant.ncols() == 1 {
            // Broadcast scalar self to match other's shape
            let scalar = self.constant[(0, 0)];
            other.constant.map(|v| v + scalar)
        } else if self.shape.size() == other.shape.size() {
            let self_flat = self
                .constant
                .clone()
                .reshape_generic(nalgebra::Dyn(new_size), nalgebra::Dyn(1));
            let other_flat = other
                .constant
                .clone()
                .reshape_generic(nalgebra::Dyn(new_size), nalgebra::Dyn(1));
            self_flat + other_flat
        } else {
            panic!(
                "cannot add linear expression constants with shapes {} and {}",
                self.shape, other.shape
            );
        };

        LinExpr {
            coeffs,
            constant: new_constant,
            shape: new_shape,
        }
    }

    /// Negate a linear expression.
    pub fn neg(&self) -> LinExpr {
        let coeffs = self.coeffs.iter().map(|(k, v)| (*k, csc_neg(v))).collect();
        LinExpr {
            coeffs,
            constant: -&self.constant,
            shape: self.shape.clone(),
        }
    }

    /// Scale by a scalar.
    pub fn scale(&self, scalar: f64) -> LinExpr {
        let coeffs = self
            .coeffs
            .iter()
            .map(|(k, v)| (*k, csc_scale(v, scalar)))
            .collect();
        LinExpr {
            coeffs,
            constant: &self.constant * scalar,
            shape: self.shape.clone(),
        }
    }

    /// Get all variable IDs in this expression.
    pub fn variables(&self) -> Vec<ExprId> {
        let mut vars: Vec<_> = self.coeffs.keys().copied().collect();
        vars.sort_by_key(|id| id.raw());
        vars
    }
}

fn coeff_for_size(
    coeff: &CscMatrix<f64>,
    source_size: usize,
    target_size: usize,
) -> CscMatrix<f64> {
    if coeff.nrows() == target_size {
        return coeff.clone();
    }
    if source_size == 1 {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (row, col, val) in coeff.triplet_iter() {
            debug_assert_eq!(row, 0);
            for target_row in 0..target_size {
                rows.push(target_row);
                cols.push(col);
                vals.push(*val);
            }
        }
        return crate::sparse::triplets_to_csc(target_size, coeff.ncols(), &rows, &cols, &vals);
    }
    panic!(
        "cannot resize coefficient rows from {} to {}",
        coeff.nrows(),
        target_size
    );
}

/// A quadratic expression: (1/2) x' P x + q' x + r
///
/// Used for quadratic objectives in QP problems.
#[derive(Debug, Clone)]
pub struct QuadExpr {
    /// Quadratic term: P matrix (symmetric, in packed form by variable).
    /// Maps (var_i, var_j) -> coefficient for x_i' P_ij x_j term.
    pub quad_coeffs: HashMap<(ExprId, ExprId), CscMatrix<f64>>,
    /// Linear term: q' x
    pub linear: LinExpr,
    /// Constant term: r
    pub constant: f64,
}

impl QuadExpr {
    /// Create a quadratic expression from a linear expression.
    pub fn from_linear(linear: LinExpr) -> Self {
        let constant = if linear.constant.nrows() == 1 && linear.constant.ncols() == 1 {
            linear.constant[(0, 0)]
        } else {
            0.0
        };
        QuadExpr {
            quad_coeffs: HashMap::new(),
            linear: LinExpr {
                coeffs: linear.coeffs,
                constant: DMatrix::zeros(1, 1),
                shape: Shape::scalar(),
            },
            constant,
        }
    }

    /// Create a pure quadratic term: x' P x for a single variable.
    pub fn quadratic(var_id: ExprId, p: CscMatrix<f64>) -> Self {
        let mut quad_coeffs = HashMap::new();
        quad_coeffs.insert((var_id, var_id), p);
        QuadExpr {
            quad_coeffs,
            linear: LinExpr::zeros(Shape::scalar()),
            constant: 0.0,
        }
    }

    /// Check if this is purely linear (no quadratic terms).
    pub fn is_linear(&self) -> bool {
        self.quad_coeffs.is_empty()
    }

    /// Add two quadratic expressions.
    pub fn add(&self, other: &QuadExpr) -> QuadExpr {
        let mut quad_coeffs = self.quad_coeffs.clone();
        for (key, coeff) in &other.quad_coeffs {
            quad_coeffs
                .entry(*key)
                .and_modify(|c| *c = csc_add(c, coeff))
                .or_insert_with(|| coeff.clone());
        }
        QuadExpr {
            quad_coeffs,
            linear: self.linear.add(&other.linear),
            constant: self.constant + other.constant,
        }
    }

    /// Scale by a scalar.
    pub fn scale(&self, scalar: f64) -> QuadExpr {
        let quad_coeffs = self
            .quad_coeffs
            .iter()
            .map(|(k, v)| (*k, csc_scale(v, scalar)))
            .collect();
        QuadExpr {
            quad_coeffs,
            linear: self.linear.scale(scalar),
            constant: self.constant * scalar,
        }
    }

    /// Get all variable IDs in this expression.
    pub fn variables(&self) -> Vec<ExprId> {
        let mut vars: Vec<_> = self.linear.variables();
        for (v1, v2) in self.quad_coeffs.keys() {
            vars.push(*v1);
            vars.push(*v2);
        }
        vars.sort_by_key(|id| id.raw());
        vars.dedup();
        vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lin_expr_zeros() {
        let e = LinExpr::zeros(Shape::vector(5));
        assert!(e.is_constant());
        assert_eq!(e.size(), 5);
    }

    #[test]
    fn test_lin_expr_variable() {
        let var_id = ExprId::new();
        let e = LinExpr::variable(var_id, Shape::vector(3));
        assert!(!e.is_constant());
        assert_eq!(e.variables(), vec![var_id]);
    }

    #[test]
    fn test_lin_expr_add() {
        let var1 = ExprId::new();
        let var2 = ExprId::new();
        let e1 = LinExpr::variable(var1, Shape::vector(3));
        let e2 = LinExpr::variable(var2, Shape::vector(3));
        let sum = e1.add(&e2);
        assert_eq!(sum.variables().len(), 2);
    }

    #[test]
    #[should_panic(expected = "cannot add linear expressions with shapes (2,) and (3,)")]
    fn test_lin_expr_add_incompatible_shapes_panics() {
        let e1 = LinExpr::variable(ExprId::new(), Shape::vector(2));
        let e2 = LinExpr::variable(ExprId::new(), Shape::vector(3));
        let _ = e1.add(&e2);
    }

    #[test]
    fn test_quad_expr_from_linear() {
        let var_id = ExprId::new();
        let lin = LinExpr::variable(var_id, Shape::scalar());
        let quad = QuadExpr::from_linear(lin);
        assert!(quad.is_linear());
    }
}
