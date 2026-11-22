/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Stacking expressions and their coefficient matrices.

use std::sync::Arc;

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::{CanonContext, CanonExpr};
use crate::canon::lin_expr::LinExpr;
use crate::expr::{Expr, Shape};
use crate::sparse::csc_vstack;

impl CanonContext<'_> {
    pub(super) fn canonicalize_vstack(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
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

    pub(super) fn vstack_lin(&self, a: &LinExpr, b: &LinExpr) -> LinExpr {
        assert_eq!(
            a.shape.cols(),
            b.shape.cols(),
            "vstack requires matching column counts"
        );
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
                (Some(ma), Some(mb)) => stack_csc_vertical_for_shapes(ma, mb, &a.shape, &b.shape),
                (Some(ma), None) => {
                    let zeros = CscMatrix::zeros(b.size(), ma.ncols());
                    stack_csc_vertical_for_shapes(ma, &zeros, &a.shape, &b.shape)
                }
                (None, Some(mb)) => {
                    let zeros = CscMatrix::zeros(a.size(), mb.ncols());
                    stack_csc_vertical_for_shapes(&zeros, mb, &a.shape, &b.shape)
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

    pub(super) fn canonicalize_hstack(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
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
        assert_eq!(
            a.shape.rows(),
            b.shape.rows(),
            "hstack requires matching row counts"
        );
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
}

fn stack_vertical(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(
        a.ncols(),
        b.ncols(),
        "vstack requires matching column counts"
    );
    let mut result = DMatrix::zeros(a.nrows() + b.nrows(), a.ncols());
    result.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    result
        .view_mut((a.nrows(), 0), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

fn stack_horizontal(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(a.nrows(), b.nrows(), "hstack requires matching row counts");
    let mut result = DMatrix::zeros(a.nrows(), a.ncols() + b.ncols());
    result.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    result
        .view_mut((0, a.ncols()), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

fn stack_csc_vertical_for_shapes(
    a: &CscMatrix<f64>,
    b: &CscMatrix<f64>,
    a_shape: &Shape,
    b_shape: &Shape,
) -> CscMatrix<f64> {
    debug_assert_eq!(
        a_shape.cols(),
        b_shape.cols(),
        "vstack requires matching column counts"
    );
    debug_assert_eq!(a.ncols(), b.ncols());

    let a_rows = a_shape.rows();
    let b_rows = b_shape.rows();
    let cols = a_shape.cols();
    let out_rows = a_rows + b_rows;

    let mut rows = Vec::new();
    let mut col_indices = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in a.triplet_iter() {
        let matrix_col = r / a_rows;
        let matrix_row = r % a_rows;
        debug_assert!(matrix_col < cols);
        rows.push(matrix_row + matrix_col * out_rows);
        col_indices.push(c);
        vals.push(*v);
    }
    for (r, c, v) in b.triplet_iter() {
        let matrix_col = r / b_rows;
        let matrix_row = r % b_rows;
        debug_assert!(matrix_col < cols);
        rows.push(a_rows + matrix_row + matrix_col * out_rows);
        col_indices.push(c);
        vals.push(*v);
    }

    crate::sparse::triplets_to_csc(
        a_shape.size() + b_shape.size(),
        a.ncols(),
        &rows,
        &col_indices,
        &vals,
    )
}
