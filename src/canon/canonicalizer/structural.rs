/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Shape transformations and indexing.

use nalgebra::DMatrix;

use super::{CanonContext, CanonExpr};
use crate::canon::lin_expr::LinExpr;
use crate::expr::{Expr, IndexSpec, Shape};
use crate::sparse::{csc_repeat_rows, csc_to_dense, dense_to_csc};

impl CanonContext<'_> {
    pub(super) fn canonicalize_promote(&mut self, expr: &Expr, target_shape: &Shape) -> CanonExpr {
        let lin = self.canonicalize_expr(expr, false).as_linear().clone();
        if lin.shape.size() != 1 {
            return CanonExpr::Linear(lin);
        }

        let coeffs = lin
            .coeffs
            .iter()
            .map(|(var_id, coeff)| (*var_id, csc_repeat_rows(coeff, target_shape.size())))
            .collect();
        let constant = DMatrix::from_element(
            target_shape.rows(),
            target_shape.cols(),
            lin.constant[(0, 0)],
        );

        CanonExpr::Linear(LinExpr {
            coeffs,
            constant,
            shape: target_shape.clone(),
        })
    }

    pub(super) fn canonicalize_reshape(&mut self, a: &Expr, shape: &Shape) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        assert_eq!(
            ca.shape.size(),
            shape.size(),
            "cannot reshape size {} into shape {}",
            ca.shape.size(),
            shape
        );
        // Reshape doesn't change the linear structure, just the shape interpretation
        CanonExpr::Linear(LinExpr {
            coeffs: ca.coeffs,
            constant: ca
                .constant
                .reshape_generic(nalgebra::Dyn(shape.rows()), nalgebra::Dyn(shape.cols())),
            shape: shape.clone(),
        })
    }

    pub(super) fn canonicalize_index(&mut self, a: &Expr, spec: &IndexSpec) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        let new_shape = spec.output_shape(&ca.shape);
        let input_rows = ca.shape.rows();
        let input_cols = ca.shape.cols();

        let input_indices = match spec.ranges.as_slice() {
            [Some((start, stop, step))] => (*start..*stop).step_by(*step).collect(),
            [None] => (0..ca.shape.size()).collect(),
            [row_spec, col_spec] => {
                let row_range: Vec<usize> = match row_spec {
                    Some((start, stop, step)) => (*start..*stop).step_by(*step).collect(),
                    None => (0..input_rows).collect(),
                };
                let col_range: Vec<usize> = match col_spec {
                    Some((start, stop, step)) => (*start..*stop).step_by(*step).collect(),
                    None => (0..input_cols).collect(),
                };
                let mut indices = Vec::with_capacity(row_range.len() * col_range.len());
                for col in col_range {
                    for &row in &row_range {
                        indices.push(row + col * input_rows);
                    }
                }
                indices
            }
            _ => (0..ca.shape.size()).collect(),
        };

        let output_size = input_indices.len();
        let mut s_rows = Vec::new();
        let mut s_cols = Vec::new();
        let mut s_vals = Vec::new();
        for (out_idx, &in_idx) in input_indices.iter().enumerate() {
            if in_idx < ca.shape.size() {
                s_rows.push(out_idx);
                s_cols.push(in_idx);
                s_vals.push(1.0);
            }
        }
        let s_mat =
            crate::sparse::triplets_to_csc(output_size, ca.shape.size(), &s_rows, &s_cols, &s_vals);

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &ca.coeffs {
            new_coeffs.insert(*var_id, crate::sparse::csc_matmul(&s_mat, coeff));
        }

        let const_flat: Vec<f64> = ca.constant.iter().cloned().collect();
        let new_const_vals: Vec<f64> = input_indices
            .iter()
            .map(|&i| {
                if i < const_flat.len() {
                    const_flat[i]
                } else {
                    0.0
                }
            })
            .collect();
        let new_const = DMatrix::from_vec(new_shape.rows(), new_shape.cols(), new_const_vals);

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: new_shape,
        })
    }

    pub(super) fn canonicalize_transpose(&mut self, a: &Expr) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        let m = ca.shape.rows();
        let n = ca.shape.cols();
        let new_shape = ca.shape.transpose();

        // Transpose the constant matrix
        let new_const = ca.constant.transpose();

        // For coefficients: we need to permute rows to match the transpose
        // Original flat index: i + j*m (column-major for m x n matrix)
        // Transposed flat index: j + i*n (column-major for n x m matrix)
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

    pub(super) fn canonicalize_diag(&mut self, x: &Expr) -> CanonExpr {
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
