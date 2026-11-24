/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Affine arithmetic, products, and reductions.

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::{CanonContext, CanonExpr, ConeConstraint};
use crate::canon::lin_expr::LinExpr;
use crate::expr::{Array, Expr, Shape};
use crate::sparse::{csc_to_dense, dense_to_csc};

impl CanonContext<'_> {
    pub(super) fn canonicalize_constant(&self, arr: &Array) -> LinExpr {
        match arr {
            Array::Scalar(v) => LinExpr::scalar(*v),
            Array::Dense(m) => LinExpr::constant(m.clone()),
            Array::Sparse(s) => {
                // Convert sparse to dense
                LinExpr::constant(csc_to_dense(s))
            }
        }
    }

    pub(super) fn canonicalize_mul(
        &mut self,
        a: &Expr,
        b: &Expr,
        for_objective: bool,
    ) -> CanonExpr {
        // Zero terms can be constant while still referencing variables with bounds.
        let a_is_const = a.variables().is_empty() || a.curvature().is_constant();
        let b_is_const = b.variables().is_empty() || b.curvature().is_constant();

        // Handle scalar multiplication first (most common case)
        if let Some(scalar) = scalar_constant_value(a) {
            let cb = self.canonicalize_expr(b, for_objective);
            return match cb {
                CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
            };
        }
        if let Some(scalar) = scalar_constant_value(b) {
            let ca = self.canonicalize_expr(a, for_objective);
            return match ca {
                CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
            };
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

        panic!("cannot canonicalize product of two non-constant expressions")
    }

    fn elementwise_mul_const_lin(&self, c: &DMatrix<f64>, lin: &LinExpr) -> LinExpr {
        // Element-wise multiplication: diag(c) lin
        // For flat representation, this scales each row of coefficients by corresponding c value
        let c_flat: Vec<f64> = c.iter().copied().collect();
        let size = lin.shape.size();
        assert_eq!(
            c_flat.len(),
            size,
            "elementwise multiplication requires matching sizes after broadcasting"
        );

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &lin.coeffs {
            let coeff_dense = csc_to_dense(coeff);
            assert_eq!(
                coeff_dense.nrows(),
                size,
                "linear coefficient rows must match expression size"
            );
            let mut new_coeff = DMatrix::zeros(size, coeff_dense.ncols());
            for i in 0..size {
                for j in 0..coeff_dense.ncols() {
                    new_coeff[(i, j)] = c_flat[i] * coeff_dense[(i, j)];
                }
            }
            new_coeffs.insert(*var_id, dense_to_csc(&new_coeff));
        }

        let c_shaped = DMatrix::from_vec(lin.shape.rows(), lin.shape.cols(), c_flat);
        let new_const = c_shaped.component_mul(&lin.constant);

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: lin.shape.clone(),
        }
    }

    pub(super) fn canonicalize_matmul(&mut self, a: &Expr, b: &Expr) -> CanonExpr {
        let a_is_const = a.variables().is_empty() || a.curvature().is_constant();
        let b_is_const = b.variables().is_empty() || b.curvature().is_constant();
        let result_shape = a.shape().matmul(&b.shape()).unwrap_or_else(|| {
            panic!(
                "cannot matrix-multiply shapes {} and {}",
                a.shape(),
                b.shape()
            )
        });

        if a_is_const && !b_is_const {
            // A is constant expression, B has variables: A B is affine in B
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let a_arr = Array::Dense(ca.constant);
            return CanonExpr::Linear(self.matmul_const_lin(&a_arr, &cb, result_shape));
        }
        if b_is_const && !a_is_const {
            // B is constant expression, A has variables: A B is affine in A
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let b_arr = Array::Dense(cb.constant);
            return CanonExpr::Linear(self.lin_matmul_const(&ca, &b_arr, result_shape));
        }
        if a_is_const && b_is_const {
            // Both constant - evaluate and return constant
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            let result = &ca.constant * &cb.constant;
            return CanonExpr::Linear(LinExpr {
                coeffs: std::collections::HashMap::new(),
                constant: result,
                shape: result_shape,
            });
        }
        panic!("cannot canonicalize matrix product of two non-constant expressions")
    }

    fn matmul_const_lin(&self, a: &Array, b: &LinExpr, shape: Shape) -> LinExpr {
        // For matrix expression A E where E has shape (m, n):
        // vec(A E) = kron(I_n, A) vec(E), where kron is the Kronecker product.
        // Transform each coefficient matrix C to kron(I_n, A) C.
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
            // kron(I_n, A) c = vec(A reshape(c, m, n)) for each column c
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

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape,
        }
    }

    fn lin_matmul_const(&self, a: &LinExpr, b: &Array, shape: Shape) -> LinExpr {
        // For matrix expression E B where E has shape (m, n):
        // vec(E B) = kron(B^T, I_m) vec(E).
        // Transform each coefficient matrix C to kron(B^T, I_m) C.
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
            // kron(B^T, I_m) c = vec(reshape(c, m, n) B) for each column c
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

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape,
        }
    }

    pub(super) fn canonicalize_sum(&mut self, a: &Expr, axis: Option<usize>) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        if let Some(axis) = axis {
            return self.canonicalize_sum_axis_lin(&ca, axis);
        }

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

    fn canonicalize_sum_axis_lin(&mut self, x: &LinExpr, axis: usize) -> CanonExpr {
        if x.shape.ndim() <= 1 {
            assert_eq!(axis, 0, "axis {} out of bounds for shape {}", axis, x.shape);
            return self.canonicalize_sum_lin(x);
        }

        let rows = x.shape.rows();
        let cols = x.shape.cols();
        let (out_size, mut s_rows, mut s_cols, mut s_vals) = match axis {
            0 => (cols, Vec::new(), Vec::new(), Vec::new()),
            1 => (rows, Vec::new(), Vec::new(), Vec::new()),
            _ => panic!("axis {} out of bounds for shape {}", axis, x.shape),
        };

        for col in 0..cols {
            for row in 0..rows {
                let input_idx = row + col * rows;
                let output_idx = if axis == 0 { col } else { row };
                s_rows.push(output_idx);
                s_cols.push(input_idx);
                s_vals.push(1.0);
            }
        }

        let selector =
            crate::sparse::triplets_to_csc(out_size, x.size(), &s_rows, &s_cols, &s_vals);

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &x.coeffs {
            new_coeffs.insert(*var_id, crate::sparse::csc_matmul(&selector, coeff));
        }

        let flat_const = x
            .constant
            .clone()
            .reshape_generic(nalgebra::Dyn(x.size()), nalgebra::Dyn(1));
        let new_const = csc_to_dense(&selector) * flat_const;

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::vector(out_size),
        })
    }

    pub(super) fn canonicalize_trace(&mut self, a: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_sum_lin(&mut self, x: &LinExpr) -> CanonExpr {
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

    pub(super) fn canonicalize_cumsum(&mut self, x: &Expr, _axis: Option<usize>) -> CanonExpr {
        // Cumulative sum using auxiliary variables and constraints
        // cumsum([x_1, x_2, x_3]) = [y_1, y_2, y_3] where:
        //   y_1 = x_1
        //   y_2 = y_1 + x_2  =>  y_2 - y_1 = x_2
        //   y_3 = y_2 + x_3  =>  y_3 - y_2 = x_3

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

                // Constrain `y0` to equal `x0`.
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
}

fn dense_sparse_matmul(dense: &DMatrix<f64>, sparse: &CscMatrix<f64>) -> CscMatrix<f64> {
    // Multiply a dense matrix by a sparse matrix.
    // nalgebra_sparse does not support this product directly.
    // A more efficient implementation would iterate through sparse columns.
    // For medium-scale problems (100-10k variables), this is acceptable.
    let result = dense * csc_to_dense(sparse);
    dense_to_csc(&result)
}

fn scalar_constant_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Constant(c) => c.value.as_scalar(),
        Expr::Promote(a, _) => scalar_constant_value(a),
        _ => None,
    }
}
