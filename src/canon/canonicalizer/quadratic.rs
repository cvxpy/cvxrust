/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Quadratic objectives and cone reformulations.

use nalgebra::DMatrix;

use super::{CanonContext, CanonExpr, ConeConstraint};
use crate::canon::lin_expr::{LinExpr, QuadExpr};
use crate::expr::{Array, Expr, Shape, symmetric_part};
use crate::sparse::{csc_add, csc_to_dense, dense_to_csc};

impl CanonContext<'_> {
    pub(super) fn canonicalize_quad_form(
        &mut self,
        x: &Expr,
        p: &Expr,
        for_objective: bool,
    ) -> CanonExpr {
        // Retain x's bounds when P = 0.
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let p_value = p
            .constant_value()
            .expect("quad_form requires a constant matrix");
        let p_mat = match p_value {
            Array::Scalar(value) => DMatrix::from_element(1, 1, *value),
            Array::Dense(matrix) => matrix.clone(),
            Array::Sparse(_) => panic!("quad_form with sparse P is not supported"),
        };
        assert!(
            cx.shape.cols() == 1 && p_mat.nrows() == cx.size() && p_mat.ncols() == cx.size(),
            "quad_form requires a vector x and a matching square matrix P"
        );
        if matches!(p_value, Array::Scalar(value) if *value == 0.0) {
            return CanonExpr::Linear(LinExpr::scalar(0.0));
        }

        // x^T P x = x^T ((P + P^T) / 2) x.
        let p_mat = symmetric_part(&p_mat);

        if for_objective {
            // x = sum_i A_i v_i + b gives Q_{ij} = A_i^T P A_j,
            // q_i = 2 A_i^T P b, and r = b^T P b.
            let coeffs: Vec<_> = cx
                .coeffs
                .iter()
                .map(|(&id, coeff)| (id, csc_to_dense(coeff)))
                .collect();
            let pb = &p_mat * &cx.constant;
            let mut quad_coeffs = std::collections::HashMap::new();
            let mut linear_coeffs = std::collections::HashMap::new();
            for (var_i, ai) in &coeffs {
                let ai_t_p = ai.transpose() * &p_mat;
                for (var_j, aj) in &coeffs {
                    quad_coeffs.insert((*var_i, *var_j), dense_to_csc(&(&ai_t_p * aj)));
                }
                linear_coeffs.insert(
                    *var_i,
                    dense_to_csc(&(ai.transpose() * &pb * 2.0).transpose()),
                );
            }
            return CanonExpr::Quadratic(QuadExpr {
                quad_coeffs,
                linear: LinExpr {
                    coeffs: linear_coeffs,
                    constant: DMatrix::zeros(1, 1),
                    shape: Shape::scalar(),
                },
                constant: cx.constant.dot(&pb),
            });
        }

        // P = sign * R^T R: use t >= ||R x||_2^2 and return sign * t.
        let (sign, factor) = match p_value {
            Array::Scalar(value) => (
                value.signum(),
                DMatrix::from_element(1, 1, value.abs().sqrt()),
            ),
            _ => match p_mat.clone().cholesky() {
                Some(cholesky) => (1.0, cholesky.l().transpose()),
                None => (
                    -1.0,
                    (-p_mat)
                        .cholesky()
                        .expect("quad_form requires a supported PSD or NSD matrix")
                        .l()
                        .transpose(),
                ),
            },
        };
        let weighted_x = LinExpr {
            coeffs: cx
                .coeffs
                .iter()
                .map(|(&id, coeff)| (id, dense_to_csc(&(&factor * csc_to_dense(coeff)))))
                .collect(),
            constant: &factor * &cx.constant,
            shape: Shape::vector(cx.size()),
        };
        let epigraph = self.canonicalize_sum_squares_lin(&weighted_x, false);
        CanonExpr::Linear(epigraph.as_linear().scale(sign))
    }

    pub(super) fn canonicalize_sum_squares(&mut self, x: &Expr, for_objective: bool) -> CanonExpr {
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let cx = self.flatten_lin(&cx);
        self.canonicalize_sum_squares_lin(&cx, for_objective)
    }

    fn canonicalize_sum_squares_lin(&mut self, x: &LinExpr, for_objective: bool) -> CanonExpr {
        if for_objective {
            // ||A x + c||_2^2 = x^T A^T A x + 2 c^T A x + ||c||_2^2.
            let vars = x.variables();
            let c = &x.constant;

            let mut quad_coeffs = std::collections::HashMap::new();
            let mut linear_coeffs = std::collections::HashMap::new();

            for &var_i in &vars {
                let ai = csc_to_dense(&x.coeffs[&var_i]);
                for &var_j in &vars {
                    let aj = csc_to_dense(&x.coeffs[&var_j]);
                    let ai_t_aj = dense_to_csc(&(ai.transpose() * &aj));
                    quad_coeffs
                        .entry((var_i, var_j))
                        .and_modify(|existing| *existing = csc_add(existing, &ai_t_aj))
                        .or_insert(ai_t_aj);
                }
                let q_col = ai.transpose() * c;
                let q_row = dense_to_csc(&(q_col * 2.0).transpose());
                linear_coeffs.insert(var_i, q_row);
            }

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

        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());
        let one = LinExpr::scalar(1.0);
        let soc_t = t.add(&one);
        let soc_x = self.vstack_lin(&t.add(&one.neg()), &x.scale(2.0));
        self.constraints
            .push(ConeConstraint::SOC { t: soc_t, x: soc_x });

        CanonExpr::Linear(t)
    }

    pub(super) fn canonicalize_quad_over_lin(&mut self, x: &Expr, y: &Expr) -> CanonExpr {
        // ||x||_2^2 / y: introduce t with ||[2 x; t - y]||_2 <= t + y.
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let cx = self.flatten_lin(&cx);
        let cy = self.canonicalize_expr(y, false).as_linear().clone();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        let soc_t = t.add(&cy);
        let soc_x = self.vstack_lin(&t.add(&cy.neg()), &cx.scale(2.0));
        self.constraints
            .push(ConeConstraint::SOC { t: soc_t, x: soc_x });

        CanonExpr::Linear(t)
    }
}
