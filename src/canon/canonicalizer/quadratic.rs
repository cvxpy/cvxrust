//! Quadratic objectives and cone reformulations.

use nalgebra::DMatrix;

use super::{CanonContext, CanonExpr, ConeConstraint};
use crate::canon::lin_expr::{LinExpr, QuadExpr};
use crate::expr::{Array, Expr, Shape};
use crate::sparse::{csc_add, csc_to_dense, dense_to_csc};

impl CanonContext<'_> {
    pub(super) fn canonicalize_quad_form(
        &mut self,
        x: &Expr,
        p: &Expr,
        for_objective: bool,
    ) -> CanonExpr {
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

    pub(super) fn canonicalize_sum_squares(&mut self, x: &Expr, for_objective: bool) -> CanonExpr {
        // ||x||_2^2 = x' x
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let cx = self.flatten_lin(&cx);
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

        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());
        let one = LinExpr::scalar(1.0);
        let soc_t = t.add(&one);
        let soc_x = self.vstack_lin(&t.add(&one.neg()), &x.scale(2.0));
        self.constraints
            .push(ConeConstraint::SOC { t: soc_t, x: soc_x });

        CanonExpr::Linear(t)
    }

    pub(super) fn canonicalize_quad_over_lin(&mut self, x: &Expr, y: &Expr) -> CanonExpr {
        // ||x||_2^2 / y: introduce t with ||[2x; t-y]||_2 <= t+y.
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
