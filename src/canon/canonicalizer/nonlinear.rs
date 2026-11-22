//! Nonlinear atoms expressed with auxiliary variables and cone constraints.

use std::sync::Arc;

use nalgebra::DMatrix;

use super::{CanonContext, CanonExpr, ConeConstraint};
use crate::atoms::broadcast::broadcast_elementwise_exprs;
use crate::canon::lin_expr::LinExpr;
use crate::expr::{Expr, Shape};

impl CanonContext<'_> {
    pub(super) fn canonicalize_norm1(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_1 = sum(|x_i|)
        // Introduce t_i >= 0, -t_i <= x_i <= t_i
        // Then ||x||_1 = sum(t_i)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let cx = self.flatten_lin(&cx);
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

    pub(super) fn canonicalize_norm2(&mut self, x: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_norm_inf(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_inf = max(|x_i|)
        // Introduce t >= 0, -t <= x_i <= t for all i
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let cx = self.flatten_lin(&cx);
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

    pub(super) fn canonicalize_abs(&mut self, x: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_pos(&mut self, x: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_neg_part(&mut self, x: &Expr) -> CanonExpr {
        // neg(x) = max(-x, 0) = pos(-x)
        let neg_x = Expr::Neg(Arc::new(x.clone()));
        self.canonicalize_pos(&neg_x)
    }

    pub(super) fn canonicalize_maximum(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // max(x1, ..., xn): Introduce t, t >= x_i for all i
        if exprs.is_empty() {
            panic!("maximum requires at least one expression");
        }
        let first_shape = exprs[0].shape();
        let all_same_shape = exprs.iter().all(|expr| expr.shape() == first_shape);
        let shape = if all_same_shape {
            first_shape
        } else {
            exprs
                .iter()
                .map(|expr| expr.shape())
                .reduce(|acc, shape| {
                    acc.broadcast(&shape)
                        .unwrap_or_else(|| panic!("cannot broadcast shapes {} and {}", acc, shape))
                })
                .expect("maximum requires at least one expression")
        };
        let (_, t) = self.new_aux_var(shape);

        if all_same_shape {
            for e in exprs {
                let ce = self
                    .canonicalize_expr(e.as_ref(), false)
                    .as_linear()
                    .clone();
                // t >= x_i, i.e., t - x_i >= 0
                self.constraints.push(ConeConstraint::NonNeg {
                    a: t.add(&ce.neg()),
                });
            }
            return CanonExpr::Linear(t);
        }

        let (_, exprs) =
            broadcast_elementwise_exprs(exprs.iter().map(|expr| expr.as_ref().clone()));
        for e in &exprs {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // t >= x_i, i.e., t - x_i >= 0
            self.constraints.push(ConeConstraint::NonNeg {
                a: t.add(&ce.neg()),
            });
        }

        CanonExpr::Linear(t)
    }

    pub(super) fn canonicalize_minimum(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // min(x1, ..., xn): Introduce t, t <= x_i for all i
        if exprs.is_empty() {
            panic!("minimum requires at least one expression");
        }
        let first_shape = exprs[0].shape();
        let all_same_shape = exprs.iter().all(|expr| expr.shape() == first_shape);
        let shape = if all_same_shape {
            first_shape
        } else {
            exprs
                .iter()
                .map(|expr| expr.shape())
                .reduce(|acc, shape| {
                    acc.broadcast(&shape)
                        .unwrap_or_else(|| panic!("cannot broadcast shapes {} and {}", acc, shape))
                })
                .expect("minimum requires at least one expression")
        };
        let (_, t) = self.new_aux_var(shape);

        if all_same_shape {
            for e in exprs {
                let ce = self
                    .canonicalize_expr(e.as_ref(), false)
                    .as_linear()
                    .clone();
                // t <= x_i, i.e., x_i - t >= 0
                self.constraints.push(ConeConstraint::NonNeg {
                    a: ce.add(&t.neg()),
                });
            }
            return CanonExpr::Linear(t);
        }

        let (_, exprs) =
            broadcast_elementwise_exprs(exprs.iter().map(|expr| expr.as_ref().clone()));
        for e in &exprs {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // t <= x_i, i.e., x_i - t >= 0
            self.constraints.push(ConeConstraint::NonNeg {
                a: ce.add(&t.neg()),
            });
        }

        CanonExpr::Linear(t)
    }

    pub(super) fn canonicalize_exp(&mut self, x: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_log(&mut self, x: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_entropy(&mut self, x: &Expr) -> CanonExpr {
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

    pub(super) fn canonicalize_power(&mut self, x: &Expr, p: f64) -> CanonExpr {
        // x^p using power cones
        // Each element requires its own power cone constraint
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        if (p - 1.0).abs() < 1e-10 {
            // x^1 = x (affine)
            return CanonExpr::Linear(cx);
        }

        if p.abs() < 1e-10 {
            // x^0 = 1 elementwise.
            return CanonExpr::Linear(LinExpr::constant(DMatrix::from_element(
                cx.shape.rows(),
                cx.shape.cols(),
                1.0,
            )));
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
}
