//! Lower user constraints into cones while preserving explicit constraint order.

use super::{ConeConstraint, VariableDomains, canonicalize};
use crate::constraints::Constraint;
use crate::expr::{ExprId, Shape};

/// Result of canonicalizing a constraint.
pub(crate) struct ConstraintCanonResult {
    /// The user constraint's cone followed by any auxiliary cones.
    pub(crate) cone_constraints: Vec<ConeConstraint>,
    pub(crate) aux_vars: Vec<(ExprId, Shape)>,
}

/// Canonicalize a user constraint into cone constraints and auxiliary variables.
pub(crate) fn canonicalize_constraint(
    constraint: &Constraint,
    domains: &mut VariableDomains,
) -> ConstraintCanonResult {
    match constraint {
        Constraint::Zero(expr) => {
            let canon = canonicalize(expr, false, domains);
            let lin = canon.expr.as_linear().clone();
            let mut cone_constraints = vec![ConeConstraint::Zero { a: lin }];
            cone_constraints.extend(canon.constraints);
            ConstraintCanonResult {
                cone_constraints,
                aux_vars: canon.aux_vars,
            }
        }
        Constraint::NonNeg(expr) => {
            let canon = canonicalize(expr, false, domains);
            let lin = canon.expr.as_linear().clone();
            let mut cone_constraints = vec![ConeConstraint::NonNeg { a: lin }];
            cone_constraints.extend(canon.constraints);
            ConstraintCanonResult {
                cone_constraints,
                aux_vars: canon.aux_vars,
            }
        }
        Constraint::SOC { t, x } => {
            let t_canon = canonicalize(t, false, domains);
            let x_canon = canonicalize(x, false, domains);
            let t_lin = t_canon.expr.as_linear().clone();
            let x_lin = x_canon.expr.as_linear().clone();
            let mut cone_constraints = vec![ConeConstraint::SOC { t: t_lin, x: x_lin }];
            cone_constraints.extend(t_canon.constraints);
            cone_constraints.extend(x_canon.constraints);
            let mut aux_vars = t_canon.aux_vars;
            aux_vars.extend(x_canon.aux_vars);
            ConstraintCanonResult {
                cone_constraints,
                aux_vars,
            }
        }
    }
}
