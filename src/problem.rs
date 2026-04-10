//! Problem definition and solving API.
//!
//! The `Problem` struct represents an optimization problem with:
//! - An objective (minimize or maximize)
//! - A set of constraints
//!
//! Use the builder pattern to construct problems:
//! ```ignore
//! let solution = Problem::minimize(objective)
//!     .subject_to([constraint1, constraint2])
//!     .solve()?;
//! ```

use std::sync::Arc;

use crate::canon::{ConeConstraint, canonicalize};
use crate::constraints::Constraint;
use crate::error::{CvxError, Result};
use crate::expr::{Expr, ExprId, Shape};
use crate::solver::{Settings, Solution, SolveStatus, solve, stuff_problem};

/// Objective type for optimization problems.
#[derive(Debug, Clone)]
pub enum Objective {
    /// Minimize the expression.
    Minimize(Expr),
    /// Maximize the expression (internally converted to minimization).
    Maximize(Expr),
}

impl Objective {
    /// Get the expression being optimized.
    pub fn expr(&self) -> &Expr {
        match self {
            Objective::Minimize(e) | Objective::Maximize(e) => e,
        }
    }

    /// Check if this is a minimization.
    pub fn is_minimize(&self) -> bool {
        matches!(self, Objective::Minimize(_))
    }
}

/// An optimization problem.
#[derive(Debug, Clone)]
pub struct Problem {
    /// The objective to optimize.
    pub objective: Objective,
    /// The constraints.
    pub constraints: Vec<Constraint>,
}

impl Problem {
    /// Create a minimization problem.
    pub fn minimize(expr: Expr) -> ProblemBuilder {
        ProblemBuilder {
            objective: Objective::Minimize(expr),
            constraints: Vec::new(),
        }
    }

    /// Create a maximization problem.
    pub fn maximize(expr: Expr) -> ProblemBuilder {
        ProblemBuilder {
            objective: Objective::Maximize(expr),
            constraints: Vec::new(),
        }
    }

    /// Check if this problem is DCP-compliant.
    ///
    /// A problem is DCP if:
    /// - Minimize: objective is convex
    /// - Maximize: objective is concave
    /// - All constraints are DCP
    pub fn is_dcp(&self) -> bool {
        let obj_valid = match &self.objective {
            Objective::Minimize(e) => e.is_convex(),
            Objective::Maximize(e) => e.is_concave(),
        };

        obj_valid && self.constraints.iter().all(|c| c.is_dcp())
    }

    /// Get all variable IDs in this problem.
    pub fn variables(&self) -> Vec<ExprId> {
        let mut vars = self.objective.expr().variables();
        for c in &self.constraints {
            vars.extend(c.variables());
        }
        vars.sort_by_key(|id| id.raw());
        vars.dedup();
        vars
    }

    /// Get all variables with their shapes.
    pub fn variable_shapes(&self) -> Vec<(ExprId, Shape)> {
        let mut var_shapes: std::collections::HashMap<ExprId, Shape> =
            std::collections::HashMap::new();

        // Collect from objective
        Self::collect_variable_shapes(self.objective.expr(), &mut var_shapes);

        // Collect from constraints
        for c in &self.constraints {
            for expr in c.expressions() {
                Self::collect_variable_shapes(expr, &mut var_shapes);
            }
        }

        let mut result: Vec<_> = var_shapes.into_iter().collect();
        result.sort_by_key(|(id, _)| id.raw());
        result
    }

    /// Recursively collect variable shapes from an expression.
    fn collect_variable_shapes(expr: &Expr, shapes: &mut std::collections::HashMap<ExprId, Shape>) {
        match expr {
            Expr::Variable(v) => {
                shapes.insert(v.id, v.shape.clone());
            }
            Expr::Constant(_) => {}
            Expr::Add(a, b) | Expr::Mul(a, b) | Expr::MatMul(a, b) => {
                Self::collect_variable_shapes(a, shapes);
                Self::collect_variable_shapes(b, shapes);
            }
            Expr::Neg(a)
            | Expr::Sum(a, _)
            | Expr::Reshape(a, _)
            | Expr::Index(a, _)
            | Expr::Transpose(a)
            | Expr::Trace(a)
            | Expr::Norm1(a)
            | Expr::Norm2(a)
            | Expr::NormInf(a)
            | Expr::Abs(a)
            | Expr::Pos(a)
            | Expr::NegPart(a)
            | Expr::SumSquares(a)
            | Expr::Exp(a)
            | Expr::Log(a)
            | Expr::Entropy(a)
            | Expr::Power(a, _)
            | Expr::Cumsum(a, _)
            | Expr::Diag(a) => {
                Self::collect_variable_shapes(a, shapes);
            }
            Expr::VStack(exprs)
            | Expr::HStack(exprs)
            | Expr::Maximum(exprs)
            | Expr::Minimum(exprs) => {
                for e in exprs {
                    Self::collect_variable_shapes(e, shapes);
                }
            }
            Expr::QuadForm(a, b) | Expr::QuadOverLin(a, b) => {
                Self::collect_variable_shapes(a, shapes);
                Self::collect_variable_shapes(b, shapes);
            }
        }
    }

    /// Solve the problem with default settings.
    pub fn solve(&self) -> Result<Solution> {
        self.solve_with(Settings::default())
    }

    /// Solve the problem with custom settings.
    pub fn solve_with(&self, settings: Settings) -> Result<Solution> {
        // Check DCP compliance
        if !self.is_dcp() {
            return Err(CvxError::NotDcp(self.dcp_violation_message()));
        }

        // Convert maximize to minimize
        let (obj_expr, negate_result) = match &self.objective {
            Objective::Minimize(e) => (e.clone(), false),
            Objective::Maximize(e) => (Expr::Neg(Arc::new(e.clone())), true),
        };

        // Canonicalize objective
        let obj_canon = canonicalize(&obj_expr, true);
        let obj_quad = obj_canon.expr.into_quadratic();

        // Collect all variables (original + auxiliary) with their shapes
        let mut all_vars: Vec<(ExprId, Shape)> = self.variable_shapes().into_iter().collect();

        // Add auxiliary variables from objective canonicalization
        all_vars.extend(obj_canon.aux_vars);

        // Canonicalize constraints
        let mut all_cone_constraints: Vec<ConeConstraint> = obj_canon.constraints;

        for constraint in &self.constraints {
            let canon_result = canonicalize_constraint(constraint);
            // Add the cone constraints
            all_cone_constraints.extend(canon_result.cone_constraints);
            // Add auxiliary variables from constraint canonicalization
            all_vars.extend(canon_result.aux_vars);
        }

        // Stuff the problem
        let stuffed = stuff_problem(&obj_quad, &all_cone_constraints, &all_vars);

        // Solve
        let mut solution = solve(&stuffed, &settings);

        // Adjust for maximization
        if negate_result {
            solution.value = solution.value.map(|v| -v);
        }

        // Check solution status
        match solution.status {
            SolveStatus::Optimal => Ok(solution),
            SolveStatus::Infeasible => Err(CvxError::SolverError("Problem is infeasible".into())),
            SolveStatus::Unbounded => Err(CvxError::SolverError("Problem is unbounded".into())),
            SolveStatus::MaxIterations => {
                Err(CvxError::SolverError("Maximum iterations reached".into()))
            }
            SolveStatus::NumericalError => Err(CvxError::NumericalError(
                "Solver encountered numerical difficulties".into(),
            )),
            SolveStatus::Unknown => Err(CvxError::SolverError("Unknown solver status".into())),
        }
    }

    /// Get a message describing why the problem is not DCP.
    fn dcp_violation_message(&self) -> String {
        let mut violations = Vec::new();

        match &self.objective {
            Objective::Minimize(e) if !e.is_convex() => {
                violations.push(format!(
                    "Objective has curvature {:?} but must be convex for minimization",
                    e.curvature()
                ));
            }
            Objective::Maximize(e) if !e.is_concave() => {
                violations.push(format!(
                    "Objective has curvature {:?} but must be concave for maximization",
                    e.curvature()
                ));
            }
            _ => {}
        }

        for (i, c) in self.constraints.iter().enumerate() {
            if !c.is_dcp() {
                violations.push(format!("Constraint {} is not DCP", i));
            }
        }

        if violations.is_empty() {
            "Unknown DCP violation".into()
        } else {
            violations.join("; ")
        }
    }
}

/// Builder for constructing problems.
#[derive(Debug, Clone)]
pub struct ProblemBuilder {
    objective: Objective,
    constraints: Vec<Constraint>,
}

impl ProblemBuilder {
    /// Add constraints to the problem.
    pub fn subject_to(mut self, constraints: impl IntoIterator<Item = Constraint>) -> Self {
        self.constraints.extend(constraints);
        self
    }

    /// Add a single constraint.
    pub fn constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Build the problem.
    pub fn build(self) -> Problem {
        Problem {
            objective: self.objective,
            constraints: self.constraints,
        }
    }

    /// Build and solve the problem with default settings.
    pub fn solve(self) -> Result<Solution> {
        self.build().solve()
    }

    /// Build and solve the problem with custom settings.
    pub fn solve_with(self, settings: Settings) -> Result<Solution> {
        self.build().solve_with(settings)
    }
}

/// Result of canonicalizing a constraint.
struct ConstraintCanonResult {
    cone_constraints: Vec<ConeConstraint>,
    aux_vars: Vec<(ExprId, Shape)>,
}

/// Canonicalize a user constraint into cone constraints and auxiliary variables.
fn canonicalize_constraint(constraint: &Constraint) -> ConstraintCanonResult {
    match constraint {
        Constraint::Zero(expr) => {
            let canon = canonicalize(expr, false);
            let lin = canon.expr.as_linear().clone();
            let mut cone_constraints = vec![ConeConstraint::Zero { a: lin }];
            cone_constraints.extend(canon.constraints);
            ConstraintCanonResult {
                cone_constraints,
                aux_vars: canon.aux_vars,
            }
        }
        Constraint::NonNeg(expr) => {
            let canon = canonicalize(expr, false);
            let lin = canon.expr.as_linear().clone();
            let mut cone_constraints = vec![ConeConstraint::NonNeg { a: lin }];
            cone_constraints.extend(canon.constraints);
            ConstraintCanonResult {
                cone_constraints,
                aux_vars: canon.aux_vars,
            }
        }
        Constraint::SOC { t, x } => {
            let t_canon = canonicalize(t, false);
            let x_canon = canonicalize(x, false);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{norm2, sum};
    use crate::constraints::ConstraintExt;
    use crate::expr::{constant, variable};
    use crate::solver::SolveStatus;

    #[test]
    fn test_problem_builder() {
        let x = variable(5);
        let problem = Problem::minimize(sum(&x)).build();
        assert!(problem.is_dcp());
    }

    #[test]
    fn test_minimize_convex_is_dcp() {
        let x = variable(5);
        let problem = Problem::minimize(norm2(&x)).build();
        assert!(problem.is_dcp());
    }

    #[test]
    fn test_maximize_convex_not_dcp() {
        let x = variable(5);
        let problem = Problem::maximize(norm2(&x)).build();
        assert!(!problem.is_dcp());
    }

    #[test]
    fn test_minimize_concave_not_dcp() {
        let x = variable(5);
        let neg_norm = Expr::Neg(Arc::new(norm2(&x)));
        let problem = Problem::minimize(neg_norm).build();
        assert!(!problem.is_dcp());
    }

    #[test]
    fn test_maximize_concave_is_dcp() {
        let x = variable(5);
        let neg_norm = Expr::Neg(Arc::new(norm2(&x)));
        let problem = Problem::maximize(neg_norm).build();
        assert!(problem.is_dcp());
    }

    #[test]
    fn test_problem_with_constraints() {
        let x = variable(5);
        let c = constant(1.0);
        let problem = Problem::minimize(sum(&x)).subject_to([x.ge(c)]).build();
        assert!(problem.is_dcp());
    }

    #[test]
    fn test_solve_simple_lp() {
        // Minimize sum(x) subject to x >= 1
        // Optimal: x = [1, 1, 1, 1, 1], value = 5
        let x = variable(5);
        let one = constant(1.0);
        let result = Problem::minimize(sum(&x))
            .subject_to([x.ge(one)])
            .solve()
            .expect("solve failed");

        assert_eq!(result.status, SolveStatus::Optimal);
        let value = result.value.expect("no value");
        assert!((value - 5.0).abs() < 1e-4, "Expected ~5.0, got {}", value);
    }

    #[test]
    fn test_solve_norm2_minimization() {
        // Minimize ||x||_2 subject to sum(x) = 5
        // Optimal: x = [1, 1, 1, 1, 1], ||x||_2 = sqrt(5)
        let x = variable(5);
        let five = constant(5.0);
        let result = Problem::minimize(norm2(&x))
            .subject_to([sum(&x).eq(five)])
            .solve()
            .expect("solve failed");

        assert_eq!(result.status, SolveStatus::Optimal);
        let value = result.value.expect("no value");
        let expected = (5.0_f64).sqrt();
        assert!(
            (value - expected).abs() < 1e-3,
            "Expected ~{}, got {}",
            expected,
            value
        );
    }
}
