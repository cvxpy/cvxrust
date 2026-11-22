/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

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

use crate::canon::constraints::canonicalize_constraint;
use crate::canon::{ConeConstraint, VariableDomains, canonicalize};
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

    /// Collect variable shapes in expression order, retaining the last occurrence.
    fn collect_variable_shapes(expr: &Expr, shapes: &mut std::collections::HashMap<ExprId, Shape>) {
        expr.visit_variables(&mut |variable| {
            shapes.insert(variable.id, variable.shape.clone());
        });
    }

    /// Solve the problem with default settings.
    ///
    /// Variable sign declarations are enforced as described in [`Self::solve_with`].
    pub fn solve(&self) -> Result<Solution> {
        self.solve_with(Settings::default())
    }

    /// Solve the problem with custom settings.
    ///
    /// Sign declarations on variables referenced by the objective or constraints
    /// are enforced elementwise. Clones with the same variable ID contribute all
    /// their declared bounds: an unrestricted clone does not remove a bound, and
    /// opposite signs require the variable to be zero. Unused clones have no effect.
    /// Bound tracking is independent for each solve.
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
        let mut domains = VariableDomains::default();
        let obj_canon = canonicalize(&obj_expr, true, &mut domains);
        let obj_quad = obj_canon.expr.into_quadratic();

        // Collect all variables (original + auxiliary) with their shapes
        let mut all_vars: Vec<(ExprId, Shape)> = self.variable_shapes().into_iter().collect();

        // Add auxiliary variables from objective canonicalization
        all_vars.extend(obj_canon.aux_vars);

        // Canonicalize constraints
        let mut all_cone_constraints: Vec<ConeConstraint> = obj_canon.constraints;
        let mut constraint_indices = Vec::with_capacity(self.constraints.len());

        for constraint in &self.constraints {
            let canon_result = canonicalize_constraint(constraint, &mut domains);
            // The first cone represents the user constraint; the rest are auxiliary.
            constraint_indices.push(all_cone_constraints.len());
            // Add the cone constraints
            all_cone_constraints.extend(canon_result.cone_constraints);
            // Add auxiliary variables from constraint canonicalization
            all_vars.extend(canon_result.aux_vars);
        }

        // Stuff the problem
        let stuffed = stuff_problem(&obj_quad, &all_cone_constraints, &all_vars);

        // Solve
        let mut solution = solve(&stuffed, &settings);
        solution.constraint_rows = Some(
            constraint_indices
                .into_iter()
                .map(|idx| stuffed.constraint_rows[idx].clone())
                .collect(),
        );

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{norm2, sum};
    use crate::constraints::ConstraintExt;
    use crate::expr::{VariableData, constant, variable};
    use crate::solver::SolveStatus;

    #[test]
    fn test_problem_builder() {
        let x = variable(5);
        let problem = Problem::minimize(sum(&x)).build();
        assert!(problem.is_dcp());
    }

    #[test]
    fn test_variable_shapes_retain_last_occurrence_in_problem_order() {
        let Expr::Variable(x) = variable(2) else {
            unreachable!()
        };
        let y = variable(3);
        let x_id = x.id;
        let y_id = y.variable_id().unwrap();
        let with_shape = |shape| Expr::Variable(VariableData { shape, ..x.clone() });
        let objective = Expr::VStack(vec![
            Arc::new(y),
            Arc::new(with_shape(Shape::vector(4))),
            Arc::new(with_shape(Shape::vector(5))),
        ]);
        let mut problem = Problem::minimize(sum(&objective)).build();
        assert_eq!(
            problem.variable_shapes(),
            vec![(x_id, Shape::vector(5)), (y_id, Shape::vector(3))]
        );

        problem
            .constraints
            .push(Constraint::Zero(Arc::new(with_shape(Shape::vector(6)))));
        assert_eq!(
            problem.variable_shapes(),
            vec![(x_id, Shape::vector(6)), (y_id, Shape::vector(3))]
        );

        problem.constraints.push(Constraint::soc(
            with_shape(Shape::scalar()),
            with_shape(Shape::vector(7)),
        ));
        assert_eq!(
            problem.variable_shapes(),
            vec![(x_id, Shape::vector(7)), (y_id, Shape::vector(3))]
        );
        assert_eq!(problem.variables(), vec![x_id, y_id]);
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
