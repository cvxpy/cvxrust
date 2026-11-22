//! Solver-independent solution values and status.

use std::collections::HashMap;
use std::ops::Range;

use crate::expr::{Array, Evaluable, ExprId};

/// Solution status from the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// Optimal solution found.
    Optimal,
    /// Problem is infeasible.
    Infeasible,
    /// Problem is unbounded.
    Unbounded,
    /// Maximum iterations reached.
    MaxIterations,
    /// Numerical difficulties.
    NumericalError,
    /// Unknown status.
    Unknown,
}

/// Solution from the solver.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution status.
    pub status: SolveStatus,
    /// Optimal value (if solved).
    pub value: Option<f64>,
    /// Primal variable values (if solved).
    pub primal: Option<HashMap<ExprId, Array>>,
    /// Dual variable values in solver cone order (if solved), including implicit bounds.
    pub dual: Option<Vec<f64>>,
    /// Solver dual row ranges for explicit constraints in problem order.
    /// Set by `Problem::solve`; `None` for low-level `solve`, which uses raw row indices.
    pub constraint_rows: Option<Vec<Range<usize>>>,
    /// Solve time in seconds.
    pub solve_time: f64,
    /// Number of iterations.
    pub iterations: u32,
}

impl Solution {
    /// Get the value of a variable.
    pub fn get_value(&self, var_id: ExprId) -> Option<&Array> {
        self.primal.as_ref().and_then(|p| p.get(&var_id))
    }

    /// Get scalar value for a variable. Panics if variable is not found or not scalar.
    ///
    /// # Example
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(());
    /// let solution = Problem::minimize(x.clone())
    ///     .subject_to([x.ge(1.0)])
    ///     .solve()
    ///     .unwrap();
    ///
    /// let x_val = solution.value(&x);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the expression is not a variable, the variable is not in the
    /// solution, or the variable is not scalar. Use `try_value()` for explicit
    /// error handling, or index operator `solution[&x]` for vectors/matrices.
    pub fn value(&self, var: &crate::expr::Expr) -> f64 {
        self.try_value(var).expect("failed to get scalar value")
    }

    /// Get scalar value for a variable, returning an error on failure.
    ///
    /// This is the fallible version of `value()`. Use this when you need
    /// explicit error handling.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The expression is not a variable
    /// - The variable is not in the solution
    /// - The variable is not scalar (use index operator for vectors/matrices)
    pub fn try_value(&self, var: &crate::expr::Expr) -> crate::Result<f64> {
        let var_id = var.variable_id().ok_or_else(|| {
            crate::CvxError::InvalidProblem("Expression is not a variable".into())
        })?;
        let arr = self
            .get_value(var_id)
            .ok_or_else(|| crate::CvxError::InvalidProblem("Variable not in solution".into()))?;
        match arr {
            Array::Scalar(v) => Ok(*v),
            Array::Dense(m) if m.nrows() == 1 && m.ncols() == 1 => Ok(m[(0, 0)]),
            _ => Err(crate::CvxError::InvalidProblem(
                "Variable is not scalar; use index operator for vectors/matrices".into(),
            )),
        }
    }

    /// Get all dual variable values (Lagrange multipliers).
    ///
    /// Includes implicit variable bounds and auxiliary constraints, in solver order:
    /// 1. Zero cone (equality constraints)
    /// 2. NonNeg cone (inequality constraints)
    /// 3. SOC constraints
    /// 4. Exponential cone constraints (3 duals per cone)
    /// 5. Power cone constraints (3 duals per cone)
    ///
    /// Returns `None` if the problem was not solved.
    ///
    /// # Example
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(2);
    /// let solution = Problem::minimize(sum(&x))
    ///     .subject_to([x.ge(constant(1.0))])
    ///     .solve()
    ///     .unwrap();
    ///
    /// if let Some(duals) = solution.duals() {
    ///     println!("Dual values: {:?}", duals);
    /// }
    /// ```
    pub fn duals(&self) -> Option<&[f64]> {
        self.dual.as_deref()
    }

    /// Get the dual value for a specific constraint by index.
    ///
    /// The constraint index corresponds to the order constraints were added to the problem.
    /// For vector or matrix constraints, returns the first component in column-major order.
    /// For SOC constraints, returns the component corresponding to `t`.
    /// Implicit bounds and auxiliary constraints do not count toward this index.
    /// Solutions from the low-level [`solve`](super::clarabel::solve) function use raw solver row indices.
    /// Returns `None` if the problem was not solved, the constraint has no components,
    /// or the index is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(2);
    /// let solution = Problem::minimize(sum(&x))
    ///     .subject_to([
    ///         x.ge(constant(1.0)),  // Constraint 0
    ///     ])
    ///     .solve()
    ///     .unwrap();
    ///
    /// if let Some(dual_0) = solution.constraint_dual(0) {
    ///     println!("Shadow price of first constraint: {}", dual_0);
    /// }
    /// ```
    pub fn constraint_dual(&self, idx: usize) -> Option<f64> {
        let dual = self.dual.as_ref()?;
        match &self.constraint_rows {
            Some(rows) => dual.get(rows.get(idx)?.clone())?.first().copied(),
            None => dual.get(idx).copied(),
        }
    }

    /// Check if the solution has dual values available.
    pub fn has_duals(&self) -> bool {
        self.dual.is_some()
    }
}

impl std::ops::Index<&crate::expr::Expr> for Solution {
    type Output = nalgebra::DMatrix<f64>;

    /// Get matrix/vector value for a variable using index operator.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or is a scalar (use `.value()` for scalars).
    ///
    /// # Example
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(5);
    /// let solution = Problem::minimize(sum(&x))
    ///     .subject_to([x.ge(1.0)])
    ///     .solve()
    ///     .unwrap();
    ///
    /// let x_vals = &solution[&x];
    /// println!("x[0] = {}", x_vals[(0, 0)]);
    /// ```
    fn index(&self, var: &crate::expr::Expr) -> &nalgebra::DMatrix<f64> {
        let var_id = var.variable_id().expect("Expression is not a variable");
        match self.get_value(var_id).expect("Variable not in solution") {
            Array::Dense(m) => m,
            Array::Scalar(_) => {
                panic!("Variable is scalar, use .value() method instead of indexing")
            }
            Array::Sparse(_) => unreachable!("Solution values are never sparse"),
        }
    }
}

impl Evaluable for Solution {
    fn get_variable_value(&self, id: ExprId) -> Option<&Array> {
        self.primal.as_ref()?.get(&id)
    }
}
