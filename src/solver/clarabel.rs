/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Clarabel solver integration.
//!
//! This module provides the interface to the Clarabel conic solver.

use std::collections::HashMap;

use clarabel::algebra::CscMatrix as ClarabelCsc;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolution, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

use super::stuffing::{ConeDims, StuffedProblem, VariableMap};
use nalgebra::DMatrix;

use crate::expr::{Array, ExprId};

pub use super::solution::{Solution, SolveStatus};

impl From<SolverStatus> for SolveStatus {
    fn from(status: SolverStatus) -> Self {
        match status {
            SolverStatus::Solved => SolveStatus::Optimal,
            SolverStatus::AlmostSolved => SolveStatus::OptimalInaccurate,
            SolverStatus::PrimalInfeasible => SolveStatus::Infeasible,
            SolverStatus::AlmostPrimalInfeasible => SolveStatus::InfeasibleInaccurate,
            SolverStatus::DualInfeasible => SolveStatus::Unbounded,
            SolverStatus::AlmostDualInfeasible => SolveStatus::UnboundedInaccurate,
            SolverStatus::MaxIterations => SolveStatus::MaxIterations,
            SolverStatus::MaxTime => SolveStatus::MaxTime,
            SolverStatus::NumericalError => SolveStatus::NumericalError,
            SolverStatus::InsufficientProgress => SolveStatus::InsufficientProgress,
            SolverStatus::CallbackTerminated => SolveStatus::CallbackTerminated,
            SolverStatus::Unsolved => SolveStatus::Unsolved,
        }
    }
}

/// Solver settings.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Print solver output.
    pub verbose: bool,
    /// Maximum iterations.
    pub max_iter: u32,
    /// Time limit in seconds.
    pub time_limit: f64,
    /// Absolute tolerance.
    pub tol_gap_abs: f64,
    /// Relative tolerance.
    pub tol_gap_rel: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            verbose: false,
            max_iter: 100,
            time_limit: f64::INFINITY,
            tol_gap_abs: 1e-8,
            tol_gap_rel: 1e-8,
        }
    }
}

/// Solve the stuffed problem using Clarabel.
///
/// Returns every termination status, including failures; see [`Solution`].
pub fn solve(problem: &StuffedProblem, settings: &Settings) -> Solution {
    // Convert to Clarabel format
    let p = to_clarabel_csc(&problem.p);
    let a = to_clarabel_csc(&problem.a);
    let cones = to_clarabel_cones(&problem.cone_dims);

    // Build Clarabel settings
    let clarabel_settings = DefaultSettingsBuilder::default()
        .verbose(settings.verbose)
        .max_iter(settings.max_iter)
        .time_limit(settings.time_limit)
        .tol_gap_abs(settings.tol_gap_abs)
        .tol_gap_rel(settings.tol_gap_rel)
        .build()
        .unwrap();

    // Create and run solver
    let mut solver = DefaultSolver::new(&p, &problem.q, &a, &problem.b, &cones, clarabel_settings)
        .expect("Settings to be correct");
    solver.solve();

    extract_solution(problem, &solver.solution)
}

/// Recover values only when the backend result represents a primal/dual iterate.
fn extract_solution(problem: &StuffedProblem, result: &DefaultSolution<f64>) -> Solution {
    let status = result.status.into();
    let (value, primal, dual) = match status {
        SolveStatus::Optimal
        | SolveStatus::OptimalInaccurate
        | SolveStatus::MaxIterations
        | SolveStatus::MaxTime => (
            Some(compute_objective(&result.x, &problem.p, &problem.q) + problem.objective_offset),
            Some(unpack_primal(&result.x, &problem.var_map)),
            Some(result.z.clone()),
        ),
        SolveStatus::Infeasible | SolveStatus::InfeasibleInaccurate => {
            (Some(f64::INFINITY), None, None)
        }
        SolveStatus::Unbounded | SolveStatus::UnboundedInaccurate => {
            (Some(f64::NEG_INFINITY), None, None)
        }
        SolveStatus::NumericalError
        | SolveStatus::InsufficientProgress
        | SolveStatus::CallbackTerminated
        | SolveStatus::Unsolved
        | SolveStatus::Unknown => (None, None, None),
    };

    Solution {
        status,
        value,
        primal,
        dual,
        constraint_rows: None,
        solve_time: result.solve_time,
        iterations: result.iterations,
    }
}

/// Convert nalgebra CSC to Clarabel CSC.
fn to_clarabel_csc(m: &nalgebra_sparse::CscMatrix<f64>) -> ClarabelCsc<f64> {
    ClarabelCsc::new(
        m.nrows(),
        m.ncols(),
        m.col_offsets().to_vec(),
        m.row_indices().to_vec(),
        m.values().to_vec(),
    )
}

/// Convert cone dimensions to Clarabel cones.
fn to_clarabel_cones(dims: &ConeDims) -> Vec<SupportedConeT<f64>> {
    let mut cones = Vec::new();

    if dims.zero > 0 {
        cones.push(SupportedConeT::ZeroConeT(dims.zero));
    }

    if dims.nonneg > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(dims.nonneg));
    }

    for &soc_dim in &dims.soc {
        cones.push(SupportedConeT::SecondOrderConeT(soc_dim));
    }

    // Exponential cones (each is 3D)
    for _ in 0..dims.exp {
        cones.push(SupportedConeT::ExponentialConeT());
    }

    // Power cones (each is 3D with its own alpha)
    for &alpha in &dims.power {
        cones.push(SupportedConeT::PowerConeT(alpha));
    }

    cones
}

/// Unpack primal solution into variable values.
fn unpack_primal(x: &[f64], var_map: &VariableMap) -> HashMap<ExprId, Array> {
    let mut result = HashMap::new();

    for (&var_id, &(start, size)) in &var_map.id_to_col {
        let values: Vec<f64> = x[start..start + size].to_vec();
        let arr = if let Some(shape) = var_map.shape(var_id) {
            if shape.is_scalar() {
                Array::Scalar(values[0])
            } else {
                Array::Dense(DMatrix::from_vec(shape.rows(), shape.cols(), values))
            }
        } else if size == 1 {
            Array::Scalar(values[0])
        } else {
            Array::from_vec(values)
        };
        result.insert(var_id, arr);
    }

    result
}

/// Compute objective value: `(1/2) x^T P x + q^T x`.
fn compute_objective(x: &[f64], p: &nalgebra_sparse::CscMatrix<f64>, q: &[f64]) -> f64 {
    // q^T x
    let linear: f64 = q.iter().zip(x.iter()).map(|(qi, xi)| qi * xi).sum();

    // (1/2) x^T P x
    let mut quadratic = 0.0;
    for (row, col, val) in p.triplet_iter() {
        if row == col {
            quadratic += 0.5 * *val * x[row] * x[col];
        } else {
            // Off-diagonal (stored as upper triangle, so count once)
            quadratic += *val * x[row] * x[col];
        }
    }

    linear + quadratic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Shape;
    use crate::sparse::csc_from_triplets;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_solver_status_conversion() {
        for (backend, expected) in [
            (SolverStatus::Solved, SolveStatus::Optimal),
            (SolverStatus::AlmostSolved, SolveStatus::OptimalInaccurate),
            (SolverStatus::PrimalInfeasible, SolveStatus::Infeasible),
            (
                SolverStatus::AlmostPrimalInfeasible,
                SolveStatus::InfeasibleInaccurate,
            ),
            (SolverStatus::DualInfeasible, SolveStatus::Unbounded),
            (
                SolverStatus::AlmostDualInfeasible,
                SolveStatus::UnboundedInaccurate,
            ),
            (SolverStatus::MaxIterations, SolveStatus::MaxIterations),
            (SolverStatus::MaxTime, SolveStatus::MaxTime),
            (SolverStatus::NumericalError, SolveStatus::NumericalError),
            (
                SolverStatus::InsufficientProgress,
                SolveStatus::InsufficientProgress,
            ),
            (
                SolverStatus::CallbackTerminated,
                SolveStatus::CallbackTerminated,
            ),
            (SolverStatus::Unsolved, SolveStatus::Unsolved),
        ] {
            assert_eq!(SolveStatus::from(backend), expected, "{backend:?}");
        }
    }

    fn extraction_fixture() -> (StuffedProblem, DefaultSolution<f64>, ExprId, ExprId) {
        let scalar = ExprId::new();
        let matrix = ExprId::new();
        let problem = StuffedProblem {
            // Objective: `x_0^2 + x_0 x_1 + x_1^2 + 3 x_0 - x_4 + 7`.
            p: csc_from_triplets(5, 5, vec![0, 0, 1], vec![0, 1, 1], vec![2.0, 1.0, 2.0]),
            q: vec![3.0, 0.0, 0.0, 0.0, -1.0],
            a: nalgebra_sparse::CscMatrix::zeros(2, 5),
            b: vec![0.0; 2],
            cone_dims: ConeDims {
                nonneg: 2,
                ..ConeDims::default()
            },
            constraint_rows: vec![0..1, 1..2],
            var_map: VariableMap::from_vars(&[
                (scalar, Shape::scalar()),
                (matrix, Shape::matrix(2, 2)),
            ]),
            objective_offset: 7.0,
        };
        let mut result = DefaultSolution::new(5, 2);
        result.x = vec![2.0, 1.0, 3.0, 2.0, 4.0];
        result.z = vec![0.5, 1.5];
        result.solve_time = 0.25;
        result.iterations = 7;
        (problem, result, scalar, matrix)
    }

    #[test]
    fn test_extract_solution_retains_iterates_and_objective() {
        let (problem, mut result, scalar, matrix) = extraction_fixture();
        for status in [
            SolverStatus::Solved,
            SolverStatus::AlmostSolved,
            SolverStatus::MaxIterations,
            SolverStatus::MaxTime,
        ] {
            result.status = status;
            let solution = extract_solution(&problem, &result);
            assert_eq!(solution.status, status.into());
            assert_abs_diff_eq!(solution.value.unwrap(), 16.0, epsilon = 1e-12);
            assert_abs_diff_eq!(
                solution.get_value(scalar).unwrap().as_scalar().unwrap(),
                2.0,
                epsilon = 1e-12
            );
            let Array::Dense(values) = solution.get_value(matrix).unwrap() else {
                panic!("matrix variable should retain its shape");
            };
            assert_eq!(values.shape(), (2, 2));
            for (index, expected) in [((0, 0), 1.0), ((0, 1), 2.0), ((1, 0), 3.0), ((1, 1), 4.0)] {
                assert_abs_diff_eq!(values[index], expected, epsilon = 1e-12);
            }
            assert!(solution.constraint_rows.is_none());
            assert!(solution.has_duals());
            assert_abs_diff_eq!(solution.constraint_dual(0).unwrap(), 0.5, epsilon = 1e-12);
            assert_abs_diff_eq!(solution.constraint_dual(1).unwrap(), 1.5, epsilon = 1e-12);
            assert_eq!(solution.iterations, 7);
            assert_abs_diff_eq!(solution.solve_time, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_extract_solution_omits_certificates_and_failed_iterates() {
        let (problem, mut result, scalar, _) = extraction_fixture();
        for (status, expected_value) in [
            (SolverStatus::PrimalInfeasible, Some(f64::INFINITY)),
            (SolverStatus::AlmostPrimalInfeasible, Some(f64::INFINITY)),
            (SolverStatus::DualInfeasible, Some(f64::NEG_INFINITY)),
            (SolverStatus::AlmostDualInfeasible, Some(f64::NEG_INFINITY)),
            (SolverStatus::NumericalError, None),
            (SolverStatus::InsufficientProgress, None),
            (SolverStatus::CallbackTerminated, None),
            (SolverStatus::Unsolved, None),
        ] {
            result.status = status;
            let solution = extract_solution(&problem, &result);
            assert_eq!(solution.status, status.into());
            assert_eq!(solution.value, expected_value, "{status:?}");
            assert!(solution.primal.is_none());
            assert!(solution.get_value(scalar).is_none());
            assert!(solution.dual.is_none());
            assert!(!solution.has_duals());
            assert!(solution.constraint_dual(0).is_none());
            assert!(solution.constraint_rows.is_none());
            assert_eq!(solution.iterations, 7);
            assert_abs_diff_eq!(solution.solve_time, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_low_level_solve_preserves_raw_dual_indices() {
        use crate::canon::{ConeConstraint, LinExpr, QuadExpr};
        use crate::expr::Shape;
        use crate::solver::stuff_problem;

        let id = ExprId::new();
        let x = LinExpr::variable(id, Shape::scalar());
        let constraints = [
            ConeConstraint::NonNeg { a: x.clone() },
            ConeConstraint::NonNeg {
                a: x.add(&LinExpr::scalar(-1.0)),
            },
        ];
        let stuffed = stuff_problem(
            &QuadExpr::from_linear(x),
            &constraints,
            &[(id, Shape::scalar())],
        );
        let solution = solve(&stuffed, &Settings::default());

        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!(solution.constraint_rows.is_none());
        assert!(solution.constraint_dual(0).unwrap().abs() < 1e-6);
        assert!((solution.constraint_dual(1).unwrap() - 1.0).abs() < 1e-6);
        assert!(solution.constraint_dual(2).is_none());
    }

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert!(!settings.verbose);
        assert_eq!(settings.max_iter, 100);
    }

    #[test]
    fn test_to_clarabel_cones() {
        let dims = ConeDims {
            zero: 2,
            nonneg: 3,
            soc: vec![4],
            exp: 0,
            power: vec![],
        };
        let cones = to_clarabel_cones(&dims);
        assert_eq!(cones.len(), 3);
    }
}
