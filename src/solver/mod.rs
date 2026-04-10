//! Solver interface for cvxrust.
//!
//! This module provides:
//! - Matrix stuffing to convert canonicalized problems to solver format
//! - Clarabel solver integration

pub mod clarabel;
pub mod stuffing;

pub use self::clarabel::{Settings, Solution, SolveStatus, solve};
pub use stuffing::{ConeDims, StuffedProblem, VariableMap, stuff_problem};
