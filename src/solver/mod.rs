/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Solver interface for cvxrust.
//!
//! This module provides:
//! - Matrix stuffing to convert canonicalized problems to solver format
//! - Clarabel solver integration

pub mod clarabel;
mod solution;
pub mod stuffing;

pub use self::clarabel::{Settings, solve};
pub use solution::{Solution, SolveStatus};
pub use stuffing::{ConeDims, StuffedProblem, VariableMap, stuff_problem};
