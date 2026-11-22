/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Error types for cvxrust.

use thiserror::Error;

/// Error type for cvxrust operations.
#[derive(Debug, Error)]
pub enum CvxError {
    /// Problem is not DCP-compliant.
    #[error("Problem is not DCP: {0}")]
    NotDcp(String),

    /// Solver error.
    #[error("Solver error: {0}")]
    SolverError(String),

    /// Shape mismatch.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Invalid problem specification.
    #[error("Invalid problem: {0}")]
    InvalidProblem(String),

    /// Numerical error.
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Result type for cvxrust operations.
pub type Result<T> = std::result::Result<T, CvxError>;
