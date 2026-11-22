//! Solve integration tests, grouped by behavior.

mod atoms;
mod broadcasting;
mod domains;
mod matrix;
mod stress;

/// Tolerance for comparing floating point results.
const TOL: f64 = 1e-4;
