/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Solve integration tests, grouped by behavior.

mod atoms;
mod broadcasting;
mod domains;
mod matrix;
mod stress;

/// Tolerance for comparing floating point results.
const TOL: f64 = 1e-4;
