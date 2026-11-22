/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Canonicalization transforms expressions into standard form.
//!
//! This module converts DCP expressions into:
//! - Linear expressions (LinExpr) for affine parts
//! - Quadratic expressions (QuadExpr) for QP objectives
//! - Cone constraints (ConeConstraint) for nonlinear atoms

pub mod canonicalizer;
pub(crate) mod constraints;
pub mod lin_expr;

pub use canonicalizer::{CanonExpr, CanonResult, ConeConstraint, VariableDomains, canonicalize};
pub use lin_expr::{LinExpr, QuadExpr};
