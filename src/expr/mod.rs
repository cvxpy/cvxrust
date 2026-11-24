/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Expression types and creation utilities.
//!
//! This module provides the core expression types for building optimization problems:
//! - `Expr` - The main expression enum representing all expressions
//! - `Shape` - Shape information for expressions
//! - Variable creation via `variable()` and `VariableBuilder`
//! - Constant creation via `constant()` and related functions

mod array;
pub mod constant;
pub mod eval;
pub mod expression;
mod indexing;
mod operators;
pub mod shape;
mod traversal;
pub mod variable;

// Re-export main types
pub(crate) use array::symmetric_part;
pub use constant::{
    IntoConstant, constant, constant_dmatrix, constant_matrix, constant_sparse, constant_vec, eye,
    ones, zeros,
};
pub use eval::Evaluable;
pub use expression::{Array, AxisIndex, ConstantData, Expr, ExprId, IndexSpec, VariableData};
pub use shape::Shape;
pub use variable::{
    VariableBuilder, VariableExt, matrix_var, named_variable, nonneg_variable, nonpos_variable,
    scalar_var, var, variable, vector_var,
};
