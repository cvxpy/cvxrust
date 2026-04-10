//! Atom functions for building expressions.
//!
//! Atoms are the building blocks of optimization problems. They include:
//!
//! - **Affine atoms**: Operations that preserve linearity (add, mul, sum, reshape, etc.)
//! - **Nonlinear atoms**: Operations with specific curvature (norms, quadratic forms, etc.)

pub mod affine;
pub mod nonlinear;

// Re-export affine operations
pub use affine::{
    cumsum, diag, dot, flatten, hstack, index, matmul, reshape, slice, sum, sum_axis, trace,
    transpose, vstack,
};

// Re-export nonlinear atoms
pub use nonlinear::{
    abs, entropy, exp, log, max2, maximum, min2, minimum, neg_part, norm, norm_inf, norm1, norm2,
    pos, power, quad_form, quad_over_lin, sqrt, sum_squares, try_norm,
};
