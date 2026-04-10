//! DCP (Disciplined Convex Programming) analysis.
//!
//! This module provides the core DCP analysis functionality:
//! - Curvature tracking (convex, concave, affine, constant)
//! - Sign tracking (non-negative, non-positive, unknown)
//! - DCP composition rules

pub mod curvature;
pub mod sign;

pub use curvature::{Curvature, PsdStatus, add_curvature, scalar_mul_curvature};
pub use sign::{Sign, add_sign, mul_sign};
