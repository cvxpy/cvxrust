//! # cvxrust
//!
//! A Rust implementation of Disciplined Convex Programming (DCP).
//!
//! cvxrust provides a domain-specific language for specifying convex optimization
//! problems in Rust, with automatic verification of convexity rules and efficient
//! solving via the Clarabel solver.
//!
//! ## Quick Start
//!
//! ```ignore
//! use cvxrust::prelude::*;
//!
//! // Create variables
//! let x = variable(5);
//!
//! // Build and solve a least-squares problem
//! let a = constant_dmatrix(/* your matrix */);
//! let b = constant_vec(/* your vector */);
//!
//! let solution = Problem::minimize(sum_squares(&(&a.matmul(&x) - &b)))
//!     .subject_to([x.ge(constant(0.0))])
//!     .solve()?;
//!
//! println!("Optimal value: {}", solution.value.unwrap());
//! ```
//!
//! ## DCP Rules
//!
//! cvxrust enforces [Disciplined Convex Programming](https://dcp.stanford.edu/) rules:
//!
//! - **Objective:** `minimize(convex)` or `maximize(concave)`
//! - **Constraints:** `convex <= concave`, `concave >= convex`, `affine == affine`
//!
//! Curvature follows DCP composition rules (e.g., `convex + convex = convex`).
//! See <https://dcp.stanford.edu/> for details.
//!
//! ## Supported Atoms
//!
//! ### Affine (both convex and concave)
//! - Arithmetic: `+`, `-`, `*` (by scalar), `/` (by scalar)
//! - Aggregation: `sum`, `trace`
//! - Structural: `reshape`, `transpose`, `vstack`, `hstack`
//! - Linear algebra: `matmul`, `dot`
//!
//! ### Convex
//! - Norms: `norm1`, `norm2`, `norm_inf`
//! - Element-wise: `abs`, `pos`, `neg_part`
//! - Aggregation: `maximum`, `sum_squares`
//! - Quadratic: `quad_form` (with PSD matrix), `quad_over_lin`
//!
//! ### Concave
//! - Aggregation: `minimum`
//! - Quadratic: `quad_form` (with NSD matrix)
//!
//! ## Architecture
//!
//! - **Expression trees** built using the `Expr` enum with `Arc` sharing
//! - **DCP verification** via curvature and sign tracking
//! - **Canonicalization** transforms to affine + cone constraints
//! - **Native QP** for quadratic objectives (not SOCP reformulation)
//! - **Clarabel solver** for LP, QP, and SOCP problems

pub mod atoms;
pub mod canon;
pub mod constraints;
pub mod dcp;
pub mod error;
pub mod expr;
pub mod problem;
pub mod solver;
pub mod sparse;

/// Prelude module for convenient imports.
///
/// ```ignore
/// use cvxrust::prelude::*;
/// ```
pub mod prelude {
    // Expression types
    pub use crate::expr::{
        Array, Evaluable, Expr, ExprId, IntoConstant, Shape, VariableBuilder, VariableExt,
        constant, constant_dmatrix, constant_matrix, constant_sparse, constant_vec, eye, ones,
        variable, zeros,
    };

    // Atoms
    pub use crate::atoms::{
        abs, cumsum, diag, dot, entropy, exp, flatten, hstack, log, matmul, max2, maximum, min2,
        minimum, neg_part, norm, norm_inf, norm1, norm2, pos, power, quad_form, quad_over_lin,
        reshape, sqrt, sum, sum_axis, sum_squares, trace, transpose, try_norm, vstack,
    };

    // Constraints
    pub use crate::constraint;
    pub use crate::constraints::{Constraint, ConstraintExt};

    // DCP
    pub use crate::dcp::{Curvature, Sign};

    // Problem
    pub use crate::problem::{Objective, Problem, ProblemBuilder};

    // Solver
    pub use crate::solver::{Settings, Solution, SolveStatus};

    // Errors
    pub use crate::error::{CvxError, Result};
}

// Re-export main types at crate root
pub use error::{CvxError, Result};
pub use problem::Problem;
pub use solver::{Solution, SolveStatus};
