/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Core expression types for cvxrust.
//!
//! The `Expr` enum represents all possible expressions in the DCP framework.
//! Expressions form an immutable DAG (directed acyclic graph) using `Arc` for sharing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use super::array::Array;
pub use super::indexing::{AxisIndex, IndexSpec};
use super::shape::Shape;

/// Unique identifier for expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(u64);

impl ExprId {
    /// Generate a new unique ID.
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        ExprId(NEXT_ID.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the raw ID value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for ExprId {
    fn default() -> Self {
        Self::new()
    }
}

/// Data for a variable expression.
#[derive(Debug, Clone)]
pub struct VariableData {
    /// Unique identifier.
    pub id: ExprId,
    /// Shape of the variable.
    pub shape: Shape,
    /// Optional name for display.
    pub name: Option<String>,
    /// Variable is constrained to be non-negative.
    pub nonneg: bool,
    /// Variable is constrained to be non-positive.
    pub nonpos: bool,
}

/// Data for a constant expression.
#[derive(Debug, Clone)]
pub struct ConstantData {
    /// Unique identifier.
    pub id: ExprId,
    /// The constant value.
    pub value: Array,
}

impl ConstantData {
    /// Get the shape of the constant.
    pub fn shape(&self) -> Shape {
        self.value.shape()
    }
}

/// The core expression type - an algebraic data type.
///
/// All expressions are immutable and use `Arc` for efficient sharing.
/// This allows building expression DAGs without copying.
#[derive(Debug, Clone)]
pub enum Expr {
    // Leaf nodes
    /// A decision variable.
    Variable(VariableData),
    /// A constant value.
    Constant(ConstantData),

    // Affine atoms
    /// Addition: `a + b`.
    Add(Arc<Expr>, Arc<Expr>),
    /// Negation: `-a`.
    Neg(Arc<Expr>),
    /// Multiplication: `a * b` (scalar or matrix).
    Mul(Arc<Expr>, Arc<Expr>),
    /// Promote a scalar-like expression to a larger shape.
    Promote(Arc<Expr>, Shape),
    /// Summation with optional axis.
    Sum(Arc<Expr>, Option<usize>),
    /// Reshape to new shape.
    Reshape(Arc<Expr>, Shape),
    /// Indexing/slicing.
    Index(Arc<Expr>, IndexSpec),
    /// Vertical stack: `[a; b; ...]`.
    VStack(Vec<Arc<Expr>>),
    /// Horizontal stack: `[a, b, ...]`.
    HStack(Vec<Arc<Expr>>),
    /// Transpose.
    Transpose(Arc<Expr>),
    /// Matrix trace.
    Trace(Arc<Expr>),
    /// Matrix-vector or matrix-matrix multiplication.
    MatMul(Arc<Expr>, Arc<Expr>),

    // Nonlinear atoms
    /// L1 norm of all entries.
    Norm1(Arc<Expr>),
    /// Euclidean norm of all entries.
    Norm2(Arc<Expr>),
    /// Largest absolute entry.
    NormInf(Arc<Expr>),
    /// Absolute value (elementwise).
    Abs(Arc<Expr>),
    /// Positive part: `max(x, 0)` (elementwise).
    Pos(Arc<Expr>),
    /// Negative part: `max(-x, 0)` (elementwise).
    NegPart(Arc<Expr>),
    /// Maximum of expressions.
    Maximum(Vec<Arc<Expr>>),
    /// Minimum of expressions.
    Minimum(Vec<Arc<Expr>>),
    /// Quadratic form: `x^T P x`.
    QuadForm(Arc<Expr>, Arc<Expr>),
    /// Sum of squares of all entries.
    SumSquares(Arc<Expr>),
    /// `sum_squares(x) / y` for scalar `y`.
    QuadOverLin(Arc<Expr>, Arc<Expr>),
    /// Exponential: `exp(x)` (elementwise).
    Exp(Arc<Expr>),
    /// Natural logarithm: `log(x)` (elementwise).
    Log(Arc<Expr>),
    /// Entropy: `-x * log(x)` (elementwise).
    Entropy(Arc<Expr>),
    /// Power: `x^p` (elementwise).
    Power(Arc<Expr>, f64),

    // Additional affine atoms
    /// Cumulative sum along axis.
    Cumsum(Arc<Expr>, Option<usize>),
    /// Diagonal matrix from vector (or diagonal of matrix).
    Diag(Arc<Expr>),
}

impl Expr {
    /// Get the shape of the expression.
    pub fn shape(&self) -> Shape {
        match self {
            Expr::Variable(v) => v.shape.clone(),
            Expr::Constant(c) => c.shape(),

            // Affine
            Expr::Add(a, b) => a.shape().broadcast(&b.shape()).unwrap_or_else(|| {
                panic!("cannot broadcast shapes {} and {}", a.shape(), b.shape())
            }),
            Expr::Neg(a) => a.shape(),
            Expr::Mul(a, b) => a.shape().broadcast(&b.shape()).unwrap_or_else(|| {
                panic!("cannot broadcast shapes {} and {}", a.shape(), b.shape())
            }),
            Expr::Promote(_, shape) => shape.clone(),
            Expr::Sum(a, axis) => {
                if axis.is_some() {
                    // Sum along axis reduces that dimension
                    let dims = a.shape();
                    match (dims.ndim(), axis.unwrap()) {
                        (0 | 1, 0) => Shape::scalar(),
                        (2, 0) => Shape::vector(dims.cols()),
                        (2, 1) => Shape::vector(dims.rows()),
                        (_, axis) => panic!("axis {} out of bounds for shape {}", axis, dims),
                    }
                } else {
                    Shape::scalar()
                }
            }
            Expr::Reshape(_, shape) => shape.clone(),
            Expr::Index(a, spec) => spec.output_shape(&a.shape()),
            Expr::VStack(exprs) => {
                if exprs.is_empty() {
                    return Shape::scalar();
                }
                let first = exprs[0].shape();
                for e in &exprs[1..] {
                    assert_eq!(
                        e.shape().cols(),
                        first.cols(),
                        "vstack requires matching column counts"
                    );
                }
                let total_rows: usize = exprs.iter().map(|e| e.shape().rows()).sum();
                Shape::matrix(total_rows, first.cols())
            }
            Expr::HStack(exprs) => {
                if exprs.is_empty() {
                    return Shape::scalar();
                }
                let first = exprs[0].shape();
                for e in &exprs[1..] {
                    assert_eq!(
                        e.shape().rows(),
                        first.rows(),
                        "hstack requires matching row counts"
                    );
                }
                let total_cols: usize = exprs.iter().map(|e| e.shape().cols()).sum();
                Shape::matrix(first.rows(), total_cols)
            }
            Expr::Transpose(a) => a.shape().transpose(),
            Expr::Trace(_) => Shape::scalar(),
            Expr::MatMul(a, b) => a.shape().matmul(&b.shape()).unwrap_or_else(|| {
                panic!(
                    "cannot matrix-multiply shapes {} and {}",
                    a.shape(),
                    b.shape()
                )
            }),

            // Nonlinear - norms return scalars
            Expr::Norm1(_) | Expr::Norm2(_) | Expr::NormInf(_) => Shape::scalar(),
            Expr::Abs(a) | Expr::Pos(a) | Expr::NegPart(a) => a.shape(),
            Expr::Maximum(exprs) | Expr::Minimum(exprs) => exprs
                .iter()
                .map(|expr| expr.shape())
                .reduce(|a, b| {
                    a.broadcast(&b)
                        .unwrap_or_else(|| panic!("cannot broadcast shapes {} and {}", a, b))
                })
                .unwrap_or_else(Shape::scalar),
            Expr::QuadForm(_, _) | Expr::SumSquares(_) | Expr::QuadOverLin(_, _) => Shape::scalar(),
            // Exponential cone atoms (elementwise)
            Expr::Exp(a) | Expr::Log(a) | Expr::Entropy(a) | Expr::Power(a, _) => a.shape(),
            // Additional affine atoms
            Expr::Cumsum(a, _) => a.shape(),
            Expr::Diag(a) => {
                let s = a.shape();
                if s.is_vector() {
                    // Vector to diagonal matrix: n -> (n, n)
                    let n = s.size();
                    Shape::matrix(n, n)
                } else {
                    // Matrix to diagonal vector: (m, n) -> min(m,n)
                    let n = s.rows().min(s.cols());
                    Shape::vector(n)
                }
            }
        }
    }

    /// Get the unique ID if this is a variable.
    pub fn variable_id(&self) -> Option<ExprId> {
        match self {
            Expr::Variable(v) => Some(v.id),
            _ => None,
        }
    }

    /// Check if this expression is a constant.
    pub fn is_constant(&self) -> bool {
        matches!(self, Expr::Constant(_))
    }

    /// Check if this expression is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, Expr::Variable(_))
    }

    /// Get the constant value if this is a constant expression.
    pub fn constant_value(&self) -> Option<&Array> {
        match self {
            Expr::Constant(c) => Some(&c.value),
            _ => None,
        }
    }

    /// Collect all variables in this expression.
    pub fn variables(&self) -> Vec<ExprId> {
        let mut vars = Vec::new();
        self.visit_variables(&mut |variable| vars.push(variable.id));
        vars.sort_by_key(|id| id.0);
        vars.dedup();
        vars
    }
}

// Convenient From implementations for automatic conversion
impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        crate::expr::constant(value)
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        crate::expr::constant(value as f64)
    }
}

impl From<&Expr> for Expr {
    fn from(expr: &Expr) -> Self {
        expr.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_id() {
        let id1 = ExprId::new();
        let id2 = ExprId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_variable_shape() {
        let var = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::vector(5),
            name: Some("x".to_string()),
            nonneg: false,
            nonpos: false,
        });
        assert_eq!(var.shape(), Shape::vector(5));
        assert!(var.is_variable());
    }

    #[test]
    fn test_constant_shape() {
        let c = Expr::Constant(ConstantData {
            id: ExprId::new(),
            value: Array::from_vec(vec![1.0, 2.0, 3.0]),
        });
        assert_eq!(c.shape(), Shape::matrix(3, 1));
        assert!(c.is_constant());
    }

    #[test]
    fn test_sum_axis_shape() {
        let x = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::matrix(2, 3),
            name: None,
            nonneg: false,
            nonpos: false,
        });

        assert_eq!(
            Expr::Sum(Arc::new(x.clone()), Some(0)).shape(),
            Shape::vector(3)
        );
        assert_eq!(Expr::Sum(Arc::new(x), Some(1)).shape(), Shape::vector(2));
    }

    #[test]
    #[should_panic(expected = "axis 2 out of bounds for shape (2, 3)")]
    fn test_sum_axis_invalid_direct_expr_shape_panics() {
        let x = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::matrix(2, 3),
            name: None,
            nonneg: false,
            nonpos: false,
        });

        let _ = Expr::Sum(Arc::new(x), Some(2)).shape();
    }

    #[test]
    #[should_panic(expected = "vstack requires matching column counts")]
    fn test_vstack_mismatched_columns_direct_expr_shape_panics() {
        let x = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::matrix(2, 3),
            name: None,
            nonneg: false,
            nonpos: false,
        });
        let y = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::matrix(3, 2),
            name: None,
            nonneg: false,
            nonpos: false,
        });

        let _ = Expr::VStack(vec![Arc::new(x), Arc::new(y)]).shape();
    }

    #[test]
    #[should_panic(expected = "hstack requires matching row counts")]
    fn test_hstack_mismatched_rows_direct_expr_shape_panics() {
        let x = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::matrix(2, 3),
            name: None,
            nonneg: false,
            nonpos: false,
        });
        let y = Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: Shape::matrix(3, 3),
            name: None,
            nonneg: false,
            nonpos: false,
        });

        let _ = Expr::HStack(vec![Arc::new(x), Arc::new(y)]).shape();
    }
}
