//! Core expression types for cvxrust.
//!
//! The `Expr` enum represents all possible expressions in the DCP framework.
//! Expressions form an immutable DAG (directed acyclic graph) using `Arc` for sharing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

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

/// Efficient array storage (dense or sparse).
#[derive(Debug, Clone)]
pub enum Array {
    /// Dense matrix storage.
    Dense(DMatrix<f64>),
    /// Sparse CSC matrix storage.
    Sparse(CscMatrix<f64>),
    /// Scalar value.
    Scalar(f64),
}

impl Array {
    /// Get the shape of the array.
    pub fn shape(&self) -> Shape {
        match self {
            Array::Dense(m) => Shape::matrix(m.nrows(), m.ncols()),
            Array::Sparse(m) => Shape::matrix(m.nrows(), m.ncols()),
            Array::Scalar(_) => Shape::scalar(),
        }
    }

    /// Get the total number of elements.
    pub fn size(&self) -> usize {
        match self {
            Array::Dense(m) => m.nrows() * m.ncols(),
            Array::Sparse(m) => m.nrows() * m.ncols(),
            Array::Scalar(_) => 1,
        }
    }

    /// Try to get as a scalar value.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Array::Scalar(v) => Some(*v),
            Array::Dense(m) if m.nrows() == 1 && m.ncols() == 1 => Some(m[(0, 0)]),
            _ => None,
        }
    }

    /// Check if all elements are non-negative.
    pub fn is_nonneg(&self) -> bool {
        match self {
            Array::Scalar(v) => *v >= 0.0,
            Array::Dense(m) => m.iter().all(|&v| v >= 0.0),
            Array::Sparse(m) => m.values().iter().all(|&v| v >= 0.0),
        }
    }

    /// Check if all elements are non-positive.
    pub fn is_nonpos(&self) -> bool {
        match self {
            Array::Scalar(v) => *v <= 0.0,
            Array::Dense(m) => m.iter().all(|&v| v <= 0.0),
            Array::Sparse(m) => {
                // For sparse, zeros are implicitly non-positive
                m.values().iter().all(|&v| v <= 0.0)
            }
        }
    }

    /// Check if the matrix is positive semi-definite (for symmetric matrices).
    pub fn is_psd(&self) -> Option<bool> {
        match self {
            Array::Scalar(v) => Some(*v >= 0.0),
            Array::Dense(m) => {
                if m.nrows() != m.ncols() {
                    return None;
                }
                // Check symmetry first
                let n = m.nrows();
                for i in 0..n {
                    for j in (i + 1)..n {
                        if (m[(i, j)] - m[(j, i)]).abs() > 1e-10 {
                            return None;
                        }
                    }
                }
                // Check eigenvalues (simple approach using Cholesky)
                Some(m.clone().cholesky().is_some())
            }
            Array::Sparse(_) => {
                // For sparse, would need to convert to dense or use specialized method
                None
            }
        }
    }

    /// Create from a scalar.
    pub fn from_scalar(v: f64) -> Self {
        Array::Scalar(v)
    }

    /// Create from a vector.
    pub fn from_vec(v: Vec<f64>) -> Self {
        let n = v.len();
        Array::Dense(DMatrix::from_vec(n, 1, v))
    }

    /// Create from a dense matrix.
    pub fn from_matrix(m: DMatrix<f64>) -> Self {
        Array::Dense(m)
    }
}

impl From<f64> for Array {
    fn from(v: f64) -> Self {
        Array::Scalar(v)
    }
}

impl From<Vec<f64>> for Array {
    fn from(v: Vec<f64>) -> Self {
        Array::from_vec(v)
    }
}

impl From<DMatrix<f64>> for Array {
    fn from(m: DMatrix<f64>) -> Self {
        Array::Dense(m)
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

/// Specification for indexing operations.
#[derive(Debug, Clone)]
pub struct IndexSpec {
    /// Ranges for each dimension: (start, stop, step).
    /// None means take the whole dimension.
    pub ranges: Vec<Option<(usize, usize, usize)>>,
}

impl IndexSpec {
    /// Create an index spec for a single element.
    pub fn element(indices: Vec<usize>) -> Self {
        IndexSpec {
            ranges: indices.into_iter().map(|i| Some((i, i + 1, 1))).collect(),
        }
    }

    /// Create an index spec for a range.
    pub fn range(start: usize, stop: usize) -> Self {
        IndexSpec {
            ranges: vec![Some((start, stop, 1))],
        }
    }

    /// Create an index spec that takes everything.
    pub fn all() -> Self {
        IndexSpec { ranges: vec![None] }
    }
}

/// The core expression type - an algebraic data type.
///
/// All expressions are immutable and use `Arc` for efficient sharing.
/// This allows building expression DAGs without copying.
#[derive(Debug, Clone)]
pub enum Expr {
    // ========== Leaf nodes ==========
    /// A decision variable.
    Variable(VariableData),
    /// A constant value.
    Constant(ConstantData),

    // ========== Affine atoms ==========
    /// Addition: a + b
    Add(Arc<Expr>, Arc<Expr>),
    /// Negation: -a
    Neg(Arc<Expr>),
    /// Multiplication: a * b (scalar or matrix)
    Mul(Arc<Expr>, Arc<Expr>),
    /// Summation with optional axis.
    Sum(Arc<Expr>, Option<usize>),
    /// Reshape to new shape.
    Reshape(Arc<Expr>, Shape),
    /// Indexing/slicing.
    Index(Arc<Expr>, IndexSpec),
    /// Vertical stack: [a; b; ...]
    VStack(Vec<Arc<Expr>>),
    /// Horizontal stack: [a, b, ...]
    HStack(Vec<Arc<Expr>>),
    /// Transpose.
    Transpose(Arc<Expr>),
    /// Matrix trace.
    Trace(Arc<Expr>),
    /// Matrix-vector or matrix-matrix multiplication.
    MatMul(Arc<Expr>, Arc<Expr>),

    // ========== Nonlinear atoms ==========
    /// L1 norm: ||x||_1
    Norm1(Arc<Expr>),
    /// L2 norm: ||x||_2
    Norm2(Arc<Expr>),
    /// Infinity norm: ||x||_inf
    NormInf(Arc<Expr>),
    /// Absolute value (elementwise).
    Abs(Arc<Expr>),
    /// Positive part: max(x, 0) (elementwise).
    Pos(Arc<Expr>),
    /// Negative part: max(-x, 0) (elementwise).
    NegPart(Arc<Expr>),
    /// Maximum of expressions.
    Maximum(Vec<Arc<Expr>>),
    /// Minimum of expressions.
    Minimum(Vec<Arc<Expr>>),
    /// Quadratic form: x' P x
    QuadForm(Arc<Expr>, Arc<Expr>),
    /// Sum of squares: ||x||_2^2
    SumSquares(Arc<Expr>),
    /// Quadratic over linear: ||x||_2^2 / y
    QuadOverLin(Arc<Expr>, Arc<Expr>),
    /// Exponential: exp(x) (elementwise).
    Exp(Arc<Expr>),
    /// Natural logarithm: log(x) (elementwise).
    Log(Arc<Expr>),
    /// Entropy: -x * log(x) (elementwise).
    Entropy(Arc<Expr>),
    /// Power: x^p (elementwise).
    Power(Arc<Expr>, f64),

    // ========== Additional affine atoms ==========
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
            Expr::Add(a, b) => a
                .shape()
                .broadcast(&b.shape())
                .unwrap_or_else(Shape::scalar),
            Expr::Neg(a) => a.shape(),
            Expr::Mul(a, b) => a
                .shape()
                .broadcast(&b.shape())
                .unwrap_or_else(Shape::scalar),
            Expr::Sum(a, axis) => {
                if axis.is_some() {
                    // Sum along axis reduces that dimension
                    let dims = a.shape();
                    if dims.ndim() <= 1 {
                        Shape::scalar()
                    } else {
                        Shape::vector(dims.cols())
                    }
                } else {
                    Shape::scalar()
                }
            }
            Expr::Reshape(_, shape) => shape.clone(),
            Expr::Index(a, spec) => {
                // Simplified: compute resulting shape from index spec
                let base = a.shape();
                let mut new_dims = Vec::new();
                for (i, r) in spec.ranges.iter().enumerate() {
                    match r {
                        Some((start, stop, step)) => {
                            let size = (stop - start).div_ceil(*step);
                            if size > 1 {
                                new_dims.push(size);
                            }
                        }
                        None => {
                            if i < base.ndim() {
                                new_dims.push(base.dims()[i]);
                            }
                        }
                    }
                }
                if new_dims.is_empty() {
                    Shape::scalar()
                } else {
                    Shape::from_dims(new_dims)
                }
            }
            Expr::VStack(exprs) => {
                if exprs.is_empty() {
                    return Shape::scalar();
                }
                let first = exprs[0].shape();
                let total_rows: usize = exprs.iter().map(|e| e.shape().rows()).sum();
                Shape::matrix(total_rows, first.cols())
            }
            Expr::HStack(exprs) => {
                if exprs.is_empty() {
                    return Shape::scalar();
                }
                let first = exprs[0].shape();
                let total_cols: usize = exprs.iter().map(|e| e.shape().cols()).sum();
                Shape::matrix(first.rows(), total_cols)
            }
            Expr::Transpose(a) => a.shape().transpose(),
            Expr::Trace(_) => Shape::scalar(),
            Expr::MatMul(a, b) => a.shape().matmul(&b.shape()).unwrap_or_else(Shape::scalar),

            // Nonlinear - norms return scalars
            Expr::Norm1(_) | Expr::Norm2(_) | Expr::NormInf(_) => Shape::scalar(),
            Expr::Abs(a) | Expr::Pos(a) | Expr::NegPart(a) => a.shape(),
            Expr::Maximum(exprs) | Expr::Minimum(exprs) => {
                if exprs.is_empty() {
                    Shape::scalar()
                } else {
                    // Element-wise max/min preserves shape
                    exprs[0].shape()
                }
            }
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
        self.collect_variables(&mut vars);
        vars.sort_by_key(|id| id.0);
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<ExprId>) {
        match self {
            Expr::Variable(v) => vars.push(v.id),
            Expr::Constant(_) => {}

            // Affine - recurse
            Expr::Add(a, b) | Expr::Mul(a, b) | Expr::MatMul(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Expr::Neg(a)
            | Expr::Sum(a, _)
            | Expr::Reshape(a, _)
            | Expr::Index(a, _)
            | Expr::Transpose(a)
            | Expr::Trace(a) => {
                a.collect_variables(vars);
            }
            Expr::VStack(exprs) | Expr::HStack(exprs) => {
                for e in exprs {
                    e.collect_variables(vars);
                }
            }

            // Nonlinear
            Expr::Norm1(a)
            | Expr::Norm2(a)
            | Expr::NormInf(a)
            | Expr::Abs(a)
            | Expr::Pos(a)
            | Expr::NegPart(a)
            | Expr::SumSquares(a) => {
                a.collect_variables(vars);
            }
            Expr::Maximum(exprs) | Expr::Minimum(exprs) => {
                for e in exprs {
                    e.collect_variables(vars);
                }
            }
            Expr::QuadForm(x, p) | Expr::QuadOverLin(x, p) => {
                x.collect_variables(vars);
                p.collect_variables(vars);
            }
            // Exponential cone atoms
            Expr::Exp(a) | Expr::Log(a) | Expr::Entropy(a) | Expr::Power(a, _) => {
                a.collect_variables(vars);
            }
            // Additional affine atoms
            Expr::Cumsum(a, _) | Expr::Diag(a) => {
                a.collect_variables(vars);
            }
        }
    }
}

impl std::ops::Index<(usize, usize)> for Array {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        match self {
            Array::Dense(m) => &m[(row, col)],
            Array::Scalar(v) => {
                assert!(row == 0 && col == 0, "scalar index out of bounds");
                v
            }
            Array::Sparse(_) => panic!("use Array::Dense for indexing"),
        }
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
    fn test_array_scalar() {
        let arr = Array::Scalar(5.0);
        assert_eq!(arr.as_scalar(), Some(5.0));
        assert!(arr.is_nonneg());
        assert!(!arr.is_nonpos());
    }

    #[test]
    fn test_array_from_vec() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(arr.shape(), Shape::matrix(3, 1));
        assert!(arr.is_nonneg());
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
}
