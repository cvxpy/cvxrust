//! Variable creation with builder pattern.

use std::sync::Arc;

use super::expression::{Expr, ExprId, VariableData};
use super::shape::Shape;

/// Builder for creating variables with various attributes.
#[derive(Default)]
pub struct VariableBuilder {
    shape: Shape,
    name: Option<String>,
    nonneg: bool,
    nonpos: bool,
}

impl VariableBuilder {
    /// Create a new variable builder with the given shape.
    pub fn new(shape: impl Into<Shape>) -> Self {
        Self {
            shape: shape.into(),
            ..Default::default()
        }
    }

    /// Create a scalar variable builder.
    pub fn scalar() -> Self {
        Self::new(Shape::scalar())
    }

    /// Create a vector variable builder.
    pub fn vector(n: usize) -> Self {
        Self::new(Shape::vector(n))
    }

    /// Create a matrix variable builder.
    pub fn matrix(m: usize, n: usize) -> Self {
        Self::new(Shape::matrix(m, n))
    }

    /// Set the name of the variable.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Constrain the variable to be non-negative (x >= 0).
    /// Replaces any previous non-positive declaration on this builder.
    pub fn nonneg(mut self) -> Self {
        self.nonneg = true;
        self.nonpos = false; // Setting a sign replaces the previous declaration.
        self
    }

    /// Constrain the variable to be non-positive (x <= 0).
    /// Replaces any previous non-negative declaration on this builder.
    pub fn nonpos(mut self) -> Self {
        self.nonpos = true;
        self.nonneg = false; // Setting a sign replaces the previous declaration.
        self
    }

    /// Build the variable expression.
    pub fn build(self) -> Expr {
        Expr::Variable(VariableData {
            id: ExprId::new(),
            shape: self.shape,
            name: self.name,
            nonneg: self.nonneg,
            nonpos: self.nonpos,
        })
    }
}

/// Create a variable with the given shape.
///
/// # Examples
///
/// ```
/// use cvxrust::expr::variable;
///
/// // Scalar variable
/// let x = variable(());
///
/// // Vector variable
/// let y = variable(5);
/// let y = variable((5,));
///
/// // Matrix variable
/// let z = variable((3, 4));
/// ```
pub fn variable(shape: impl Into<Shape>) -> Expr {
    VariableBuilder::new(shape).build()
}

/// Extension trait for variable-like operations on Expr.
///
/// Sign setters preserve the variable ID and replace the sign declaration on
/// this expression. Other clones are unchanged. When multiple clones appear in
/// one problem, their sign declarations are all enforced.
pub trait VariableExt {
    /// Declare this variable non-negative, replacing its non-positive declaration.
    fn nonneg(self) -> Expr;

    /// Declare this variable non-positive, replacing its non-negative declaration.
    fn nonpos(self) -> Expr;

    /// Give a name to this expression (if it's a variable).
    fn named(self, name: impl Into<String>) -> Expr;
}

impl VariableExt for Expr {
    fn nonneg(self) -> Expr {
        match self {
            Expr::Variable(mut v) => {
                v.nonneg = true;
                v.nonpos = false;
                Expr::Variable(v)
            }
            other => other,
        }
    }

    fn nonpos(self) -> Expr {
        match self {
            Expr::Variable(mut v) => {
                v.nonpos = true;
                v.nonneg = false;
                Expr::Variable(v)
            }
            other => other,
        }
    }

    fn named(self, name: impl Into<String>) -> Expr {
        match self {
            Expr::Variable(mut v) => {
                v.name = Some(name.into());
                Expr::Variable(v)
            }
            other => other,
        }
    }
}

/// Create a named variable with the given shape.
pub fn named_variable(name: impl Into<String>, shape: impl Into<Shape>) -> Expr {
    VariableBuilder::new(shape).name(name).build()
}

/// Create a non-negative variable with the given shape.
pub fn nonneg_variable(shape: impl Into<Shape>) -> Expr {
    VariableBuilder::new(shape).nonneg().build()
}

/// Create a non-positive variable with the given shape.
pub fn nonpos_variable(shape: impl Into<Shape>) -> Expr {
    VariableBuilder::new(shape).nonpos().build()
}

// Convenience functions with common shapes

/// Create a scalar variable.
pub fn scalar_var() -> Expr {
    VariableBuilder::scalar().build()
}

/// Create a vector variable.
pub fn vector_var(n: usize) -> Expr {
    VariableBuilder::vector(n).build()
}

/// Create a matrix variable.
pub fn matrix_var(m: usize, n: usize) -> Expr {
    VariableBuilder::matrix(m, n).build()
}

/// Create an Arc-wrapped variable for use in expressions.
pub fn var(shape: impl Into<Shape>) -> Arc<Expr> {
    Arc::new(variable(shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_builder() {
        let x = VariableBuilder::vector(5).name("x").nonneg().build();

        if let Expr::Variable(v) = &x {
            assert_eq!(v.shape, Shape::vector(5));
            assert_eq!(v.name, Some("x".to_string()));
            assert!(v.nonneg);
            assert!(!v.nonpos);
        } else {
            panic!("Expected Variable");
        }
    }

    #[test]
    fn test_variable_function() {
        let x = variable((3, 4));
        assert_eq!(x.shape(), Shape::matrix(3, 4));
    }

    #[test]
    fn test_variable_ext() {
        let x = variable(5).nonneg().named("x");
        if let Expr::Variable(v) = &x {
            assert!(v.nonneg);
            assert_eq!(v.name, Some("x".to_string()));
        } else {
            panic!("Expected Variable");
        }
    }

    #[test]
    fn test_convenience_functions() {
        assert_eq!(scalar_var().shape(), Shape::scalar());
        assert_eq!(vector_var(5).shape(), Shape::vector(5));
        assert_eq!(matrix_var(3, 4).shape(), Shape::matrix(3, 4));
    }
}
