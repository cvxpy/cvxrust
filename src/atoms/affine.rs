/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Affine atom constructors for reductions, indexing, and matrix operations.

use std::sync::Arc;

use crate::expr::{AxisIndex, Expr, IndexSpec, Shape};

/// Sum of all elements, or along an axis.
pub fn sum(expr: &Expr) -> Expr {
    Expr::Sum(Arc::new(expr.clone()), None)
}

/// Sum along a specific axis.
pub fn sum_axis(expr: &Expr, axis: usize) -> Expr {
    let shape = expr.shape();
    assert!(
        axis < shape.ndim().max(1),
        "axis {} out of bounds for shape {}",
        axis,
        shape
    );
    Expr::Sum(Arc::new(expr.clone()), Some(axis))
}

/// Promote a scalar-like expression to a target shape.
pub fn promote(expr: &Expr, shape: impl Into<Shape>) -> Expr {
    let target_shape = shape.into();
    let expr_shape = expr.shape();

    if expr_shape == target_shape {
        return expr.clone();
    }

    assert!(
        expr_shape.is_scalar_like(),
        "only scalar expressions can be promoted"
    );

    Expr::Promote(Arc::new(expr.clone()), target_shape)
}

/// Reshape an expression to a new shape.
pub fn reshape(expr: &Expr, shape: impl Into<Shape>) -> Expr {
    let shape = shape.into();
    assert_eq!(
        expr.shape().size(),
        shape.size(),
        "cannot reshape size {} into shape {}",
        expr.shape().size(),
        shape
    );
    Expr::Reshape(Arc::new(expr.clone()), shape)
}

/// Flatten an expression to a vector.
pub fn flatten(expr: &Expr) -> Expr {
    let size = expr.shape().size();
    Expr::Reshape(Arc::new(expr.clone()), Shape::vector(size))
}

/// Transpose an expression.
pub fn transpose(expr: &Expr) -> Expr {
    Expr::Transpose(Arc::new(expr.clone()))
}

/// Matrix trace.
pub fn trace(expr: &Expr) -> Expr {
    Expr::Trace(Arc::new(expr.clone()))
}

/// Vertical stack (row-wise concatenation).
pub fn vstack(exprs: Vec<Expr>) -> Expr {
    if let Some(first) = exprs.first() {
        let cols = first.shape().cols();
        for expr in &exprs[1..] {
            assert_eq!(
                expr.shape().cols(),
                cols,
                "vstack requires matching column counts"
            );
        }
    }
    Expr::VStack(exprs.into_iter().map(Arc::new).collect())
}

/// Horizontal stack (column-wise concatenation).
pub fn hstack(exprs: Vec<Expr>) -> Expr {
    if let Some(first) = exprs.first() {
        let rows = first.shape().rows();
        for expr in &exprs[1..] {
            assert_eq!(
                expr.shape().rows(),
                rows,
                "hstack requires matching row counts"
            );
        }
    }
    Expr::HStack(exprs.into_iter().map(Arc::new).collect())
}

/// Matrix-vector or matrix-matrix multiplication.
pub fn matmul(a: &Expr, b: &Expr) -> Expr {
    Expr::MatMul(Arc::new(a.clone()), Arc::new(b.clone()))
}

/// Dot product (inner product) of two vectors.
pub fn dot(a: &Expr, b: &Expr) -> Expr {
    // dot(a, b) = sum(a * b) for element-wise product
    // or a^T b for vector dot product
    Expr::MatMul(
        Arc::new(Expr::Transpose(Arc::new(a.clone()))),
        Arc::new(b.clone()),
    )
}

/// Index into an expression.
pub fn index(expr: &Expr, idx: usize) -> Expr {
    select(expr, AxisIndex::Index(idx), AxisIndex::All)
}

/// Slice a range from an expression.
pub fn slice(expr: &Expr, start: usize, stop: usize) -> Expr {
    select(expr, AxisIndex::Slice(start, stop), AxisIndex::All)
}

/// Index a matrix column.
pub fn indexc(expr: &Expr, idx: usize) -> Expr {
    select(expr, AxisIndex::All, AxisIndex::Index(idx))
}

/// Slice a range of matrix columns.
pub fn slicec(expr: &Expr, start: usize, stop: usize) -> Expr {
    select(expr, AxisIndex::All, AxisIndex::Slice(start, stop))
}

/// Select rows and columns from an expression.
pub fn select(expr: &Expr, rows: AxisIndex, cols: AxisIndex) -> Expr {
    let shape = expr.shape();
    assert!(!shape.is_scalar(), "cannot select from a scalar expression");

    let spec = if shape.is_vector() {
        assert!(
            cols == AxisIndex::All,
            "vector selection only supports AxisIndex::All for columns"
        );
        IndexSpec {
            ranges: vec![axis_index_to_range(rows, shape.rows(), "first")],
            drop_axes: vec![axis_index_drops_axis(rows)],
        }
    } else if shape.is_matrix() {
        IndexSpec {
            ranges: vec![
                axis_index_to_range(rows, shape.rows(), "row"),
                axis_index_to_range(cols, shape.cols(), "column"),
            ],
            drop_axes: vec![axis_index_drops_axis(rows), axis_index_drops_axis(cols)],
        }
    } else {
        panic!("select only supports vector and matrix expressions");
    };
    Expr::Index(Arc::new(expr.clone()), spec)
}

fn axis_index_to_range(
    selector: AxisIndex,
    axis_len: usize,
    axis_name: &str,
) -> Option<(usize, usize, usize)> {
    match selector {
        AxisIndex::Index(idx) => {
            assert!(
                idx < axis_len,
                "{} index {} out of bounds for axis with length {}",
                axis_name,
                idx,
                axis_len
            );
            Some((idx, idx + 1, 1))
        }
        AxisIndex::Slice(start, stop) => {
            assert!(
                start <= stop,
                "{} slice start {} must be less than or equal to stop {}",
                axis_name,
                start,
                stop
            );
            assert!(
                stop <= axis_len,
                "{} slice stop {} out of bounds for axis with length {}",
                axis_name,
                stop,
                axis_len
            );
            Some((start, stop, 1))
        }
        AxisIndex::All => None,
    }
}

fn axis_index_drops_axis(selector: AxisIndex) -> bool {
    matches!(selector, AxisIndex::Index(_))
}

/// Cumulative sum along an axis.
///
/// Returns `cumsum([x_1, x_2, x_3]) = [x_1, x_1 + x_2, x_1 + x_2 + x_3]`.
pub fn cumsum(expr: &Expr) -> Expr {
    Expr::Cumsum(Arc::new(expr.clone()), None)
}

/// Diagonal matrix from vector, or diagonal of matrix.
///
/// - Vector input: Creates diagonal matrix with vector on diagonal
/// - Matrix input: Extracts diagonal as vector
pub fn diag(expr: &Expr) -> Expr {
    Expr::Diag(Arc::new(expr.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, constant_matrix, constant_vec, nonneg_variable, variable};

    #[test]
    fn test_sum() {
        let x = variable((3, 4));
        let s = sum(&x);
        assert_eq!(s.shape(), Shape::scalar());
    }

    #[test]
    fn test_sum_axis_vector_shape() {
        let x = variable(3);
        let s = sum_axis(&x, 0);
        assert_eq!(s.shape(), Shape::scalar());
    }

    #[test]
    #[should_panic(expected = "axis 1 out of bounds for shape (3,)")]
    fn test_sum_axis_invalid_axis_panics() {
        let x = variable(3);
        let _ = sum_axis(&x, 1);
    }

    #[test]
    fn test_promote_shape_and_metadata() {
        let x = nonneg_variable(());
        let p = promote(&x, (2, 3));

        assert_eq!(p.shape(), Shape::matrix(2, 3));
        assert_eq!(p.variables(), x.variables());
        assert!(p.curvature().is_affine());
        assert!(p.sign().is_nonneg());
    }

    #[test]
    fn test_promote_accepts_scalar_like_shapes() {
        let row_scalar = constant_vec(vec![2.0]);
        let matrix_scalar = constant_matrix(vec![3.0], 1, 1);

        assert_eq!(promote(&row_scalar, (2, 2)).shape(), Shape::matrix(2, 2));
        assert_eq!(promote(&matrix_scalar, (2, 2)).shape(), Shape::matrix(2, 2));
    }

    #[test]
    fn test_promote_same_shape_is_noop() {
        let x = variable(3);
        assert_eq!(promote(&x, 3).shape(), Shape::vector(3));
    }

    #[test]
    #[should_panic(expected = "cannot reshape size 3 into shape (2, 2)")]
    fn test_reshape_size_mismatch_panics() {
        let x = variable(3);
        let _ = reshape(&x, (2, 2));
    }

    #[test]
    #[should_panic(expected = "only scalar expressions can be promoted")]
    fn test_promote_rejects_non_scalar_like_shape() {
        let x = variable(2);
        let _ = promote(&x, (2, 2));
    }

    #[test]
    fn test_transpose() {
        let x = variable((3, 4));
        let t = transpose(&x);
        assert_eq!(t.shape(), Shape::matrix(4, 3));
    }

    #[test]
    fn test_matmul() {
        let a = variable((3, 4));
        let x = variable(4);
        let b = matmul(&a, &x);
        assert_eq!(b.shape(), Shape::vector(3));
    }

    #[test]
    #[should_panic(expected = "cannot matrix-multiply shapes (3, 4) and (3,)")]
    fn test_invalid_matmul_shape_panics() {
        let a = variable((3, 4));
        let x = variable(3);
        let _ = matmul(&a, &x).shape();
    }

    #[test]
    fn test_index_and_slice_shapes() {
        let x = variable(10);
        assert_eq!(index(&x, 1).shape(), Shape::scalar());
        assert_eq!(slice(&x, 0, 5).shape(), Shape::vector(5));
        assert_eq!(
            select(&x, AxisIndex::Slice(1, 3), AxisIndex::All).shape(),
            Shape::vector(2)
        );

        let x = variable((10, 10));
        assert_eq!(index(&x, 1).shape(), Shape::vector(10));
        assert_eq!(slice(&x, 0, 5).shape(), Shape::matrix(5, 10));
        assert_eq!(slice(&x, 1, 2).shape(), Shape::matrix(1, 10));
        assert_eq!(
            select(&x, AxisIndex::All, AxisIndex::Index(2)).shape(),
            Shape::vector(10)
        );
        assert_eq!(indexc(&x, 2).shape(), Shape::vector(10));
        assert_eq!(slicec(&x, 1, 3).shape(), Shape::matrix(10, 2));
        assert_eq!(
            select(&x, AxisIndex::Index(1), AxisIndex::Index(2)).shape(),
            Shape::scalar()
        );
        assert_eq!(
            select(&x, AxisIndex::Slice(0, 5), AxisIndex::Slice(1, 3)).shape(),
            Shape::matrix(5, 2)
        );
    }

    #[test]
    #[should_panic(expected = "row index 10 out of bounds")]
    fn test_matrix_index_out_of_bounds_panics() {
        let x = variable((10, 10));
        let _ = index(&x, 10);
    }

    #[test]
    #[should_panic(expected = "row slice stop 11 out of bounds")]
    fn test_matrix_slice_stop_out_of_bounds_panics() {
        let x = variable((10, 10));
        let _ = slice(&x, 0, 11);
    }

    #[test]
    #[should_panic(expected = "row slice start 5 must be less than or equal to stop 3")]
    fn test_slice_start_after_stop_panics() {
        let x = variable((10, 10));
        let _ = slice(&x, 5, 3);
    }

    #[test]
    #[should_panic(expected = "cannot select from a scalar expression")]
    fn test_scalar_index_panics() {
        let x = variable(());
        let _ = index(&x, 0);
    }

    #[test]
    #[should_panic(expected = "vector selection only supports AxisIndex::All for columns")]
    fn test_vector_column_select_panics() {
        let x = variable(10);
        let _ = select(&x, AxisIndex::All, AxisIndex::Index(0));
    }

    #[test]
    #[should_panic(expected = "column index 10 out of bounds")]
    fn test_matrix_column_index_out_of_bounds_panics() {
        let x = variable((10, 10));
        let _ = select(&x, AxisIndex::All, AxisIndex::Index(10));
    }

    #[test]
    #[should_panic(expected = "column slice stop 11 out of bounds")]
    fn test_matrix_column_slice_stop_out_of_bounds_panics() {
        let x = variable((10, 10));
        let _ = select(&x, AxisIndex::All, AxisIndex::Slice(0, 11));
    }

    #[test]
    fn test_vstack() {
        let x = variable((2, 3));
        let y = variable((3, 3));
        let z = vstack(vec![x, y]);
        assert_eq!(z.shape(), Shape::matrix(5, 3));
    }

    #[test]
    #[should_panic(expected = "vstack requires matching column counts")]
    fn test_vstack_mismatched_columns_panics() {
        let x = variable((2, 3));
        let y = variable((3, 2));
        let _ = vstack(vec![x, y]);
    }

    #[test]
    #[should_panic(expected = "hstack requires matching row counts")]
    fn test_hstack_mismatched_rows_panics() {
        let x = variable((2, 3));
        let y = variable((3, 3));
        let _ = hstack(vec![x, y]);
    }

    #[test]
    fn test_affine_is_affine() {
        let x = variable(5);
        let y = variable(5);
        let _c = constant(2.0);

        // x + y is affine
        let z = &x + &y;
        assert!(z.is_affine());

        // 2*x is affine
        let z = 2.0 * &x;
        assert!(z.is_affine());

        // sum(x) is affine
        let s = sum(&x);
        assert!(s.is_affine());
    }
}
