//! Expression evaluation given a solution.
//!
//! This module provides the `Evaluable` trait and `Expr::eval()` method,
//! allowing users to compute the value of any expression (not just variables)
//! after solving a problem.
//!
//! # Example
//!
//! ```
//! use cvxrust::prelude::*;
//!
//! let x = variable(5);
//! let obj = norm2(&x);
//!
//! let solution = Problem::minimize(obj.clone())
//!     .subject_to([x.ge(1.0)])
//!     .solve()
//!     .unwrap();
//!
//! // Evaluate any expression, not just variables
//! let norm_val = obj.value(&solution);
//! let x_vals = x.eval(&solution).unwrap();
//! ```

use nalgebra::DMatrix;

use super::expression::{Array, Expr, ExprId, IndexSpec};
use super::shape::Shape;

/// Trait for types that provide variable values (used for expression evaluation).
///
/// Implement this trait to allow `Expr::eval()` to look up variable values.
/// The `Solution` type in `cvxrust::solver` implements this trait.
pub trait Evaluable {
    /// Get the value of a variable by its ID.
    fn get_variable_value(&self, id: ExprId) -> Option<&Array>;
}

impl Expr {
    /// Evaluate this expression given variable values from a solution.
    ///
    /// This allows computing the value of any expression — not just variables —
    /// after solving a problem. It mirrors CVXPY's `expr.value` attribute.
    ///
    /// # Example
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(3);
    /// let expr = norm2(&x);
    ///
    /// let solution = Problem::minimize(expr.clone())
    ///     .subject_to([x.ge(1.0)])
    ///     .solve()
    ///     .unwrap();
    ///
    /// let arr = expr.eval(&solution).unwrap();
    /// ```
    /// Evaluate this expression, returning an `Array`.
    ///
    /// Panics if a variable in the expression is not present in `ctx`.
    /// Use `eval()` for explicit error handling.
    ///
    /// The returned `Array` supports `[(row, col)]` indexing, matching the
    /// ergonomics of the old `&solution[&x]` API:
    ///
    /// ```
    /// use cvxrust::prelude::*;
    ///
    /// let x = variable(2);
    /// let solution = Problem::minimize(sum(&x))
    ///     .subject_to([x.ge(1.0)])
    ///     .solve()
    ///     .unwrap();
    ///
    /// let vals = x.value(&solution);
    /// println!("{}", vals[(0, 0)]);
    /// ```
    pub fn value<E: Evaluable>(&self, ctx: &E) -> Array {
        self.eval(ctx).expect("failed to evaluate expression")
    }

    /// Evaluate this expression, returning `Result<Array>`.
    pub fn eval<E: Evaluable>(&self, ctx: &E) -> crate::Result<Array> {
        match self {
            Expr::Variable(v) => ctx
                .get_variable_value(v.id)
                .cloned()
                .ok_or_else(|| crate::CvxError::InvalidProblem("Variable not in solution".into())),
            Expr::Constant(c) => Ok(c.value.clone()),

            Expr::Add(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                eval_add(av, bv)
            }
            Expr::Neg(a) => Ok(eval_neg(a.eval(ctx)?)),
            Expr::Mul(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                eval_mul(av, bv)
            }
            Expr::Sum(a, axis) => eval_sum(a.eval(ctx)?, *axis),
            Expr::Reshape(a, shape) => eval_reshape(a.eval(ctx)?, shape),
            Expr::Index(a, spec) => eval_index(a.eval(ctx)?, spec),
            Expr::VStack(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs.iter().map(|e| e.eval(ctx)).collect();
                eval_vstack(arrs?)
            }
            Expr::HStack(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs.iter().map(|e| e.eval(ctx)).collect();
                eval_hstack(arrs?)
            }
            Expr::Transpose(a) => Ok(eval_transpose(a.eval(ctx)?)),
            Expr::Trace(a) => Ok(eval_trace(a.eval(ctx)?)),
            Expr::MatMul(a, b) => {
                let av = a.eval(ctx)?;
                let bv = b.eval(ctx)?;
                eval_matmul(av, bv)
            }
            Expr::Norm1(a) => Ok(Array::Scalar(eval_norm1(&a.eval(ctx)?))),
            Expr::Norm2(a) => Ok(Array::Scalar(eval_norm2(&a.eval(ctx)?))),
            Expr::NormInf(a) => Ok(Array::Scalar(eval_norminf(&a.eval(ctx)?))),
            Expr::Abs(a) => Ok(eval_elementwise(a.eval(ctx)?, f64::abs)),
            Expr::Pos(a) => Ok(eval_elementwise(a.eval(ctx)?, |x| x.max(0.0))),
            Expr::NegPart(a) => Ok(eval_elementwise(a.eval(ctx)?, |x| (-x).max(0.0))),
            Expr::Maximum(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs.iter().map(|e| e.eval(ctx)).collect();
                eval_maximum(arrs?)
            }
            Expr::Minimum(exprs) => {
                let arrs: crate::Result<Vec<_>> = exprs.iter().map(|e| e.eval(ctx)).collect();
                eval_minimum(arrs?)
            }
            Expr::QuadForm(x, p) => {
                let xv = x.eval(ctx)?;
                let pv = p.eval(ctx)?;
                eval_quad_form(xv, pv)
            }
            Expr::SumSquares(a) => {
                let av = a.eval(ctx)?;
                Ok(Array::Scalar(eval_norm2(&av).powi(2)))
            }
            Expr::QuadOverLin(x, y) => {
                let xv = x.eval(ctx)?;
                let yv = y.eval(ctx)?;
                eval_quad_over_lin(xv, yv)
            }
            Expr::Exp(a) => Ok(eval_elementwise(a.eval(ctx)?, f64::exp)),
            Expr::Log(a) => Ok(eval_elementwise(a.eval(ctx)?, f64::ln)),
            Expr::Entropy(a) => Ok(eval_elementwise(a.eval(ctx)?, |x| {
                if x <= 0.0 { 0.0 } else { -x * x.ln() }
            })),
            Expr::Power(a, p) => {
                let p = *p;
                Ok(eval_elementwise(a.eval(ctx)?, move |x| x.powf(p)))
            }
            Expr::Cumsum(a, axis) => eval_cumsum(a.eval(ctx)?, *axis),
            Expr::Diag(a) => Ok(eval_diag(a.eval(ctx)?)),
        }
    }
}

// ---- Array arithmetic helpers ----

/// Convert Array to DMatrix (always 2D column-major).
fn arr_to_dense(a: Array) -> DMatrix<f64> {
    match a {
        Array::Dense(m) => m,
        Array::Scalar(v) => DMatrix::from_element(1, 1, v),
        Array::Sparse(m) => {
            let mut dense = DMatrix::zeros(m.nrows(), m.ncols());
            for (row, col, val) in m.triplet_iter() {
                dense[(row, col)] = *val;
            }
            dense
        }
    }
}

fn eval_add(a: Array, b: Array) -> crate::Result<Array> {
    match (a, b) {
        (Array::Scalar(x), Array::Scalar(y)) => Ok(Array::Scalar(x + y)),
        (Array::Scalar(s), b) => Ok(Array::Dense(arr_to_dense(b).map(|x| x + s))),
        (a, Array::Scalar(s)) => Ok(Array::Dense(arr_to_dense(a).map(|x| x + s))),
        (a, b) => {
            let am = arr_to_dense(a);
            let bm = arr_to_dense(b);
            if am.nrows() != bm.nrows() || am.ncols() != bm.ncols() {
                return Err(crate::CvxError::InvalidProblem(
                    "Shape mismatch in addition".into(),
                ));
            }
            Ok(Array::Dense(am + bm))
        }
    }
}

fn eval_neg(a: Array) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(-v),
        Array::Dense(m) => Array::Dense(-m),
        a @ Array::Sparse(_) => Array::Dense(-arr_to_dense(a)),
    }
}

fn eval_mul(a: Array, b: Array) -> crate::Result<Array> {
    match (a, b) {
        (Array::Scalar(x), Array::Scalar(y)) => Ok(Array::Scalar(x * y)),
        (Array::Scalar(s), b) => Ok(Array::Dense(arr_to_dense(b) * s)),
        (a, Array::Scalar(s)) => Ok(Array::Dense(arr_to_dense(a) * s)),
        (a, b) => {
            let am = arr_to_dense(a);
            let bm = arr_to_dense(b);
            if am.nrows() != bm.nrows() || am.ncols() != bm.ncols() {
                return Err(crate::CvxError::InvalidProblem(
                    "Shape mismatch in element-wise multiply".into(),
                ));
            }
            Ok(Array::Dense(am.component_mul(&bm)))
        }
    }
}

fn eval_matmul(a: Array, b: Array) -> crate::Result<Array> {
    let am = arr_to_dense(a);
    let bm = arr_to_dense(b);
    if am.ncols() != bm.nrows() {
        return Err(crate::CvxError::InvalidProblem(
            "Shape mismatch in matrix multiply".into(),
        ));
    }
    Ok(Array::Dense(am * bm))
}

fn eval_sum(a: Array, axis: Option<usize>) -> crate::Result<Array> {
    match axis {
        None => {
            let total: f64 = match &a {
                Array::Scalar(v) => *v,
                Array::Dense(m) => m.sum(),
                Array::Sparse(m) => m.values().iter().sum(),
            };
            Ok(Array::Scalar(total))
        }
        Some(0) => {
            // Sum along axis 0 (rows) → column vector of size ncols
            let m = arr_to_dense(a);
            let result = DMatrix::from_fn(m.ncols(), 1, |j, _| m.column(j).iter().sum::<f64>());
            Ok(Array::Dense(result))
        }
        Some(1) => {
            // Sum along axis 1 (cols) → column vector of size nrows
            let m = arr_to_dense(a);
            let result = DMatrix::from_fn(m.nrows(), 1, |i, _| m.row(i).iter().sum::<f64>());
            Ok(Array::Dense(result))
        }
        Some(ax) => Err(crate::CvxError::InvalidProblem(format!(
            "Invalid axis {} for sum",
            ax
        ))),
    }
}

fn eval_reshape(a: Array, shape: &Shape) -> crate::Result<Array> {
    let flat: Vec<f64> = match a {
        Array::Scalar(v) => vec![v],
        Array::Dense(m) => m.iter().cloned().collect(),
        Array::Sparse(m) => {
            let mut v = vec![0.0; m.nrows() * m.ncols()];
            for (r, c, val) in m.triplet_iter() {
                v[c * m.nrows() + r] = *val;
            }
            v
        }
    };
    let (rows, cols) = (shape.rows(), shape.cols());
    if flat.len() != rows * cols {
        return Err(crate::CvxError::InvalidProblem(
            "Reshape size mismatch".into(),
        ));
    }
    if shape.is_scalar() {
        Ok(Array::Scalar(flat[0]))
    } else {
        Ok(Array::Dense(DMatrix::from_vec(rows, cols, flat)))
    }
}

fn eval_index(a: Array, spec: &IndexSpec) -> crate::Result<Array> {
    let m = arr_to_dense(a);
    let nrows = m.nrows();
    let ncols = m.ncols();

    match spec.ranges.as_slice() {
        [Some((start, stop, step))] => {
            let indices: Vec<usize> = (*start..*stop).step_by(*step).collect();
            // For column vectors, index rows; otherwise index flat (column-major)
            let data: Vec<f64> = if ncols == 1 {
                indices.iter().map(|&i| m[(i, 0)]).collect()
            } else {
                indices
                    .iter()
                    .map(|&i| *m.iter().nth(i).unwrap_or(&0.0))
                    .collect()
            };
            if data.len() == 1 {
                Ok(Array::Scalar(data[0]))
            } else {
                Ok(Array::Dense(DMatrix::from_vec(data.len(), 1, data)))
            }
        }
        [None] => Ok(Array::Dense(m)),
        [row_spec, col_spec] => {
            let row_range: Vec<usize> = match row_spec {
                Some((s, e, step)) => (*s..*e).step_by(*step).collect(),
                None => (0..nrows).collect(),
            };
            let col_range: Vec<usize> = match col_spec {
                Some((s, e, step)) => (*s..*e).step_by(*step).collect(),
                None => (0..ncols).collect(),
            };
            let result = DMatrix::from_fn(row_range.len(), col_range.len(), |i, j| {
                m[(row_range[i], col_range[j])]
            });
            if result.nrows() == 1 && result.ncols() == 1 {
                Ok(Array::Scalar(result[(0, 0)]))
            } else {
                Ok(Array::Dense(result))
            }
        }
        _ => Err(crate::CvxError::InvalidProblem(
            "Unsupported index spec in eval".into(),
        )),
    }
}

fn eval_vstack(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Ok(Array::Scalar(0.0));
    }
    let mats: Vec<DMatrix<f64>> = arrays.into_iter().map(arr_to_dense).collect();
    let ncols = mats[0].ncols();
    let total_rows: usize = mats.iter().map(|m| m.nrows()).sum();
    let mut result = DMatrix::zeros(total_rows, ncols);
    let mut row_offset = 0;
    for m in mats {
        let nrows = m.nrows();
        result.rows_mut(row_offset, nrows).copy_from(&m);
        row_offset += nrows;
    }
    Ok(Array::Dense(result))
}

fn eval_hstack(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Ok(Array::Scalar(0.0));
    }
    let mats: Vec<DMatrix<f64>> = arrays.into_iter().map(arr_to_dense).collect();
    let nrows = mats[0].nrows();
    let total_cols: usize = mats.iter().map(|m| m.ncols()).sum();
    let mut result = DMatrix::zeros(nrows, total_cols);
    let mut col_offset = 0;
    for m in mats {
        let ncols = m.ncols();
        result.columns_mut(col_offset, ncols).copy_from(&m);
        col_offset += ncols;
    }
    Ok(Array::Dense(result))
}

fn eval_transpose(a: Array) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(v),
        Array::Dense(m) => Array::Dense(m.transpose()),
        a @ Array::Sparse(_) => Array::Dense(arr_to_dense(a).transpose()),
    }
}

fn eval_trace(a: Array) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(v),
        Array::Dense(m) => Array::Scalar(m.trace()),
        a @ Array::Sparse(_) => Array::Scalar(arr_to_dense(a).trace()),
    }
}

fn eval_norm1(a: &Array) -> f64 {
    match a {
        Array::Scalar(v) => v.abs(),
        Array::Dense(m) => m.iter().map(|x| x.abs()).sum(),
        Array::Sparse(m) => m.values().iter().map(|x| x.abs()).sum(),
    }
}

fn eval_norm2(a: &Array) -> f64 {
    match a {
        Array::Scalar(v) => v.abs(),
        Array::Dense(m) => m.iter().map(|x| x * x).sum::<f64>().sqrt(),
        Array::Sparse(m) => m.values().iter().map(|x| x * x).sum::<f64>().sqrt(),
    }
}

fn eval_norminf(a: &Array) -> f64 {
    match a {
        Array::Scalar(v) => v.abs(),
        Array::Dense(m) => m.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
        Array::Sparse(m) => m.values().iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
    }
}

fn eval_elementwise(a: Array, f: impl Fn(f64) -> f64) -> Array {
    match a {
        Array::Scalar(v) => Array::Scalar(f(v)),
        Array::Dense(m) => Array::Dense(m.map(f)),
        a @ Array::Sparse(_) => Array::Dense(arr_to_dense(a).map(f)),
    }
}

fn eval_maximum(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Err(crate::CvxError::InvalidProblem(
            "maximum of empty list".into(),
        ));
    }
    let mut iter = arrays.into_iter();
    let first = arr_to_dense(iter.next().unwrap());
    let result = iter.try_fold(first, |acc, a| {
        let m = arr_to_dense(a);
        if acc.nrows() != m.nrows() || acc.ncols() != m.ncols() {
            return Err(crate::CvxError::InvalidProblem(
                "Shape mismatch in maximum".into(),
            ));
        }
        Ok(acc.zip_map(&m, f64::max))
    })?;
    Ok(Array::Dense(result))
}

fn eval_minimum(arrays: Vec<Array>) -> crate::Result<Array> {
    if arrays.is_empty() {
        return Err(crate::CvxError::InvalidProblem(
            "minimum of empty list".into(),
        ));
    }
    let mut iter = arrays.into_iter();
    let first = arr_to_dense(iter.next().unwrap());
    let result = iter.try_fold(first, |acc, a| {
        let m = arr_to_dense(a);
        if acc.nrows() != m.nrows() || acc.ncols() != m.ncols() {
            return Err(crate::CvxError::InvalidProblem(
                "Shape mismatch in minimum".into(),
            ));
        }
        Ok(acc.zip_map(&m, f64::min))
    })?;
    Ok(Array::Dense(result))
}

fn eval_quad_form(x: Array, p: Array) -> crate::Result<Array> {
    let xm = arr_to_dense(x);
    let pm = arr_to_dense(p);
    if pm.nrows() != pm.ncols() || xm.nrows() != pm.nrows() {
        return Err(crate::CvxError::InvalidProblem(
            "Shape mismatch in quad_form".into(),
        ));
    }
    // x' P x
    let result = (xm.transpose() * &pm * &xm)[(0, 0)];
    Ok(Array::Scalar(result))
}

fn eval_quad_over_lin(x: Array, y: Array) -> crate::Result<Array> {
    let norm_sq = eval_norm2(&x).powi(2);
    let y_val = y.as_scalar().ok_or_else(|| {
        crate::CvxError::InvalidProblem("quad_over_lin: denominator must be scalar".into())
    })?;
    Ok(Array::Scalar(norm_sq / y_val))
}

fn eval_cumsum(a: Array, axis: Option<usize>) -> crate::Result<Array> {
    let mut m = arr_to_dense(a);
    match axis {
        None | Some(0) => {
            // Cumsum along rows
            for i in 1..m.nrows() {
                for j in 0..m.ncols() {
                    m[(i, j)] += m[(i - 1, j)];
                }
            }
        }
        Some(1) => {
            // Cumsum along columns
            for j in 1..m.ncols() {
                for i in 0..m.nrows() {
                    m[(i, j)] += m[(i, j - 1)];
                }
            }
        }
        Some(ax) => {
            return Err(crate::CvxError::InvalidProblem(format!(
                "Invalid axis {} for cumsum",
                ax
            )));
        }
    }
    Ok(Array::Dense(m))
}

fn eval_diag(a: Array) -> Array {
    let m = arr_to_dense(a);
    let (rows, cols) = (m.nrows(), m.ncols());
    if cols == 1 || rows == 1 {
        // Vector → diagonal matrix
        let n = rows.max(cols);
        let mut result = DMatrix::zeros(n, n);
        for i in 0..n {
            let v = if cols == 1 { m[(i, 0)] } else { m[(0, i)] };
            result[(i, i)] = v;
        }
        Array::Dense(result)
    } else {
        // Matrix → diagonal vector
        let n = rows.min(cols);
        let data: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
        Array::Dense(DMatrix::from_vec(n, 1, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, variable};
    use crate::prelude::*;
    use std::collections::HashMap;

    /// Simple test context using a HashMap of variable values.
    struct TestCtx(HashMap<ExprId, Array>);

    impl Evaluable for TestCtx {
        fn get_variable_value(&self, id: ExprId) -> Option<&Array> {
            self.0.get(&id)
        }
    }

    fn make_var_scalar(val: f64) -> (Expr, TestCtx) {
        let x = variable(());
        let id = x.variable_id().unwrap();
        let mut map = HashMap::new();
        map.insert(id, Array::Scalar(val));
        (x, TestCtx(map))
    }

    fn make_var_vec(vals: Vec<f64>) -> (Expr, TestCtx) {
        let n = vals.len();
        let x = variable(n);
        let id = x.variable_id().unwrap();
        let mut map = HashMap::new();
        map.insert(id, Array::from_vec(vals));
        (x, TestCtx(map))
    }

    #[test]
    fn test_eval_variable() {
        let (x, ctx) = make_var_scalar(3.0);
        assert_eq!(x.value(&ctx)[(0, 0)], 3.0);
    }

    #[test]
    fn test_eval_constant() {
        let c = constant(5.0);
        let (_, ctx) = make_var_scalar(0.0);
        assert_eq!(c.value(&ctx)[(0, 0)], 5.0);
    }

    #[test]
    fn test_eval_add_scalars() {
        let (x, ctx) = make_var_scalar(3.0);
        let expr = x.clone() + constant(2.0);
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_neg() {
        let (x, ctx) = make_var_scalar(3.0);
        let expr = -&x;
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v + 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_mul_scalar() {
        let (x, ctx) = make_var_scalar(4.0);
        let expr = &x * 2.5;
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_norm2() {
        let (x, ctx) = make_var_vec(vec![3.0, 4.0]);
        let expr = norm2(&x);
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_norm1() {
        let (x, ctx) = make_var_vec(vec![-1.0, 2.0, -3.0]);
        let expr = norm1(&x);
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sum() {
        let (x, ctx) = make_var_vec(vec![1.0, 2.0, 3.0]);
        let expr = sum(&x);
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sum_squares() {
        let (x, ctx) = make_var_vec(vec![1.0, 2.0, 3.0]);
        let expr = sum_squares(&x);
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_abs() {
        let (x, ctx) = make_var_scalar(-5.0);
        let expr = abs(&x);
        let v = expr.value(&ctx).as_scalar().unwrap();
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_exp_log() {
        let (x, ctx) = make_var_scalar(1.0);
        let e = exp(&x);
        let ev = e.value(&ctx).as_scalar().unwrap();
        assert!((ev - std::f64::consts::E).abs() < 1e-10);
        let l = log(&x);
        let lv = l.value(&ctx).as_scalar().unwrap();
        assert!(lv.abs() < 1e-10);
    }

    #[test]
    fn test_eval_with_solution() {
        let x = variable(());
        let obj = &x * 2.0;
        let solution = Problem::minimize(x.clone())
            .subject_to([x.ge(3.0)])
            .solve()
            .unwrap();
        // The variable x should be ~3.0, so 2*x ~= 6.0
        let v = obj.value(&solution).as_scalar().unwrap();
        assert!((v - 6.0).abs() < 1e-4);
    }
}
