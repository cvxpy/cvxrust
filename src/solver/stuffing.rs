//! Matrix stuffing: converts canonicalized expressions to solver format.
//!
//! This module builds the matrices (P, q, A, b) and cone specifications
//! required by Clarabel from the canonicalized problem.

use std::collections::HashMap;
use std::ops::Range;

use nalgebra_sparse::CscMatrix;

use crate::canon::{ConeConstraint, LinExpr, QuadExpr};
use crate::expr::{ExprId, Shape};
use crate::sparse::{csc_from_triplets, csc_scale};

/// Cone dimensions for Clarabel.
#[derive(Debug, Clone, Default)]
pub struct ConeDims {
    /// Number of zero cone (equality) constraints.
    pub zero: usize,
    /// Number of nonnegative cone constraints.
    pub nonneg: usize,
    /// Second-order cone dimensions (each entry is the cone dimension).
    pub soc: Vec<usize>,
    /// Number of exponential cones (each is 3D).
    pub exp: usize,
    /// Power cone alpha values (each cone is 3D with its own alpha).
    pub power: Vec<f64>,
}

impl ConeDims {
    /// Total number of constraint rows.
    pub fn total(&self) -> usize {
        self.zero
            + self.nonneg
            + self.soc.iter().sum::<usize>()
            + (self.exp * 3)
            + (self.power.len() * 3)
    }
}

/// Mapping from variable IDs to column indices in the optimization variable.
#[derive(Debug, Clone)]
pub struct VariableMap {
    /// Map from variable ID to (start_col, size).
    pub id_to_col: HashMap<ExprId, (usize, usize)>,
    /// Original variable shapes, used when unpacking solver results.
    pub id_to_shape: HashMap<ExprId, Shape>,
    /// Total number of optimization variables.
    pub total_vars: usize,
}

impl VariableMap {
    /// Create from a list of (variable_id, shape) pairs.
    pub fn from_vars(vars: &[(ExprId, Shape)]) -> Self {
        let mut id_to_col = HashMap::new();
        let mut id_to_shape = HashMap::new();
        let mut offset = 0;

        for (var_id, shape) in vars {
            let size = shape.size();
            id_to_col.insert(*var_id, (offset, size));
            id_to_shape.insert(*var_id, shape.clone());
            offset += size;
        }

        VariableMap {
            id_to_col,
            id_to_shape,
            total_vars: offset,
        }
    }

    /// Get the column range for a variable.
    pub fn get(&self, var_id: ExprId) -> Option<(usize, usize)> {
        self.id_to_col.get(&var_id).copied()
    }

    /// Get the original shape for a variable.
    pub fn shape(&self, var_id: ExprId) -> Option<&Shape> {
        self.id_to_shape.get(&var_id)
    }
}

/// Stuffed problem ready for Clarabel.
#[derive(Debug)]
pub struct StuffedProblem {
    /// Quadratic cost matrix P (n x n, symmetric).
    pub p: CscMatrix<f64>,
    /// Linear cost vector q (n).
    pub q: Vec<f64>,
    /// Constraint matrix A (m x n).
    pub a: CscMatrix<f64>,
    /// Constraint vector b (m).
    pub b: Vec<f64>,
    /// Cone dimensions.
    pub cone_dims: ConeDims,
    /// Solver row ranges for input cone constraints, in their original order.
    pub constraint_rows: Vec<Range<usize>>,
    /// Variable mapping for solution recovery.
    pub var_map: VariableMap,
    /// Constant offset in objective.
    pub objective_offset: f64,
}

/// Build the stuffed problem from canonicalized components.
pub fn stuff_problem(
    objective: &QuadExpr,
    constraints: &[ConeConstraint],
    variables: &[(ExprId, Shape)],
) -> StuffedProblem {
    let var_map = VariableMap::from_vars(variables);

    // Build objective: q vector and P matrix
    let (p, q) = stuff_objective(objective, &var_map);

    // Build constraints: A matrix, b vector, cone dims
    let (a, b, cone_dims, constraint_rows) = stuff_constraints(constraints, &var_map);

    StuffedProblem {
        p,
        q,
        a,
        b,
        cone_dims,
        constraint_rows,
        var_map,
        objective_offset: objective.constant,
    }
}

/// Stuff the objective into P and q.
fn stuff_objective(objective: &QuadExpr, var_map: &VariableMap) -> (CscMatrix<f64>, Vec<f64>) {
    let n = var_map.total_vars;

    // Build q vector from linear term
    // The coefficient matrix for an objective has shape (output_size x var_size).
    // For a scalar objective, it's (1 x var_size), so each column corresponds to
    // a variable component. We need to sum over rows (usually just row 0 for scalar).
    let mut q = vec![0.0; n];
    for (var_id, coeff) in &objective.linear.coeffs {
        if let Some((start, size)) = var_map.get(*var_id) {
            for (_row, col, val) in coeff.triplet_iter() {
                // col is the variable component index within this variable
                if col < size {
                    q[start + col] += *val;
                }
            }
        }
    }

    // Build P matrix from quadratic terms
    let mut p_rows = Vec::new();
    let mut p_cols = Vec::new();
    let mut p_vals = Vec::new();

    for ((var_i, var_j), coeff) in &objective.quad_coeffs {
        if let (Some((start_i, _)), Some((start_j, _))) = (var_map.get(*var_i), var_map.get(*var_j))
        {
            for (row, col, val) in coeff.triplet_iter() {
                let global_row = start_i + row;
                let global_col = start_j + col;
                // P should be symmetric; Clarabel expects upper triangle only.
                // Since P is already symmetric, only keep upper triangle entries
                // (don't swap lower entries - they would double-count).
                if global_row <= global_col {
                    p_rows.push(global_row);
                    p_cols.push(global_col);
                    p_vals.push(*val);
                }
                // Skip lower triangle entries - the upper triangle already has them
            }
        }
    }

    let p = csc_from_triplets(n, n, p_rows, p_cols, p_vals);

    // Clarabel uses objective (1/2) x' P x + q' x, so we scale P by 2
    // to get our intended objective x' P x + q' x
    let p_scaled = csc_scale(&p, 2.0);

    (p_scaled, q)
}

/// Stuff constraints into A, b, and cone dims.
fn stuff_constraints(
    constraints: &[ConeConstraint],
    var_map: &VariableMap,
) -> (CscMatrix<f64>, Vec<f64>, ConeDims, Vec<Range<usize>>) {
    let n = var_map.total_vars;

    // Separate constraints by type
    let mut zeros = Vec::new();
    let mut nonnegs = Vec::new();
    let mut socs = Vec::new();
    let mut exps = Vec::new();
    let mut powers = Vec::new();

    for (idx, c) in constraints.iter().enumerate() {
        match c {
            ConeConstraint::Zero { a } => zeros.push((idx, a)),
            ConeConstraint::NonNeg { a } => nonnegs.push((idx, a)),
            ConeConstraint::SOC { t, x } => socs.push((idx, t, x)),
            ConeConstraint::ExpCone { x, y, z } => exps.push((idx, x, y, z)),
            ConeConstraint::PowerCone { x, y, z, alpha } => powers.push((idx, x, y, z, *alpha)),
        }
    }

    // Calculate dimensions
    let zero_rows: usize = zeros.iter().map(|(_, e)| e.size()).sum();
    let nonneg_rows: usize = nonnegs.iter().map(|(_, e)| e.size()).sum();
    let soc_dims: Vec<usize> = socs.iter().map(|(_, t, x)| t.size() + x.size()).collect();
    let soc_rows: usize = soc_dims.iter().sum();
    let exp_rows = exps.len() * 3; // Each exp cone is 3D (x, y, z)
    let power_alphas: Vec<f64> = powers.iter().map(|(_, _, _, _, alpha)| *alpha).collect();
    let power_rows = powers.len() * 3; // Each power cone is 3D (x, y, z)

    let total_rows = zero_rows + nonneg_rows + soc_rows + exp_rows + power_rows;

    let cone_dims = ConeDims {
        zero: zero_rows,
        nonneg: nonneg_rows,
        soc: soc_dims,
        exp: exps.len(),
        power: power_alphas,
    };

    // Build A and b
    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();
    let mut b = vec![0.0; total_rows];
    let mut constraint_rows = vec![0..0; constraints.len()];

    let mut row_offset = 0;

    // Zero cone (equalities): expr = 0
    // In Clarabel form Ax = b, we have Ax + const = 0, so Ax = -const
    for (idx, expr) in zeros {
        constraint_rows[idx] = row_offset..row_offset + expr.size();
        stuff_linear_expr(
            expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            false, // No negation for equality
        );
        row_offset += expr.size();
    }

    // Nonnegative cone: expr >= 0
    // In Clarabel form Ax + s = b, s >= 0 gives Ax <= b
    // We want expr = Ax + const >= 0, i.e., -Ax <= const
    for (idx, expr) in nonnegs {
        constraint_rows[idx] = row_offset..row_offset + expr.size();
        stuff_linear_expr(
            expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for inequality
        );
        row_offset += expr.size();
    }

    // SOC: ||x||_2 <= t, represented as [t; x] in K_soc
    // In Clarabel: s in K_soc means s[0] >= ||s[1:]||_2
    // We want [t_expr; x_expr] in K_soc
    // So s = [t_expr; x_expr], meaning -A @ vars + s = const
    for (idx, t_expr, x_expr) in socs {
        constraint_rows[idx] = row_offset..row_offset + t_expr.size() + x_expr.size();
        // t part - negate to get s = t_expr
        stuff_linear_expr(
            t_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for SOC
        );
        row_offset += t_expr.size();

        // x part - negate to get s = x_expr
        stuff_linear_expr(
            x_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for SOC
        );
        row_offset += x_expr.size();
    }

    // Exponential cone: (x, y, z) in K_exp
    // Clarabel form: s in K_exp means y*exp(x/y) <= z with y >= 0
    // Variable order: [x; y; z]
    // Negate to get s = [x_expr; y_expr; z_expr]
    for (idx, x_expr, y_expr, z_expr) in exps {
        constraint_rows[idx] =
            row_offset..row_offset + x_expr.size() + y_expr.size() + z_expr.size();
        // x part
        stuff_linear_expr(
            x_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for exp cone
        );
        row_offset += x_expr.size();

        // y part
        stuff_linear_expr(
            y_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for exp cone
        );
        row_offset += y_expr.size();

        // z part
        stuff_linear_expr(
            z_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for exp cone
        );
        row_offset += z_expr.size();
    }

    // Power cone: (x, y, z) in K_pow(alpha)
    // Clarabel form: s in K_pow(alpha) means x^alpha * y^(1-alpha) >= |z| with x, y >= 0
    // Variable order: [x; y; z]
    // Negate to get s = [x_expr; y_expr; z_expr]
    for (idx, x_expr, y_expr, z_expr, _alpha) in powers {
        constraint_rows[idx] =
            row_offset..row_offset + x_expr.size() + y_expr.size() + z_expr.size();
        // x part
        stuff_linear_expr(
            x_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for power cone
        );
        row_offset += x_expr.size();

        // y part
        stuff_linear_expr(
            y_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for power cone
        );
        row_offset += y_expr.size();

        // z part
        stuff_linear_expr(
            z_expr,
            var_map,
            row_offset,
            &mut a_rows,
            &mut a_cols,
            &mut a_vals,
            &mut b,
            true, // Negate for power cone
        );
        row_offset += z_expr.size();
    }

    let a = csc_from_triplets(total_rows, n, a_rows, a_cols, a_vals);

    (a, b, cone_dims, constraint_rows)
}

/// Stuff a single linear expression into A and b.
///
/// The LinExpr represents: expr = sum_i(A_i * x_i) + constant
///
/// For Zero cone (equality): expr = 0
///   - Clarabel form: Ax = b
///   - We have: Ax + constant = 0, so Ax = -constant
///   - A = coeffs, b = -constant
///
/// For NonNeg cone: expr >= 0
///   - Clarabel form: Ax + s = b, s >= 0 gives Ax <= b
///   - We want: Ax + constant >= 0, i.e., -Ax <= constant
///   - A = -coeffs, b = constant
///
/// This function uses the convention where negate=false for Zero cone
/// and negate=true for NonNeg cone (handled by caller negating the expr).
#[allow(clippy::too_many_arguments)]
fn stuff_linear_expr(
    expr: &LinExpr,
    var_map: &VariableMap,
    row_offset: usize,
    a_rows: &mut Vec<usize>,
    a_cols: &mut Vec<usize>,
    a_vals: &mut Vec<f64>,
    b: &mut [f64],
    negate: bool,
) {
    let sign = if negate { -1.0 } else { 1.0 };

    // Add coefficients to A (possibly negated)
    for (var_id, coeff) in &expr.coeffs {
        if let Some((col_start, _)) = var_map.get(*var_id) {
            for (row, col, val) in coeff.triplet_iter() {
                a_rows.push(row_offset + row);
                a_cols.push(col_start + col);
                a_vals.push(*val * sign);
            }
        }
    }

    // Add constant to b
    // For Zero: Ax + const = 0 => Ax = -const, so b = -const
    // For NonNeg: -Ax <= const, so b = const (when negate=true, we use +const)
    let size = expr.size();
    for i in 0..expr.constant.nrows() {
        for j in 0..expr.constant.ncols() {
            let idx = i + j * expr.constant.nrows();
            if idx < size {
                let const_val = expr.constant[(i, j)];
                // For Zero (negate=false): b = -constant
                // For NonNeg (negate=true): b = constant
                b[row_offset + idx] = if negate { const_val } else { -const_val };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ExprId;
    use nalgebra::DMatrix;

    #[test]
    fn test_constraint_rows_recover_original_cone_constants() {
        // Deliberately mix cone types and repeat cones with different row counts.
        let constraints = vec![
            ConeConstraint::PowerCone {
                x: LinExpr::scalar(31.0),
                y: LinExpr::scalar(32.0),
                z: LinExpr::scalar(33.0),
                alpha: 0.5,
            },
            ConeConstraint::NonNeg {
                a: LinExpr::constant(DMatrix::from_column_slice(2, 2, &[11.0, 12.0, 13.0, 14.0])),
            },
            ConeConstraint::SOC {
                t: LinExpr::scalar(21.0),
                x: LinExpr::constant(DMatrix::from_column_slice(2, 1, &[22.0, 23.0])),
            },
            ConeConstraint::ExpCone {
                x: LinExpr::scalar(41.0),
                y: LinExpr::scalar(42.0),
                z: LinExpr::scalar(43.0),
            },
            ConeConstraint::Zero {
                a: LinExpr::constant(DMatrix::from_column_slice(
                    2,
                    3,
                    &[51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                )),
            },
            ConeConstraint::NonNeg {
                a: LinExpr::scalar(61.0),
            },
            ConeConstraint::SOC {
                t: LinExpr::scalar(71.0),
                x: LinExpr::constant(DMatrix::from_column_slice(3, 1, &[72.0, 73.0, 74.0])),
            },
        ];
        let objective = QuadExpr::from_linear(LinExpr::scalar(0.0));
        let stuffed = stuff_problem(&objective, &constraints, &[]);

        assert_eq!(stuffed.constraint_rows.len(), constraints.len());
        let mut covered = vec![false; stuffed.b.len()];
        for (constraint, rows) in constraints.iter().zip(&stuffed.constraint_rows) {
            let expected: Vec<f64> = match constraint {
                ConeConstraint::Zero { a } => a.constant.iter().map(|value| -value).collect(),
                ConeConstraint::NonNeg { a } => a.constant.iter().copied().collect(),
                ConeConstraint::SOC { t, x } => t
                    .constant
                    .iter()
                    .chain(x.constant.iter())
                    .copied()
                    .collect(),
                ConeConstraint::ExpCone { x, y, z } | ConeConstraint::PowerCone { x, y, z, .. } => {
                    x.constant
                        .iter()
                        .chain(y.constant.iter())
                        .chain(z.constant.iter())
                        .copied()
                        .collect()
                }
            };
            assert_eq!(
                &stuffed.b[rows.clone()],
                expected.as_slice(),
                "{constraint:?}"
            );
            for row in rows.clone() {
                assert!(
                    !covered[row],
                    "solver row {row} belongs to multiple constraints"
                );
                covered[row] = true;
            }
        }
        assert!(covered.iter().all(|&covered| covered));
    }

    #[test]
    fn test_variable_map() {
        let vars = vec![
            (ExprId::new(), Shape::vector(3)),
            (ExprId::new(), Shape::vector(2)),
        ];
        let map = VariableMap::from_vars(&vars);
        assert_eq!(map.total_vars, 5);
    }

    #[test]
    fn test_cone_dims() {
        let dims = ConeDims {
            zero: 2,
            nonneg: 3,
            soc: vec![4, 5],
            exp: 0,
            power: vec![],
        };
        assert_eq!(dims.total(), 14);
    }

    #[test]
    fn test_cone_dims_with_exp_power() {
        let dims = ConeDims {
            zero: 2,
            nonneg: 3,
            soc: vec![4, 5],
            exp: 2,                // 2 exp cones = 6 rows
            power: vec![0.5, 0.7], // 2 power cones = 6 rows
        };
        // 2 + 3 + 4 + 5 + 6 + 6 = 26
        assert_eq!(dims.total(), 26);
    }
}
