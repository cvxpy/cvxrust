/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Least Squares Regression Example
//!
//! This example demonstrates linear regression:
//!
//! ```text
//! minimize    ||A x - b||_2^2
//! ```

use cvxrust::prelude::*;

fn main() {
    println!("=== Least Squares Regression ===\n");

    // Fit a line y = w_0 + w_1*x to data
    let a = constant_matrix(vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0], 5, 2);
    let b = constant_vec(vec![3.1, 5.2, 6.8, 9.1, 10.9]);

    println!("Fitting y = w0 + w1*x to 5 data points\n");

    // Unconstrained least squares
    let w = variable(2);
    let residual = matmul(&a, &w) - &b;

    println!("Solving unconstrained least squares...");
    let solution = Problem::minimize(sum_squares(&residual))
        .solve()
        .expect("Failed to solve");

    println!("\nUnconstrained Results:");
    println!("  Status: {:?}", solution.status);
    println!("  Optimal value: {:.6}", solution.value.unwrap());

    // Two equivalent ways to retrieve variable values:
    let w_vals = &solution[&w]; // index into solution directly
    let w_vals2 = w.value(&solution); // call .value() on the variable
    println!(
        "  w0 (intercept) = {:.6} / {:.6}",
        w_vals[(0, 0)],
        w_vals2[(0, 0)]
    );
    println!(
        "  w1 (slope)     = {:.6} / {:.6}",
        w_vals[(1, 0)],
        w_vals2[(1, 0)]
    );

    // Expression values: evaluate any sub-expression at the solution
    let fitted = matmul(&a, &w).value(&solution);
    println!(
        "  Fitted values: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        fitted[(0, 0)],
        fitted[(1, 0)],
        fitted[(2, 0)],
        fitted[(3, 0)],
        fitted[(4, 0)]
    );

    // Constrained least squares (w >= 0)
    println!("\n--- Constrained Least Squares (w >= 0) ---\n");

    let w2 = variable(2);
    let residual2 = matmul(&a, &w2) - &b;

    let solution2 = Problem::minimize(sum_squares(&residual2))
        .constraint(constraint!(w2 >= 0.0))
        .solve()
        .expect("Failed to solve");

    println!("Constrained Results:");
    println!("  Status: {:?}", solution2.status);
    println!("  Optimal value: {:.6}", solution2.value.unwrap());

    let w2_vals = w2.value(&solution2);
    println!("  w0 (intercept) = {:.6}", w2_vals[(0, 0)]);
    println!("  w1 (slope)     = {:.6}", w2_vals[(1, 0)]);
}
