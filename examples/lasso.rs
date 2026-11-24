/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! LASSO (L1-Regularized Regression) Example
//!
//! This example demonstrates L1-regularized least squares:
//!
//! ```text
//! minimize    ||A x - b||_2^2 + lambda * ||x||_1
//! ```
//!
//! L1 regularization encourages sparse solutions.

use cvxrust::prelude::*;

fn main() {
    println!("=== LASSO Regression ===\n");

    // Sparse regression problem: only 3 out of 10 coefficients are important
    #[rustfmt::skip]
    let a = constant_matrix(vec![
        1.0, 0.5, 0.2, 0.8, 0.1, 0.3, 0.4, 0.6, 0.2, 0.1,
        0.8, 0.3, 0.1, 0.9, 0.2, 0.4, 0.3, 0.7, 0.1, 0.2,
        0.6, 0.4, 0.3, 0.7, 0.3, 0.2, 0.5, 0.5, 0.3, 0.3,
        0.9, 0.2, 0.4, 0.6, 0.1, 0.5, 0.2, 0.8, 0.4, 0.1,
        0.7, 0.6, 0.2, 0.5, 0.4, 0.1, 0.6, 0.4, 0.2, 0.4,
        0.5, 0.1, 0.5, 0.4, 0.2, 0.6, 0.1, 0.9, 0.1, 0.2,
    ], 6, 10);

    // True model: y = 3*x_1 + 2*x_4 + 1*x_8 (sparse!)
    let b = constant_vec(vec![4.6, 4.9, 3.7, 4.7, 3.5, 3.2]);

    println!("Problem: Recover sparse coefficients (6 samples, 10 features)");
    println!("True model: y = 3*x1 + 2*x4 + 1*x8\n");

    // Standard least squares
    println!("--- Least Squares (lambda = 0) ---\n");

    let x_ls = variable(10);
    let residual_ls = matmul(&a, &x_ls) - &b;

    let solution_ls = Problem::minimize(sum_squares(&residual_ls))
        .solve()
        .expect("Failed to solve");

    let x_ls_vals = x_ls.value(&solution_ls);
    println!("Coefficients:");
    for i in 0..10 {
        println!("  x{}: {:.6}", i + 1, x_ls_vals[(i, 0)]);
    }
    println!("  Objective: {:.6}", solution_ls.value.unwrap());

    // LASSO with moderate regularization
    println!("\n--- LASSO (lambda = 0.5) ---\n");

    let lambda = 0.5;
    let x = variable(10);
    let residual = matmul(&a, &x) - &b;
    let objective = sum_squares(&residual) + constant(lambda) * norm1(&x);

    let solution = Problem::minimize(objective)
        .solve()
        .expect("Failed to solve");

    let x_vals = x.value(&solution);
    println!("Coefficients:");
    for i in 0..10 {
        let val = x_vals[(i, 0)];
        let marker = if val.abs() > 0.1 { " <--" } else { "" };
        println!("  x{}: {:.6}{}", i + 1, val, marker);
    }
    println!("  Objective: {:.6}", solution.value.unwrap());

    // Expression values: inspect individual terms
    let fit_err = sum_squares(&residual).value(&solution).as_scalar().unwrap();
    let l1_pen = norm1(&x).value(&solution).as_scalar().unwrap();
    println!("  Fit error (expression eval):   {:.6}", fit_err);
    println!("  L1 penalty (expression eval):  {:.6}", l1_pen);

    // LASSO with strong regularization
    println!("\n--- LASSO (lambda = 1.0) ---\n");

    let lambda2 = 1.0;
    let x2 = variable(10);
    let residual2 = matmul(&a, &x2) - &b;
    let objective2 = sum_squares(&residual2) + constant(lambda2) * norm1(&x2);

    let solution2 = Problem::minimize(objective2)
        .solve()
        .expect("Failed to solve");

    let x2_vals = x2.value(&solution2);
    println!("Coefficients:");
    for i in 0..10 {
        let val = x2_vals[(i, 0)];
        let marker = if val.abs() > 0.1 { " <--" } else { "" };
        println!("  x{}: {:.6}{}", i + 1, val, marker);
    }
    println!("  Objective: {:.6}", solution2.value.unwrap());
    println!("\nConclusion: LASSO identifies the sparse structure!");
}
