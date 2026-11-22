/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Portfolio Optimization Example
//!
//! This example demonstrates Markowitz portfolio optimization:
//!
//! minimize    x' Σ x                (minimize risk)
//! subject to  μ' x >= target        (minimum return)
//!             sum(x) = 1            (fully invested)
//!             x >= 0                (long-only)

use cvxrust::prelude::*;

fn main() {
    println!("=== Portfolio Optimization ===\n");

    // 4 assets with different risk/return profiles
    let mu = constant_vec(vec![0.12, 0.10, 0.07, 0.05]);

    // Covariance matrix (risk)
    #[rustfmt::skip]
    let sigma = constant_matrix(vec![
        0.04,  0.01,  0.00, -0.01,
        0.01,  0.03,  0.00,  0.00,
        0.00,  0.00,  0.02,  0.00,
        -0.01,  0.00,  0.00,  0.01,
    ], 4, 4);

    println!("Assets: A, B, C, D");
    println!("Expected returns: [12%, 10%, 7%, 5%]");
    println!("Target return: 9%\n");

    // Portfolio weights
    let x = variable(4);

    // Objective: minimize portfolio variance
    let risk = quad_form(&x, &sigma);

    // Solve
    let target_return = 0.09;
    let solution = Problem::minimize(risk)
        .subject_to([
            constraint!((dot(&mu, &x)) >= target_return),
            constraint!((sum(&x)) == 1.0),
            constraint!(x >= 0.0),
        ])
        .solve()
        .expect("Failed to solve");

    // Results
    println!("Optimal Portfolio:");
    let portfolio = x.value(&solution);
    let assets = ["A", "B", "C", "D"];

    for i in 0..4 {
        println!("  Asset {}: {:.2}%", assets[i], portfolio[(i, 0)] * 100.0);
    }

    // Expression values: evaluate return and risk directly
    let actual_return = dot(&mu, &x).value(&solution).as_scalar().unwrap();
    println!(
        "  Actual return (expression eval): {:.2}%",
        actual_return * 100.0
    );

    let variance = solution.value.unwrap();
    let std_dev = variance.sqrt();
    println!("\nPortfolio Statistics:");
    println!("  Expected return: 9.00%");
    println!("  Risk (std dev): {:.2}%", std_dev * 100.0);
    println!("  Sharpe ratio: {:.4}", 0.09 / std_dev);

    // Dual variables (shadow prices)
    println!("\nDual Variables (Shadow Prices):");
    if let Some(dual_return) = solution.constraint_dual(0) {
        // The dual of the return constraint tells us the marginal cost
        // of increasing the return target (in terms of variance)
        println!("  Return constraint dual: {:.4}", dual_return);
        println!("  Interpretation: Increasing target return by 1% would");
        println!(
            "  increase portfolio variance by ~{:.4}",
            dual_return.abs() * 0.01
        );
    }

    // Efficient frontier
    println!("\n--- Efficient Frontier ---\n");

    for &target in &[0.06, 0.08, 0.10, 0.12] {
        let y = variable(4);
        let risk_y = quad_form(&y, &sigma);

        if let Ok(sol) = Problem::minimize(risk_y)
            .subject_to([
                constraint!((dot(&mu, &y)) >= target),
                constraint!((sum(&y)) == 1.0),
                constraint!(y >= 0.0),
            ])
            .solve()
        {
            let std = sol.value.unwrap().sqrt();
            println!(
                "  Return: {:.1}%  →  Risk: {:.2}%",
                target * 100.0,
                std * 100.0
            );
        }
    }

    println!("\nHigher returns require accepting higher risk!");
}
