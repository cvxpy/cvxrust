/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Basic Linear Programming Example
//!
//! This example demonstrates solving a simple linear program:
//!
//! ```text
//! maximize    c^T x
//! subject to  A x <= b, x >= 0
//! ```

use cvxrust::prelude::*;

fn main() {
    println!("=== Basic Linear Program ===\n");

    // Problem: Maximize profit = 3*x_1 + 2*x_2 + 5*x_3
    // subject to:
    //   2*x_1 + x_2 + 3*x_3 <= 100  (resource constraint 1)
    //   x_1 + 2*x_2 + x_3 <= 80     (resource constraint 2)
    //   x >= 0

    println!("Problem: Maximize 3*x1 + 2*x2 + 5*x3");
    println!("Subject to:");
    println!("  2*x1 + x2 + 3*x3 <= 100");
    println!("  x1 + 2*x2 + x3 <= 80");
    println!("  x >= 0\n");

    // Create variables
    let x = variable(3);

    // Coefficients
    let c = constant_vec(vec![3.0, 2.0, 5.0]);
    let a = constant_matrix(vec![2.0, 1.0, 3.0, 1.0, 2.0, 1.0], 2, 3);
    let b = constant_vec(vec![100.0, 80.0]);

    // Objective: maximize c^T x (minimize -c^T x)
    let objective = -dot(&c, &x);

    // Solve
    println!("Solving...");
    let solution = Problem::minimize(objective)
        .subject_to([constraint!((matmul(&a, &x)) <= b), constraint!(x >= 0.0)])
        .solve()
        .expect("Failed to solve");

    // Display results
    println!("\nResults:");
    println!("  Status: {:?}", solution.status);
    println!("  Optimal profit: {:.4}", -solution.value.unwrap());

    let x_vals = x.value(&solution);
    println!("  x1 = {:.4}", x_vals[(0, 0)]);
    println!("  x2 = {:.4}", x_vals[(1, 0)]);
    println!("  x3 = {:.4}", x_vals[(2, 0)]);
}
