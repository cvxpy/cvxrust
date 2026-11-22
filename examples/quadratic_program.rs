/*
Copyright 2026 CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/

//! Quadratic Programming Example
//!
//! This example demonstrates solving a quadratic program:
//!
//! minimize    (1/2) x'Px + q'x
//! subject to  Ax = b, x >= 0

use cvxrust::prelude::*;

fn main() {
    println!("=== Quadratic Programming ===\n");

    // Problem: Find point closest to [3, 2] satisfying x1 + x2 = 4, x >= 0
    // minimize (x1 - 3)^2 + (x2 - 2)^2

    println!("Problem: Find point closest to [3, 2]");
    println!("Subject to: x1 + x2 = 4, x >= 0\n");

    let x = variable(2);

    // Objective: ||x - [3, 2]||^2
    let target = constant_vec(vec![3.0, 2.0]);
    let objective = sum_squares(&(&x - &target));

    // Solve
    println!("Solving...");
    let solution = Problem::minimize(objective)
        .subject_to([constraint!((sum(&x)) == 4.0), constraint!(x >= 0.0)])
        .solve()
        .expect("Failed to solve");

    println!("\nResults:");
    println!("  Status: {:?}", solution.status);
    println!("  Optimal value: {:.6}", solution.value.unwrap());

    let x_vals = x.value(&solution);
    println!("  x1 = {:.6}", x_vals[(0, 0)]);
    println!("  x2 = {:.6}", x_vals[(1, 0)]);
    println!("  Sum: {:.6}", x_vals[(0, 0)] + x_vals[(1, 0)]);
    println!("  Distance: {:.6}", solution.value.unwrap().sqrt());
}
