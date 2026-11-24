# cvxrust

[![Crates.io](https://img.shields.io/crates/v/cvxrust.svg)](https://crates.io/crates/cvxrust)
[![Docs.rs](https://docs.rs/cvxrust/badge.svg)](https://docs.rs/cvxrust)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Disciplined Convex Programming (DCP) library for Rust, inspired by [CVXPY](https://www.cvxpy.org/).

cvxrust provides a domain-specific language for specifying convex optimization problems with automatic convexity verification and solving via the [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) solver.

## Features

- **DCP Verification**: Automatically verifies problem convexity using disciplined convex programming rules
- **Rich Atom Library**: Norms, quadratic forms, exponential/log, element-wise operations, and more
- **Native QP Support**: Quadratic programs are solved directly (not reformulated as SOCPs)
- **Intuitive Constraint Syntax**: Use `constraint!` macro for natural `>=`, `<=`, `==` notation
- **Dual Variables**: Access shadow prices and sensitivity information
- **Sparse Matrix Support**: Efficient handling of large-scale problems

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cvxrust = "0.1"
```

## Quick Start

```rust
use cvxrust::prelude::*;

// Minimize ||x||_2 subject to sum(x) = 1, x >= 0
let x = variable(5);

let solution = Problem::minimize(norm2(&x))
    .subject_to([
        constraint!((sum(&x)) == 1.0),
        constraint!(x >= 0.0),
    ])
    .solve()
    .unwrap();

println!("Optimal value: {}", solution.value.unwrap());
println!("x = {:?}", &solution[&x]);
```

## Building and Testing

```bash
# Build the library
cargo build

# Run all tests (159 tests)
cargo test

# Run a specific test
cargo test test_name

# Run tests with output visible
cargo test -- --nocapture

# Run examples
cargo run --example portfolio
cargo run --example least_squares
cargo run --example quadratic_program

# Lint with clippy
cargo clippy

# Generate documentation
cargo doc --open
```

## Supported Problem Classes

| Problem Type | Objective | Constraints |
|-------------|-----------|-------------|
| **LP** | Linear | Linear equality/inequality |
| **QP** | Quadratic (convex) | Linear equality/inequality |
| **SOCP** | Linear | Second-order cone |
| **Exp Cone** | Exponential/log | Exponential cone |
| **Power Cone** | Powers/geo mean | Power cone |
| **Mixed** | Any convex | Combination of above |

## Expression Atoms

### Affine Operations
- Arithmetic: `+`, `-`, `*` (scalar), `/` (scalar)
- Aggregation: `sum`, `sum_axis`, `cumsum`
- Structural: `reshape`, `flatten`, `vstack`, `hstack`, `transpose`, `diag`
- Linear algebra: `matmul`, `dot`, `trace`
- Indexing: `index`, `slice`

### Convex Atoms
- Norms: `norm1`, `norm2`, `norm_inf`, `norm`
- Element-wise: `abs`, `pos`, `neg_part`, `exp`
- Aggregation: `maximum`, `max2`
- Quadratic: `quad_form` (PSD), `sum_squares`, `quad_over_lin`
- Powers: `power` (`p >= 1` or `p < 0`)

### Concave Atoms
- Aggregation: `minimum`, `min2`
- Quadratic: `quad_form` (NSD)
- Logarithmic: `log`, `entropy`
- Powers: `power` (`0 < p < 1`), `sqrt`

## Variable Construction

```rust
// Simple vector variable
let x = variable(5);

// Matrix variable
let X = variable((3, 4));

// Scalar variable
let t = variable(());

// Named variable with bounds
let x = VariableBuilder::vector(5)
    .name("x")
    .nonneg()  // x >= 0
    .build();
```

## Constraints

cvxrust provides the `constraint!` macro for natural constraint syntax:

```rust
use cvxrust::prelude::*;

let x = variable(5);

// Using constraint! macro (recommended)
let constraints = [
    constraint!(x >= 0.0),           // x >= 0
    constraint!(x <= 10.0),          // x <= 10
    constraint!((sum(&x)) == 1.0),   // sum(x) = 1
];

// Method syntax also available
let c1 = x.ge(0.0);   // x >= 0
let c2 = x.le(10.0);  // x <= 10
let c3 = sum(&x).eq(1.0);

// Second-order cone constraint: ||x||_2 <= t
let t = variable(());
let soc = Constraint::soc(t, x);
```

## Accessing Solutions

```rust
let solution = problem.solve()?;

// Check status
assert_eq!(solution.status, SolveStatus::Optimal);

// Get optimal objective value
let value = solution.value.unwrap();

// Get variable values using indexing
let x_val = &solution[&x];

// Get scalar value
let t_val = solution.value(&t);

// Access dual variables (shadow prices)
let duals = solution.duals();
let dual_0 = solution.constraint_dual(0);  // Dual for first constraint
```

## Examples

### Least Squares
```rust
let x = variable(n);
let residual = &A * &x - &b;
let solution = Problem::minimize(sum_squares(&residual)).solve()?;
```

### L1 Regularized Regression (Lasso)
```rust
let x = variable(n);
let loss = sum_squares(&(&A * &x - &b));
let reg = norm1(&x);
let solution = Problem::minimize(loss + lambda * reg).solve()?;
```

### Portfolio Optimization
```rust
let w = variable(n);
let risk = quad_form(&w, &sigma);  // w^T Sigma w

let solution = Problem::minimize(risk)
    .subject_to([
        constraint!((dot(&mu, &w)) >= target_return),
        constraint!((sum(&w)) == 1.0),
        constraint!(w >= 0.0),
    ])
    .solve()?;

// Access dual variable for return constraint (shadow price)
if let Some(dual) = solution.constraint_dual(0) {
    println!("Marginal cost of increasing return: {}", dual);
}
```

### Logistic Regression
```rust
let theta = variable(n);
// log(1 + exp(-y * X theta)) using log-sum-exp
let solution = Problem::minimize(log_sum_exp_loss)
    .subject_to([constraint!((norm2(&theta)) <= C)])
    .solve()?;
```

## DCP Rules

cvxrust enforces [Disciplined Convex Programming](https://dcp.stanford.edu/) rules:

**Objective:**
- `minimize(convex)` or `maximize(concave)`

**Constraints:**
- `convex <= concave`
- `concave >= convex`
- `affine == affine`

Curvature is determined by DCP composition rules - for example, `convex + convex = convex`, and `nonneg * convex = convex`. See [dcp.stanford.edu](https://dcp.stanford.edu/) for a complete reference.

Problems that violate DCP rules will return a `DcpError`.

## Architecture

```
Expression -> DCP Verification -> Canonicalization -> Matrix Stuffing -> Clarabel -> Solution
```

1. Build expression trees using atoms and operator overloads
2. Verify convexity via curvature and sign propagation
3. Transform to canonical form (LinExpr/QuadExpr + cone constraints)
4. Stuff into sparse matrices P, q, A, b
5. Solve with Clarabel and map solution back to variables

## License

Apache 2.0
