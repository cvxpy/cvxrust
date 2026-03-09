use cvxrust::prelude::*;

/// sum_squares(Ax - b) must be canonicalized as ||r||^2, not ||r||_2.
/// The old bug used a plain SOC `||r||_2 <= t`, which minimized the L2 norm
/// instead of its square, causing solution.value to be off by a square root.

#[test]
fn test_sum_squares_scalar_constrained() {
    // minimize (x - 3)^2  s.t.  x <= 1
    // True optimum: x* = 1,  obj* = (1 - 3)^2 = 4
    // Old bug would report: |1 - 3| = 2
    let x = variable(());
    let residual = x.clone() - constant(3.0);

    let sol = Problem::minimize(sum_squares(&residual))
        .constraint(constraint!(x <= 1.0))
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let eval_sq = sum_squares(&residual).value(&sol).as_scalar().unwrap();
    let eval_norm = norm2(&residual).value(&sol).as_scalar().unwrap();

    assert!(
        (reported - 4.0).abs() < 1e-4,
        "objective should be 4.0, got {reported}"
    );
    assert!(
        (reported - eval_sq).abs() < 1e-4,
        "reported obj must equal sum_squares evaluated at solution"
    );
    // Sanity check: the L2 norm (sqrt(4) = 2) is clearly different from the correct answer
    assert!((eval_norm - 2.0).abs() < 1e-4);
    assert!(
        (reported - eval_norm).abs() > 0.5,
        "objective must not equal the L2 norm (old bug)"
    );
}

#[test]
fn test_sum_squares_vector_constrained() {
    // minimize ||x - [3, 4]||^2  s.t.  x <= 2
    // True optimum: x* = [2, 2],  obj* = (2-3)^2 + (2-4)^2 = 5
    // Old bug would report: sqrt(5) ≈ 2.236
    let x = variable(2);
    let residual = x.clone() - constant_vec(vec![3.0, 4.0]);

    let sol = Problem::minimize(sum_squares(&residual))
        .constraint(constraint!(x <= 2.0))
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let eval_sq = sum_squares(&residual).value(&sol).as_scalar().unwrap();
    let eval_norm = norm2(&residual).value(&sol).as_scalar().unwrap();

    assert!(
        (reported - 5.0).abs() < 1e-4,
        "objective should be 5.0, got {reported}"
    );
    assert!(
        (reported - eval_sq).abs() < 1e-4,
        "reported obj must equal sum_squares evaluated at solution"
    );
    // Sanity check: the L2 norm (sqrt(5) ≈ 2.236) is clearly different
    assert!((eval_norm - 5f64.sqrt()).abs() < 1e-4);
    assert!(
        (reported - eval_norm).abs() > 0.5,
        "objective must not equal the L2 norm (old bug)"
    );
}

#[test]
fn test_sum_squares_matmul_constrained() {
    // minimize ||Ax - b||^2  s.t.  x <= 1
    // A = [[1], [1]] (2x1), b = [2, 4], x scalar
    // Unconstrained LS: x* = 3, obj* = (3-2)^2 + (3-4)^2 = 2
    // With x <= 1: x* = 1, residual = [-1, -3], obj* = 1 + 9 = 10
    // Old bug would report: sqrt(10) ≈ 3.162
    let a = constant_matrix(vec![1.0, 1.0], 2, 1);
    let b = constant_vec(vec![2.0, 4.0]);
    let x = variable(());
    let residual = matmul(&a, &x) - &b;

    let sol = Problem::minimize(sum_squares(&residual))
        .constraint(constraint!(x <= 1.0))
        .solve()
        .unwrap();

    let reported = sol.value.unwrap();
    let eval_sq = sum_squares(&residual).value(&sol).as_scalar().unwrap();
    let eval_norm = norm2(&residual).value(&sol).as_scalar().unwrap();

    assert!(
        (reported - 10.0).abs() < 1e-4,
        "objective should be 10.0, got {reported}"
    );
    assert!(
        (reported - eval_sq).abs() < 1e-4,
        "reported obj must equal sum_squares evaluated at solution"
    );
    assert!((eval_norm - 10f64.sqrt()).abs() < 1e-4);
    assert!(
        (reported - eval_norm).abs() > 0.5,
        "objective must not equal the L2 norm (old bug)"
    );
}
