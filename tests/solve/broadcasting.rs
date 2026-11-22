//! Broadcasting in affine objectives and constraints.

use super::TOL;
use cvxrust::prelude::*;

#[test]
fn test_broadcasted_affine_constraints_public_workflow() {
    let x = variable((2, 3));
    let row_cap = variable((1, 3));
    let col_floor = variable((2, 1));

    let prob = Problem::maximize(sum(&x))
        .subject_to([
            row_cap.eq(constant_matrix(vec![2.0, 3.0, 4.0], 1, 3)),
            col_floor.eq(constant_matrix(vec![1.0, 2.0], 2, 1)),
            x.le(row_cap.clone()),
            x.ge(col_floor.clone()),
        ])
        .build();

    assert!(prob.is_dcp());

    let solution = prob.solve().expect("broadcasted problem should solve");
    let value = solution.value.expect("should have objective");
    assert!(
        (value - 18.0).abs() < TOL,
        "broadcasted constraints: expected 18, got {}",
        value
    );

    let x_vals = &solution[&x];
    for i in 0..2 {
        assert!((x_vals[(i, 0)] - 2.0).abs() < TOL);
        assert!((x_vals[(i, 1)] - 3.0).abs() < TOL);
        assert!((x_vals[(i, 2)] - 4.0).abs() < TOL);
    }
}

#[test]
fn test_row_affine_broadcasts_in_matrix_constraint() {
    let x = variable((2, 3));
    let row = variable((1, 3));
    let row_values = constant_matrix(vec![1.0, 2.0, 3.0], 1, 3);

    let sol = Problem::maximize(sum(&x))
        .subject_to([x.le(row.clone()), row.eq(row_values), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 12.0).abs() < TOL);

    if let Array::Dense(x_vals) = x.value(&sol) {
        for i in 0..2 {
            assert!((x_vals[(i, 0)] - 1.0).abs() < TOL);
            assert!((x_vals[(i, 1)] - 2.0).abs() < TOL);
            assert!((x_vals[(i, 2)] - 3.0).abs() < TOL);
        }
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_column_affine_broadcasts_in_matrix_constraint() {
    let x = variable((2, 3));
    let col = variable((2, 1));
    let col_values = constant_matrix(vec![1.0, 2.0], 2, 1);

    let sol = Problem::maximize(sum(&x))
        .subject_to([x.le(col.clone()), col.eq(col_values), x.ge(0.0)])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 9.0).abs() < TOL);

    if let Array::Dense(x_vals) = x.value(&sol) {
        for j in 0..3 {
            assert!((x_vals[(0, j)] - 1.0).abs() < TOL);
            assert!((x_vals[(1, j)] - 2.0).abs() < TOL);
        }
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_row_broadcast_constant_times_affine_matrix() {
    let x = variable((2, 3));
    let weights = constant_matrix(vec![10.0, 20.0, 30.0], 1, 3);
    let weighted = weights * x.clone();

    let sol = Problem::minimize(sum(&weighted))
        .subject_to([x.eq(ones((2, 3)))])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 120.0).abs() < TOL);
}

#[test]
fn test_column_broadcast_constant_times_affine_matrix() {
    let x = variable((2, 3));
    let weights = constant_matrix(vec![10.0, 20.0], 2, 1);
    let weighted = weights * x.clone();

    let sol = Problem::minimize(sum(&weighted))
        .subject_to([x.eq(ones((2, 3)))])
        .solve()
        .expect("problem should solve");

    assert!((sol.value.unwrap() - 90.0).abs() < TOL);
}
