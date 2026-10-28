use alea_math::matrix::{ColumnMajor, MatrixError, RowMajor, SquareMatrixRef, cholesky_in_place};
#[test]
fn layouts_recover_the_same_factor_without_transposition() {
    let mut row = [4., 2., -2., 2., 10., 2., -2., 2., 11.];
    let mut column = row;
    cholesky_in_place::<RowMajor>(3, &mut row).unwrap();
    cholesky_in_place::<ColumnMajor>(3, &mut column).unwrap();
    let r = SquareMatrixRef::<RowMajor>::new(3, &row).unwrap();
    let c = SquareMatrixRef::<ColumnMajor>::new(3, &column).unwrap();
    let expected = [[2., 0., 0.], [1., 3., 0.], [-1., 1., 3.]];
    for (i, entries) in expected.iter().enumerate() {
        for (j, expected) in entries.iter().enumerate() {
            assert_eq!(r.get(i, j), Some(*expected));
            assert_eq!(c.get(i, j), Some(*expected));
        }
    }
    assert_ne!(row, column);
    assert_eq!(r.get(3, 0), None);
}
#[test]
fn shape_failure_preserves_output_and_invalid_pivots_are_reported() {
    let mut values = [1., 2., 3.];
    assert_eq!(
        cholesky_in_place::<RowMajor>(2, &mut values),
        Err(MatrixError::Shape)
    );
    assert_eq!(values, [1., 2., 3.]);
    assert!(SquareMatrixRef::<ColumnMajor>::new(usize::MAX, &[]).is_err());
    assert_eq!(
        cholesky_in_place::<RowMajor>(2, &mut [1., 2., 2., 1.]),
        Err(MatrixError::NotPositiveDefinite { index: 1 })
    );
    assert!(cholesky_in_place::<ColumnMajor>(0, &mut []).is_ok());
}
