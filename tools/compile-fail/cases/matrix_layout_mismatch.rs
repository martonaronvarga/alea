use alea::math::matrix::{ColumnMajor, RowMajor, SquareMatrixRef};
fn row_major(_: SquareMatrixRef<'_, RowMajor>) {}
fn main() {
    let matrix = SquareMatrixRef::<ColumnMajor>::new(2, &[1., 0., 0., 1.]).unwrap();
    row_major(matrix);
}
