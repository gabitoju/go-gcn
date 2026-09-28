package data

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSparseMatrixMulDense(t *testing.T) {
	sparse := NewSparseMatrix(3, 3, [][2]int{{0, 1}, {1, 0}, {1, 2}, {2, 1}})
	input := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	actual := mat.NewDense(3, 2, nil)
	sparse.MulDense(actual, input)
	want := mat.NewDense(3, 2, []float64{3, 4, 6, 8, 3, 4})
	if !mat.EqualApprox(actual, want, 1e-12) {
		t.Fatalf("sparse multiplication = %v; want %v", mat.Formatted(actual), mat.Formatted(want))
	}
}

func TestSparseNormalizationMatchesDense(t *testing.T) {
	entries := [][2]int{{0, 1}, {1, 0}, {1, 2}, {2, 1}}
	sparse := NewSparseMatrix(3, 3, entries).NormalizedWithSelfLoops()
	actual := mat.NewDense(3, 3, nil)
	sparse.MulDense(actual, IdentityMatrix(3))

	dense := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,
	})
	want := NormalizeAdjacencyMatrix(dense)
	if !mat.EqualApprox(actual, want, 1e-12) {
		t.Fatalf("sparse normalization = %v; want %v", mat.Formatted(actual), mat.Formatted(want))
	}
}
