package data

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNormalizeAdjacencyMatrix(t *testing.T) {
	tests := []struct {
		name     string
		adj      *mat.Dense
		expected *mat.Dense
	}{
		{
			name: "simple_adj",
			adj: mat.NewDense(3, 3, []float64{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}),
			expected: mat.NewDense(3, 3, []float64{
				0.5, 0.4082, 0,
				0.4082, 0.3333, 0.4082,
				0, 0.4082, 0.5,
			}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Running test: ", test.name)
			actual := NormalizeAdjacencyMatrix(test.adj)
			if !mat.EqualApprox(actual, test.expected, 1e-4) {
				t.Errorf("NormalizeAdjacencyMatrix(%v) = %v; want %v", test.adj, actual, test.expected)
			}
		})
	}
}

func TestIdentityMatrix(t *testing.T) {
	tests := []struct {
		name string
		n    int
		want *mat.Dense
	}{
		{
			name: "simple_identity",
			n:    3,
			want: mat.NewDense(3, 3, []float64{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1,
			}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := IdentityMatrix(test.n)
			if !mat.EqualApprox(actual, test.want, 0) {
				t.Errorf("IdentityMatrix(%v) = %v; want %v", test.n, actual, test.want)
			}
			t.Logf("IdentityMatrix(%v) = %v; want %v", test.n, actual, test.want)
		})
	}
}

func TestDegreeMatrix(t *testing.T) {
	tests := []struct {
		name string
		adj  *mat.Dense
		want *mat.Dense
	}{
		{
			name: "simple_degree",
			adj: mat.NewDense(3, 3, []float64{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}),
			want: mat.NewDense(3, 3, []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 1,
			}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := DegreeMatrix(test.adj)
			if !mat.EqualApprox(actual, test.want, 0) {
				t.Errorf("DegreeMatrix(%v) = %v; want %v", test.adj, actual, test.want)
			}
			t.Logf("DegreeMatrix(%v) = %v; want %v", test.adj, actual, test.want)
		})
	}
}
