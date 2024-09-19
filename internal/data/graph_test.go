package data

import (
	"testing"

	"github.com/gabitoju/go-gcn/internal/utils"
)

func TestNormalizeAdjacencyMatrix(t *testing.T) {
	tests := []struct {
		name     string
		adj      [][]float64
		expected [][]float64
	}{
		{
			name: "simple_adj",
			adj: [][]float64{
				{0, 1, 0},
				{1, 0, 1},
				{0, 1, 0},
			},
			expected: [][]float64{
				{0.5, 0.4082, 0},
				{0.4082, 0.3333, 0.4082},
				{0, 0.4082, 0.5},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Running test: ", test.name)
			actual := NormalizeAdjacencyMatrix(test.adj)
			if !utils.EqualMatrices(actual, test.expected, 1e-4) {
				t.Errorf("NormalizeAdjacencyMatrix(%v) = %v; want %v", test.adj, actual, test.expected)
			}
		})
	}
}

func TestIdentityMatrix(t *testing.T) {
	tests := []struct {
		name string
		n    int
		want [][]float64
	}{
		{
			name: "simple_identity",
			n:    3,
			want: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := IdentityMatrix(test.n)
			if !utils.EqualMatrices(actual, test.want, 0) {
				t.Errorf("IdentityMatrix(%v) = %v; want %v", test.n, actual, test.want)
			}
			t.Logf("IdentityMatrix(%v) = %v; want %v", test.n, actual, test.want)
		})
	}
}

func TestDegreeMatrix(t *testing.T) {
	tests := []struct {
		name string
		adj  [][]float64
		want [][]float64
	}{
		{
			name: "simple_degree",
			adj: [][]float64{
				{0, 1, 0},
				{1, 0, 1},
				{0, 1, 0},
			},
			want: [][]float64{
				{1, 0, 0},
				{0, 2, 0},
				{0, 0, 1},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := DegreeMatrix(test.adj)
			if !utils.EqualMatrices(actual, test.want, 0) {
				t.Errorf("DegreeMatrix(%v) = %v; want %v", test.adj, actual, test.want)
			}
			t.Logf("DegreeMatrix(%v) = %v; want %v", test.adj, actual, test.want)
		})
	}
}

func TestSum(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		want float64
	}{
		{
			name: "simple_sum",
			x:    []float64{1, 2, 3},
			want: 6,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := Sum(test.x)
			if actual != test.want {
				t.Errorf("Sum(%v) = %v; want %v", test.x, actual, test.want)
			}
			t.Logf("Sum(%v) = %v; want %v", test.x, actual, test.want)
		})
	}
}
