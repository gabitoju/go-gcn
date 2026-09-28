package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRelu(t *testing.T) {
	input := mat.NewDense(2, 3, []float64{
		1, -2, 3,
		4, -5, 6,
	})
	expected := mat.NewDense(2, 3, []float64{
		1, 0, 3,
		4, 0, 6,
	})

	actual := Relu(input)
	if !mat.EqualApprox(actual, expected, 1e-9) {
		t.Fatalf("Relu() = %v; want %v", mat.Formatted(actual), mat.Formatted(expected))
	}
}

func TestReluDerivative(t *testing.T) {
	input := mat.NewDense(2, 3, []float64{
		1, -2, 3,
		4, -5, 6,
	})
	expected := mat.NewDense(2, 3, []float64{
		1, 0, 1,
		1, 0, 1,
	})

	actual := ReluDerivative(input)
	if !mat.EqualApprox(actual, expected, 1e-9) {
		t.Fatalf("ReluDerivative() = %v; want %v", mat.Formatted(actual), mat.Formatted(expected))
	}
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name string
		dim  int
		in   *mat.Dense
		out  *mat.Dense
	}{
		{
			name: "row_softmax",
			dim:  1,
			in: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			out: mat.NewDense(2, 3, []float64{
				0.09003057317038046, 0.24472847105479764, 0.6652409557748219,
				0.09003057317038046, 0.24472847105479764, 0.6652409557748219,
			}),
		},
		{
			name: "column_softmax",
			dim:  0,
			in: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			out: mat.NewDense(2, 3, []float64{
				0.04742587317756678, 0.04742587317756678, 0.04742587317756678,
				0.9525741268224331, 0.9525741268224331, 0.9525741268224331,
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := Softmax(tt.in, tt.dim)
			if !mat.EqualApprox(actual, tt.out, 1e-9) {
				t.Fatalf("Softmax() = %v; want %v", mat.Formatted(actual), mat.Formatted(tt.out))
			}
		})
	}
}
