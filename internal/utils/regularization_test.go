package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDropout(t *testing.T) {
	InitializeRand(42)
	tests := []struct {
		name        string
		input       *mat.Dense
		dropoutRate float64
		expected    *mat.Dense
	}{
		{
			name: "zero_dropout",
			input: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			dropoutRate: 0,
			expected: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
		},
		{
			name: "dropout_full",
			input: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			dropoutRate: 1,
			expected: mat.NewDense(2, 3, []float64{
				0, 0, 0,
				0, 0, 0,
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := Dropout(tt.input, tt.dropoutRate)
			if !mat.EqualApprox(actual, tt.expected, 1e-9) {
				t.Fatalf("Dropout() = %v; want %v", mat.Formatted(actual), mat.Formatted(tt.expected))
			}
		})
	}
}
