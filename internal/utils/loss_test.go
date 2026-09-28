package utils

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropyLoss(t *testing.T) {
	tests := []struct {
		name     string
		input    *mat.Dense
		labels   []int32
		expected float64
	}{
		{
			name: "simple_prediction",
			input: mat.NewDense(2, 3, []float64{
				0.1, 0.2, 0.7,
				0.7, 0.2, 0.1,
			}),
			labels:   []int32{2, 0},
			expected: 0.36,
		},
		{
			name: "softmax_input",
			input: Softmax(mat.NewDense(2, 3, []float64{
				0.1, 0.2, 0.7,
				0.7, 0.2, 0.1,
			}), 1),
			labels:   []int32{2, 0},
			expected: 0.77,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := CrossEntropyLoss(tt.input, tt.labels)
			if math.Round(actual*100)/100 != tt.expected {
				t.Fatalf("CrossEntropyLoss() = %f; want %f", actual, tt.expected)
			}
		})
	}
}
