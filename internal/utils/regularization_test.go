package utils

import (
	"testing"
)

func TestDropout(t *testing.T) {
	tests := []struct {
		name        string
		input       [][]float64
		dropoutRate float64
		expected    [][]float64
	}{
		{
			name: "zero_dropout",
			input: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
			dropoutRate: 0,
			expected: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
		},
		{
			name: "dropout_100_percent",
			input: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
			dropoutRate: 1,
			expected: [][]float64{
				{0, 0, 0},
				{0, 0, 0},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := dropout(test.input, test.dropoutRate)
			if !EqualMatrices(actual, test.expected, 1e-9) {
				t.Errorf("dropout(%v, %f) = %v; want %v", test.input, test.dropoutRate, actual, test.expected)
			}
		})
	}
}
