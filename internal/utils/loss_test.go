package utils

import (
	"math"
	"testing"
)

func TestCrossEntropyLoss(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]float64
		labels   []int32
		expected float64
	}{
		{
			name: "simple_prediction",
			input: [][]float64{
				{0.1, 0.2, 0.7},
				{0.7, 0.2, 0.1},
			},
			labels:   []int32{2, 0},
			expected: 0.36,
		},
		{
			name: "simple_prediction_logits",
			input: Softmax([][]float64{
				{0.1, 0.2, 0.7},
				{0.7, 0.2, 0.1},
			}, 1),
			labels:   []int32{2, 0},
			expected: 0.77,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := CrossEntropyLoss(test.input, test.labels)
			if (math.Round(actual*100) / 100) != test.expected {
				t.Errorf("CrossEntropyLoss(%v, %v) = %f; want %f", test.input, test.labels, actual, test.expected)
			}
		})
	}
}
