package utils

import "testing"

func TestRELU(t *testing.T) {
	tests := []struct {
		name     string
		input    float64
		expected float64
	}{
		{name: "zero_input", input: 0, expected: 0},
		{name: "positive_input", input: 1, expected: 1},
		{name: "negative_input", input: -1, expected: 0},
		{name: "positive_float_input", input: 0.5, expected: 0.5},
		{name: "negative_float_input", input: -0.5, expected: 0},
		{name: "large_positive_input", input: 1000, expected: 1000},
		{name: "large_negative_input", input: -1000, expected: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := Relu(test.input)
			if actual != test.expected {
				t.Errorf("relu(%f) = %f; want %f", test.input, actual, test.expected)
			}
		})
	}
}

func TestRELUDerivative(t *testing.T) {
	tests := []struct {
		name     string
		input    float64
		expected float64
	}{
		{name: "zero_input", input: 0, expected: 1},
		{name: "positive_input", input: 1, expected: 1},
		{name: "negative_input", input: -1, expected: 0},
		{name: "positive_float_input", input: 0.5, expected: 1},
		{name: "negative_float_input", input: -0.5, expected: 0},
		{name: "large_positive_input", input: 1000, expected: 1},
		{name: "large_negative_input", input: -1000, expected: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := relu_derivative(test.input)
			if actual != test.expected {
				t.Errorf("relu_derivative(%f) = %f; want %f", test.input, actual, test.expected)
			}
		})
	}
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name   string
		dim    int
		input  [][]float64
		output [][]float64
	}{
		{name: "simple_softmax_row", dim: 1, input: [][]float64{{1, 2, 3}, {4, 5, 6}}, output: [][]float64{{0.09003057317038046, 0.24472847105479764, 0.6652409557748219}, {0.09003057317038046, 0.24472847105479764, 0.6652409557748219}}},
		{name: "simple_softmax_column", dim: 0, input: [][]float64{{1, 2, 3}, {4, 5, 6}}, output: [][]float64{{0.04742587317756678, 0.04742587317756678, 0.04742587317756678}, {0.9525741268224331, 0.9525741268224331, 0.9525741268224331}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := Softmax(test.input, test.dim)
			if !EqualMatrices(actual, test.output, 1e-9) {
				t.Errorf("softmax(%v, 1) = %v; want %v", test.input, actual, test.output)
			}
		})
	}
}
