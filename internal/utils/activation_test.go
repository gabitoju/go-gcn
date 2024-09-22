package utils

import "testing"

func TestRelu(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]float64
		expected [][]float64
	}{
		{name: "simple_relu", input: [][]float64{{1, -2, 3}, {4, -5, 6}}, expected: [][]float64{{1, 0, 3}, {4, 0, 6}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := Relu(test.input)
			if !EqualMatrices(actual, test.expected, 0) {
				t.Errorf("relu(%v) = %v; want %v", test.input, actual, test.expected)
			}
		})
	}
}

func TestReluDerivative(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]float64
		expected [][]float64
	}{
		{name: "simple_relu_derivative", input: [][]float64{{1, -2, 3}, {4, -5, 6}}, expected: [][]float64{{1, 0, 1}, {1, 0, 1}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := ReluDerivative(test.input)
			if !EqualMatrices(actual, test.expected, 0) {
				t.Errorf("relu_derivative(%v) = %v; want %v", test.input, actual, test.expected)
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
