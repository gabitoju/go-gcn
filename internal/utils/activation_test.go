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
			actual := relu(test.input)
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
