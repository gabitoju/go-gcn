package utils

import "testing"

func TestEqualMatrices(t *testing.T) {
	tests := []struct {
		name      string
		matrix1   [][]float64
		matrix2   [][]float64
		tolerance float64
		expected  bool
	}{
		{
			name:      "equal_matrices",
			matrix1:   [][]float64{{1, 2, 3}, {4, 5, 6}},
			matrix2:   [][]float64{{1, 2, 3}, {4, 5, 6}},
			tolerance: 1e-9,
			expected:  true,
		},
		{
			name:      "not_equal_matrices",
			matrix1:   [][]float64{{1, 2, 3}, {4, 5, 6}},
			matrix2:   [][]float64{{1, 2, 3}, {4, 5, 7}},
			tolerance: 1e-9,
			expected:  false,
		},
		{
			name:      "different_dimensions",
			matrix1:   [][]float64{{1, 2, 3}, {4, 5, 6}},
			matrix2:   [][]float64{{1, 2, 3}, {4, 5}},
			tolerance: 1e-9,
			expected:  false,
		},
		{
			name:      "different_dimensions_2",
			matrix1:   [][]float64{{1, 2, 3}, {4, 5, 6}},
			matrix2:   [][]float64{{1, 2}, {3, 4}, {5, 6}},
			tolerance: 1e-9,
			expected:  false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := EqualMatrices(test.matrix1, test.matrix2, test.tolerance)
			if actual != test.expected {
				t.Errorf("EqualMatrices(%v, %v, %f) = %t; want %t", test.matrix1, test.matrix2, test.tolerance, actual, test.expected)
			}
		})
	}
}

func TestMatMul(t *testing.T) {
	tests := []struct {
		name     string
		matrix1  [][]float64
		matrix2  [][]float64
		expected [][]float64
	}{
		{
			name:     "simple_matrix_multiplication",
			matrix1:  [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			matrix2:  [][]float64{{9, 8}, {6, 5}, {3, 2}},
			expected: [][]float64{{30, 24}, {84, 69}, {138, 114}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := MatMul(test.matrix1, test.matrix2)
			if !EqualMatrices(actual, test.expected, 1e-9) {
				t.Errorf("MatMul(%v, %v) = %v; want %v", test.matrix1, test.matrix2, actual, test.expected)
			}
		})
	}
}

func TestMatAdd(t *testing.T) {
	tests := []struct {
		name     string
		matrix1  [][]float64
		matrix2  [][]float64
		expected [][]float64
	}{
		{
			name:     "simple_matrix_addition",
			matrix1:  [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			matrix2:  [][]float64{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}},
			expected: [][]float64{{10, 10, 10}, {10, 10, 10}, {10, 10, 10}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := MatAdd(test.matrix1, test.matrix2)
			if !EqualMatrices(actual, test.expected, 1e-9) {
				t.Errorf("MatAdd(%v, %v) = %v; want %v", test.matrix1, test.matrix2, actual, test.expected)
			}
		})
	}
}

func TestMatAddBroadcast(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		vector   []float64
		expected [][]float64
	}{
		{
			name:     "simple_broadcast",
			matrix:   [][]float64{{1, 2}, {3, 4}},
			vector:   []float64{1, 2},
			expected: [][]float64{{2, 4}, {4, 6}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := MatAddBroadcast(test.matrix, test.vector)
			if !EqualMatrices(actual, test.expected, 1e-9) {
				t.Errorf("MatAddBroadcast(%v, %v) = %v; want %v", test.matrix, test.vector, actual, test.expected)
			}
		})
	}

}

func TestTranspose(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		expected [][]float64
	}{
		{
			name:     "simple_transpose",
			matrix:   [][]float64{{1, 2, 3}, {4, 5, 6}},
			expected: [][]float64{{1, 4}, {2, 5}, {3, 6}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := Transpose(test.matrix)
			if !EqualMatrices(actual, test.expected, 1e-9) {
				t.Errorf("Transpose(%v) = %v; want %v", test.matrix, actual, test.expected)
			}
		})
	}
}

func TestMatElementWiseMul(t *testing.T) {
	tests := []struct {
		name     string
		matrix1  [][]float64
		matrix2  [][]float64
		expected [][]float64
	}{
		{
			name:     "simple_element_wise_multiplication",
			matrix1:  [][]float64{{1, 2, 3}, {4, 5, 6}},
			matrix2:  [][]float64{{1, 2, 3}, {4, 5, 6}},
			expected: [][]float64{{1, 4, 9}, {16, 25, 36}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := MatElementWiseMul(test.matrix1, test.matrix2)
			if !EqualMatrices(actual, test.expected, 1e-9) {
				t.Errorf("MatElementWiseMul(%v, %v) = %v; want %v", test.matrix1, test.matrix2, actual, test.expected)
			}
		})
	}
}
