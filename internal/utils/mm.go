package utils

import "math"

func EqualMatrices(a, b [][]float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
	}

	for i := range a {
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}

func MatMul(a, b [][]float64) [][]float64 {
	if len(a[0]) != len(b) {
		panic("Matrix multiplication not possible")
	}

	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
	}

	for i := range a {
		for j := range b[0] {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

func MatAdd(a, b [][]float64) [][]float64 {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		panic("Matrix addition not possible")
	}

	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[i]))
	}

	for i := range a {
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func MatAddBroadcast(a [][]float64, b []float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[i]))
	}

	for i := range a {
		for j := range a[i] {
			result[i][j] = a[i][j] + b[j]
		}
	}
	return result
}
