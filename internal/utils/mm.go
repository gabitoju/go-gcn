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
