package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

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

	aMat := mat.NewDense(len(a), len(a[0]), nil)
	bMat := mat.NewDense(len(b), len(b[0]), nil)
	for i := range a {
		for j := range a[i] {
			aMat.Set(i, j, a[i][j])
		}
	}
	for i := range b {
		for j := range b[i] {
			bMat.Set(i, j, b[i][j])
		}
	}

	resultMat := mat.NewDense(len(a), len(b[0]), nil)

	resultMat.Mul(aMat, bMat)

	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
		for j := range result[i] {
			result[i][j] = resultMat.At(i, j)
		}
	}

	return result
}

func MatElementWiseMul(a, b [][]float64) [][]float64 {

	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		panic("Element-wise multiplication not possible")
	}

	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[0]))
		for j := range a[i] {
			result[i][j] = a[i][j] * b[i][j]
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
		m := len(a[i])
		for j := 0; j < m; j++ {
			result[i][j] = a[i][j] + b[j]
		}
	}
	return result
}

func Transpose(a [][]float64) [][]float64 {
	r, c := len(a), len(a[0])
	aMat := mat.NewDense(r, c, nil)
	for i := range a {
		for j := range a[i] {
			aMat.Set(i, j, a[i][j])
		}
	}

	tMat := aMat.T()

	result := make([][]float64, c)
	for i := range result {
		result[i] = make([]float64, r)
		for j := range result[i] {
			result[i][j] = tMat.At(i, j)
		}
	}

	return result
}
