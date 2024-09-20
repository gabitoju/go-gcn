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

	// Convertir las matrices a formato gonum
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

	// Crear la matriz de resultado
	resultMat := mat.NewDense(len(a), len(b[0]), nil)

	// Realizar la multiplicación de matrices
	resultMat.Mul(aMat, bMat)

	// Convertir el resultado de nuevo a [][]float64
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
		for j := range result[i] {
			result[i][j] = resultMat.At(i, j)
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
