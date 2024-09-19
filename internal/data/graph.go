package data

import (
	"math"

	"github.com/gabitoju/go-gcn/internal/utils"
)

func NormalizeAdjacencyMatrix(adj [][]float64) [][]float64 {
	I := IdentityMatrix(len(adj))
	selfLoopsMatrix := utils.MatAdd(adj, I)

	D := DegreeMatrix(selfLoopsMatrix)

	invSqrtD := make([][]float64, len(D))
	for i, row := range D {
		invSqrtD[i] = make([]float64, len(row))
		for j, val := range row {
			if val == 0 {
				invSqrtD[i][j] = 0
			} else {
				invSqrtD[i][j] = 1 / math.Sqrt(val)
			}
		}
	}
	normalizedAdj := utils.MatMul(utils.MatMul(invSqrtD, selfLoopsMatrix), invSqrtD)
	return normalizedAdj
}

func IdentityMatrix(n int) [][]float64 {
	identity := make([][]float64, n)
	for i := range identity {
		identity[i] = make([]float64, n)
		identity[i][i] = 1
	}
	return identity
}

func DegreeMatrix(adj [][]float64) [][]float64 {
	degree := make([][]float64, len(adj))
	for i, row := range adj {
		degree[i] = make([]float64, len(row))
		degree[i][i] = Sum(row)
	}
	return degree
}

func Sum(row []float64) float64 {
	var sum float64
	for _, val := range row {
		sum += val
	}
	return sum
}
