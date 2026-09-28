package data

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func NormalizeAdjacencyMatrix(adj *mat.Dense) *mat.Dense {
	n, _ := adj.Dims()
	var selfLoops mat.Dense
	selfLoops.Add(adj, IdentityMatrix(n))

	D := DegreeMatrix(&selfLoops)
	invSqrtD := mat.NewDense(n, n, nil)
	invSqrtD.Apply(func(i, j int, v float64) float64 {
		if i != j {
			return 0
		}
		if v == 0 {
			return 0
		}
		return 1 / math.Sqrt(v)
	}, D)

	var normalized mat.Dense
	var temp mat.Dense
	temp.Mul(invSqrtD, &selfLoops)
	normalized.Mul(&temp, invSqrtD)
	return &normalized
}

func IdentityMatrix(n int) *mat.Dense {
	identity := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		identity.Set(i, i, 1)
	}
	return identity
}

func DegreeMatrix(adj *mat.Dense) *mat.Dense {
	n, _ := adj.Dims()
	degree := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		var sum float64
		row := adj.RawRowView(i)
		for _, val := range row {
			sum += val
		}
		degree.Set(i, i, sum)
	}
	return degree
}
