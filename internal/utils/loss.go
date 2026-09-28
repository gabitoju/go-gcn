package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func CrossEntropyLoss(input *mat.Dense, labels []int32) float64 {
	epsilon := 1e-10
	rows, _ := input.Dims()

	loss := 0.0
	for i := 0; i < rows; i++ {
		trueIdx := int(labels[i])
		predicted := input.At(i, trueIdx)
		loss += -math.Log(predicted + epsilon)
	}

	return loss / float64(rows)
}

func CrossEntropyLossDerivative(input *mat.Dense, labels []int32, indices []int) *mat.Dense {
	rows, cols := input.Dims()
	grad := mat.NewDense(rows, cols, nil)
	if len(indices) == 0 {
		return grad
	}

	norm := float64(len(indices))
	for i, idx := range indices {
		label := int(labels[i])
		for j := 0; j < cols; j++ {
			val := input.At(idx, j)
			if j == label {
				grad.Set(idx, j, (val-1.0+1e-10)/norm)
			} else {
				grad.Set(idx, j, val/norm)
			}
		}
	}
	return grad
}
