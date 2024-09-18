package utils

import "math"

func CrossEntropyLoss(input [][]float64, labels []int) float64 {
	epsilon := 1e-10

	loss := 0.0
	for i := range input {
		trueValueIndex := labels[i]
		predictedValue := input[i][trueValueIndex]

		loss += -math.Log(predictedValue + epsilon)
	}
	return loss / float64(len(input))
}
