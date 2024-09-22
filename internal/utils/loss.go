package utils

import "math"

func CrossEntropyLoss(input [][]float64, labels []int32) float64 {
	epsilon := 1e-10

	loss := 0.0
	for i := range labels {
		trueValueIndex := labels[i]
		predictedValue := input[i][trueValueIndex]

		loss += -math.Log(predictedValue + epsilon)
	}
	return loss / float64(len(input))
}

func CrossEntropyLossDerivative(input [][]float64, labels []int32) [][]float64 {
	output := make([][]float64, len(input))

	for i := range labels {
		output[i] = make([]float64, len(input[0]))
		for j := range input[i] {
			if j == int(labels[i]) {
				output[i][j] = input[i][j] - 1.0 + 1e-10
			} else {
				output[i][j] = input[i][j]
			}
		}
	}

	for i := range output {
		for j := range output[i] {
			output[i][j] /= float64(len(input))
		}
	}

	return output
}
