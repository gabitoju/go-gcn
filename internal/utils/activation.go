package utils

import (
	"math"
)

func Relu(x [][]float64) [][]float64 {
	output := make([][]float64, len(x))
	for i, row := range x {
		output[i] = make([]float64, len(row))
		for j, val := range row {
			output[i][j] = relu(val)
		}
	}
	return output
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func relu_derivative(x [][]float64) [][]float64 {
	output := make([][]float64, len(x))
	for i, row := range x {
		output[i] = make([]float64, len(row))
		for j, val := range row {
			output[i][j] = relu_derivative1d(val)
		}
	}
	return output
}

func relu_derivative1d(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func Softmax(input [][]float64, dim int) [][]float64 {

	output := make([][]float64, len(input))

	if dim == 0 {

		rows := len(input)
		columns := len(input[0])

		for i := range output {
			output[i] = make([]float64, columns)
		}

		for i := 0; i < columns; i++ {
			column := make([]float64, rows)
			for j := 0; j < rows; j++ {
				column[j] = input[j][i]
			}

			softmaxColumns := softmax1d(column)

			for j := 0; j < rows; j++ {
				output[j][i] = softmaxColumns[j]
			}
		}
	} else {
		for i, row := range input {
			output[i] = softmax1d(row)
		}
	}

	return output
}

func softmax1d(logits []float64) []float64 {
	output := make([]float64, len(logits))
	sum := 0.0
	for i := range logits {
		sum += math.Exp(logits[i])
	}
	for i := range logits {
		output[i] = math.Exp(logits[i]) / sum
	}
	return output
}
