package utils

import (
	"math/rand"
)

func Dropout(x [][]float64, dropoutRate float64) [][]float64 {

	output := make([][]float64, len(x))

	for i := range x {
		output[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			if rand.Float64() < dropoutRate {
				output[i][j] = 0
			} else {
				output[i][j] = x[i][j] / (1 - dropoutRate)
			}
		}
	}
	return output
}
