package utils

import "gonum.org/v1/gonum/mat"

func Accuracy(yPred *mat.Dense, yTrue []int32) float64 {
	rows, cols := yPred.Dims()
	correct := 0
	for i := 0; i < rows; i++ {
		maxIdx := 0
		maxVal := yPred.At(i, 0)
		for j := 1; j < cols; j++ {
			if val := yPred.At(i, j); val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}
		if int32(maxIdx) == yTrue[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}
