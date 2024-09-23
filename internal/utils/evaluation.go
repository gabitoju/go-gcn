package utils

import (
	"slices"
)

func Accuracy(y_pred [][]float64, y_true []int32) float64 {
	correct := 0
	for i := range y_pred {
		pred := slices.Index(y_pred[i], slices.Max(y_pred[i]))
		if int32(pred) == y_true[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(y_true))
}
