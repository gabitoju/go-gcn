package utils

import "gonum.org/v1/gonum/mat"

func Dropout(x *mat.Dense, dropoutRate float64) *mat.Dense {
	r, c := x.Dims()
	output := mat.NewDense(r, c, nil)
	output.Apply(func(_, _ int, v float64) float64 {
		if RandFloat64() < dropoutRate {
			return 0
		}
		return v / (1 - dropoutRate)
	}, x)
	return output
}
