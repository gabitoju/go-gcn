package utils

import "gonum.org/v1/gonum/mat"

func Dropout(x *mat.Dense, dropoutRate float64) *mat.Dense {
	return DropoutInPlace(mat.DenseCopyOf(x), dropoutRate)
}

func DropoutInPlace(x *mat.Dense, dropoutRate float64) *mat.Dense {
	x.Apply(func(_, _ int, v float64) float64 {
		if RandFloat64() < dropoutRate {
			return 0
		}
		return v / (1 - dropoutRate)
	}, x)
	return x
}
