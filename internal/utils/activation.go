package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Relu(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	out.Apply(func(_, _ int, v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	}, x)
	return out
}

func ReluDerivative(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	out.Apply(func(_, _ int, v float64) float64 {
		if v < 0 {
			return 0
		}
		return 1
	}, x)
	return out
}

func Softmax(input *mat.Dense, dim int) *mat.Dense {
	r, c := input.Dims()
	output := mat.NewDense(r, c, nil)

	if dim == 0 {
		for j := 0; j < c; j++ {
			sum := 0.0
			expVals := make([]float64, r)
			for i := 0; i < r; i++ {
				val := math.Exp(input.At(i, j))
				expVals[i] = val
				sum += val
			}
			for i := 0; i < r; i++ {
				output.Set(i, j, expVals[i]/sum)
			}
		}
		return output
	}

	for i := 0; i < r; i++ {
		sum := 0.0
		expVals := make([]float64, c)
		for j := 0; j < c; j++ {
			val := math.Exp(input.At(i, j))
			expVals[j] = val
			sum += val
		}
		for j := 0; j < c; j++ {
			output.Set(i, j, expVals[j]/sum)
		}
	}

	return output
}
