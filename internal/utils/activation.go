package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Relu(x *mat.Dense) *mat.Dense {
	return ReluInPlace(mat.DenseCopyOf(x))
}

func ReluInPlace(x *mat.Dense) *mat.Dense {
	x.Apply(func(_, _ int, v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	}, x)
	return x
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
	return SoftmaxInPlace(mat.DenseCopyOf(input), dim)
}

func SoftmaxInPlace(input *mat.Dense, dim int) *mat.Dense {
	r, c := input.Dims()
	if dim == 0 {
		for j := 0; j < c; j++ {
			sum := 0.0
			for i := 0; i < r; i++ {
				val := math.Exp(input.At(i, j))
				input.Set(i, j, val)
				sum += val
			}
			for i := 0; i < r; i++ {
				input.Set(i, j, input.At(i, j)/sum)
			}
		}
		return input
	}

	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			val := math.Exp(input.At(i, j))
			input.Set(i, j, val)
			sum += val
		}
		for j := 0; j < c; j++ {
			input.Set(i, j, input.At(i, j)/sum)
		}
	}
	return input
}
