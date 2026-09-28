package model

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayerForwardWithGivenWeights(t *testing.T) {
	layer := NewLayerFromWeightsAndBias(
		mat.NewDense(3, 2, []float64{
			1, 1,
			1, 1,
			1, 1,
		}),
		mat.NewVecDense(2, []float64{1, 1}),
	)

	features := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		1, 2, 3,
	})
	adj := mat.NewDense(2, 2, []float64{
		1, 0,
		0, 1,
	})
	expected := mat.NewDense(2, 2, []float64{
		7, 7,
		7, 7,
	})

	actual := layer.Forward(features, adj)
	if !mat.EqualApprox(actual, expected, 1e-9) {
		t.Fatalf("Layer.Forward() = %v; want %v", mat.Formatted(actual), mat.Formatted(expected))
	}
}

func TestLayerForwardDimensions(t *testing.T) {
	layer := NewLayer(3, 2)
	features := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		1, 2, 3,
	})
	adj := mat.NewDense(2, 2, []float64{
		1, 0,
		0, 1,
	})

	actual := layer.Forward(features, adj)
	rows, _ := actual.Dims()
	if rows != 2 {
		t.Fatalf("expected 2 rows, got %d", rows)
	}
}

func TestLayerBackwardPropagatesThroughAdjacency(t *testing.T) {
	layer := NewLayerFromWeightsAndBias(
		mat.NewDense(1, 1, []float64{2}),
		mat.NewVecDense(1, []float64{0}),
	)
	features := mat.NewDense(2, 1, []float64{1, 3})
	adj := mat.NewDense(2, 2, []float64{
		0, 1,
		1, 0,
	})
	layer.Forward(features, adj)
	layer.Backward(mat.NewDense(2, 1, []float64{1, 2}))

	if got := layer.dW.At(0, 0); got != 5 {
		t.Fatalf("dW = %v; want 5", got)
	}
	if got := layer.dH; !mat.EqualApprox(got, mat.NewDense(2, 1, []float64{4, 2}), 1e-12) {
		t.Fatalf("dH = %v; want [[4] [2]]", mat.Formatted(got))
	}
	if got := layer.dB.AtVec(0); got != 3 {
		t.Fatalf("dB = %v; want 3", got)
	}
}
