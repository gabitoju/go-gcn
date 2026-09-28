package model

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/gabitoju/go-gcn/internal/data"
	"github.com/gabitoju/go-gcn/internal/utils"
)

func TestGCNForward(t *testing.T) {
	gcn := NewGCN(2, 3, 2, 2, 0.5, 0.001)
	x := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		1, 2, 3,
	})
	adj := mat.NewDense(2, 2, []float64{
		1, 0,
		0, 1,
	})

	output := gcn.Forward(x, adj)
	rows, _ := output.Dims()
	if rows != 2 {
		t.Fatalf("expected 2 rows, got %d", rows)
	}
}

func TestGCNBackwardMatchesFiniteDifference(t *testing.T) {
	gcn := NewGCN(2, 1, 1, 2, 0, 0.01)
	gcn.Layers[0] = NewLayerFromWeightsAndBias(mat.NewDense(1, 1, []float64{0.5}), mat.NewVecDense(1, []float64{0}))
	gcn.Layers[1] = NewLayerFromWeightsAndBias(mat.NewDense(1, 2, []float64{0.2, -0.1}), mat.NewVecDense(2, []float64{0, 0}))

	x := mat.NewDense(2, 1, []float64{1, 2})
	adj := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	labels := []int32{0, 1}
	indices := []int{0, 1}

	output := gcn.Forward(x, adj)
	gcn.Backward(utils.CrossEntropyLossDerivative(output, labels, indices))
	analytic := gcn.Layers[0].dW.At(0, 0)

	const epsilon = 1e-6
	weight := gcn.Layers[0].Weights.At(0, 0)
	gcn.Layers[0].Weights.Set(0, 0, weight+epsilon)
	plus := utils.CrossEntropyLoss(gcn.Forward(x, adj), labels)
	gcn.Layers[0].Weights.Set(0, 0, weight-epsilon)
	minus := utils.CrossEntropyLoss(gcn.Forward(x, adj), labels)
	gcn.Layers[0].Weights.Set(0, 0, weight)
	numerical := (plus - minus) / (2 * epsilon)

	if math.Abs(analytic-numerical) > 1e-5 {
		t.Fatalf("first-layer gradient = %.8f; finite difference = %.8f", analytic, numerical)
	}
}

func TestGCNAdjacencyCaching(t *testing.T) {
	gcn := NewGCN(2, 3, 2, 2, 0.5, 0.001)
	x := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		1, 2, 3,
	})
	adj := mat.NewDense(2, 2, []float64{
		1, 0,
		0, 1,
	})

	gcn.Forward(x, adj)
	first := gcn.cachedAdj
	if first == nil {
		t.Fatal("expected cached adjacency after first forward")
	}

	gcn.Forward(x, adj)
	if gcn.cachedAdj != first {
		t.Fatal("expected cached adjacency to be reused with same pointer")
	}
}

func TestGCNSparseForwardMatchesDense(t *testing.T) {
	gcn := NewGCN(2, 1, 1, 2, 0, 0.001)
	gcn.Layers[0] = NewLayerFromWeightsAndBias(mat.NewDense(1, 1, []float64{0.5}), mat.NewVecDense(1, []float64{0}))
	gcn.Layers[1] = NewLayerFromWeightsAndBias(mat.NewDense(1, 2, []float64{0.2, -0.1}), mat.NewVecDense(2, []float64{0, 0}))
	x := mat.NewDense(3, 1, []float64{1, 2, 3})
	denseAdj := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,
	})
	sparseAdj := data.NewSparseMatrix(3, 3, [][2]int{{0, 1}, {1, 0}, {1, 2}, {2, 1}})

	denseOutput := gcn.Forward(x, denseAdj)
	sparseOutput := gcn.ForwardSparse(x, sparseAdj)
	if !mat.EqualApprox(sparseOutput, denseOutput, 1e-12) {
		t.Fatalf("sparse output = %v; dense output = %v", mat.Formatted(sparseOutput), mat.Formatted(denseOutput))
	}
}
