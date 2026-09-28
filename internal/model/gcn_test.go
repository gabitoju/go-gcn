package model

import (
	"testing"

	"gonum.org/v1/gonum/mat"
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
