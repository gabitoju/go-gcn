package model

import (
	"testing"
)

func TestGCNForward(t *testing.T) {
	tests := []struct {
		name         string
		nLayers      int
		nFeatures    int
		nHidden      int
		nClasses     int
		dropout      float64
		learningRate float64
		x            [][]float64
		adj          [][]float64
	}{
		{
			name:         "simple_gcn_forward",
			nLayers:      2,
			nFeatures:    3,
			nHidden:      2,
			nClasses:     2,
			dropout:      0.5,
			learningRate: 0.001,
			x:            [][]float64{{1, 2, 3}, {1, 2, 3}},
			adj:          [][]float64{{1, 0}, {0, 1}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gcn := NewGCN(test.nLayers, test.nFeatures, test.nHidden, test.nClasses, test.dropout, test.learningRate)
			actual := gcn.Forward(test.x, test.adj)
			if len(actual) != len(test.x) {
				t.Errorf("GCN.Forward(%v, %v) = %v; want %v", test.x, test.adj, actual, test.x)
			}
		})
	}
}
