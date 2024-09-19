package model

import (
	"testing"

	"github.com/gabitoju/go-gcn/internal/utils"
)

func TestLayerForwardWithGivenWeights(t *testing.T) {
	tests := []struct {
		name      string
		features  [][]float64
		adjacency [][]float64
		weights   [][]float64
		bias      []float64
		expected  [][]float64
	}{
		{
			name:      "simple_layer_forward",
			features:  [][]float64{{1, 2, 3}, {1, 2, 3}},
			adjacency: [][]float64{{1, 0}, {0, 1}},
			weights:   [][]float64{{1, 1}, {1, 1}, {1, 1}},
			bias:      []float64{1, 1},
			expected:  [][]float64{{7, 7}, {7, 7}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			layer := NewLayerFromWeightsAndBias(test.weights, test.bias)
			actual := layer.Forward(test.features, test.adjacency)
			if !utils.EqualMatrices(actual, test.expected, 0) {
				t.Errorf("Layer.Forward(%v, %v) = %v; want %v", test.features, test.adjacency, actual, test.expected)
			}
		})
	}
}

func TestLayerForward(t *testing.T) {
	tests := []struct {
		name        string
		inFeatures  int
		outFeatures int
		features    [][]float64
		adjacency   [][]float64
	}{
		{
			name:        "layer_forward",
			inFeatures:  3,
			outFeatures: 2,
			features:    [][]float64{{1, 2, 3}, {1, 2, 3}},
			adjacency:   [][]float64{{1, 0}, {0, 1}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			layer := NewLayer(test.inFeatures, test.outFeatures)
			actual := layer.Forward(test.features, test.adjacency)
			if len(actual) != len(test.features) {
				t.Errorf("Layer.Forward(%v, %v) = %v; want %v", test.features, test.adjacency, actual, test.features)
			}
		})
	}

}
