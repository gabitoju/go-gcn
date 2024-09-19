package model

import (
	"math/rand"

	"github.com/gabitoju/go-gcn/internal/utils"
)

type Layer struct {
	InFeatures  int
	OutFeatures int
	Weights     [][]float64
	Bias        []float64
}

func NewLayer(inFeatures, outFeatures int) *Layer {
	weights := make([][]float64, inFeatures)
	bias := make([]float64, outFeatures)

	layer := &Layer{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     weights,
		Bias:        bias,
	}

	layer.ResetWeightsAndBias()

	return layer
}

func NewLayerFromWeightsAndBias(weights [][]float64, bias []float64) *Layer {
	inFeatures := len(weights)
	outFeatures := len(bias)

	layer := &Layer{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     weights,
		Bias:        bias,
	}

	return layer
}

func (l *Layer) ResetWeightsAndBias() {
	for i := range l.Weights {
		l.Weights[i] = make([]float64, l.OutFeatures)
		for j := range l.Weights[i] {
			l.Weights[i][j] = rand.Float64()
		}
	}

	for i := range l.Bias {
		l.Bias[i] = rand.Float64()
	}
}

func (l *Layer) Forward(input [][]float64, adj [][]float64) [][]float64 {
	support := utils.MatMul(input, l.Weights)
	output := utils.MatMul(adj, support)

	output = utils.MatAddBroadcast(output, l.Bias)

	return output
}
