package model

import "math/rand"

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
