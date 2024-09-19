package model

import "github.com/gabitoju/go-gcn/internal/utils"

type GCN struct {
	Layers    []*Layer
	NLayers   int
	NFeatures int
	NHidden   int
	NClasses  int
	Dropout   float64
}

func NewGCN(nLayers, nFeatures, nHidden, nClasses int, dropout float64) *GCN {
	layers := make([]*Layer, nLayers)
	for i := range layers {
		if i == 0 {
			layers[i] = NewLayer(nFeatures, nHidden)
		} else if i == nLayers-1 {
			layers[i] = NewLayer(nHidden, nClasses)
		} else {
			layers[i] = NewLayer(nHidden, nHidden)
		}
	}
	return &GCN{
		Layers:    layers,
		NLayers:   nLayers,
		NFeatures: nFeatures,
		NHidden:   nHidden,
		NClasses:  nClasses,
		Dropout:   dropout,
	}
}

func (g *GCN) Forward(x, adj [][]float64) [][]float64 {
	out := x
	for i, layer := range g.Layers {
		out = layer.Forward(out, adj)
		if i < g.NLayers-1 {
			out = utils.Relu(out)
			out = utils.Dropout(out, g.Dropout)
		}
	}
	return utils.Softmax(out, 1)
}
