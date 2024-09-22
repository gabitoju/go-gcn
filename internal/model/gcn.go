package model

import (
	"github.com/gabitoju/go-gcn/internal/data"
	"github.com/gabitoju/go-gcn/internal/utils"
)

type GCN struct {
	Layers          []*Layer
	NLayers         int
	NFeatures       int
	NHidden         int
	NClasses        int
	Dropout         float64
	internalDropout float64
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
		layers[i].learningRate = 0.001
	}
	return &GCN{
		Layers:          layers,
		NLayers:         nLayers,
		NFeatures:       nFeatures,
		NHidden:         nHidden,
		NClasses:        nClasses,
		Dropout:         dropout,
		internalDropout: dropout,
	}
}

func (g *GCN) Train() {
	g.Dropout = g.internalDropout
}

func (g *GCN) Eval() {
	g.Dropout = 0
}

func (g *GCN) Forward(x, adj [][]float64) [][]float64 {
	normAdj := data.NormalizeAdjacencyMatrix(adj)
	out := x
	for i, layer := range g.Layers {
		out = layer.Forward(out, normAdj)
		if i < g.NLayers-1 {
			out = utils.Relu(out)
			out = utils.Dropout(out, g.Dropout)
		}
	}
	return utils.Softmax(out, 1)
}

func (g *GCN) Backward(gradOutput [][]float64) {

	gradients := gradOutput

	for i := g.NLayers - 1; i >= 0; i-- {
		g.Layers[i].Backward(gradients)
		gradients = g.Layers[i].dH
	}
}

func (gcn *GCN) SGDUpdateWeights(learningRate float64) {
	for _, layer := range gcn.Layers {
		layer.SGDUpdate(learningRate)
	}
}
