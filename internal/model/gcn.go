package model

import (
	"gonum.org/v1/gonum/mat"

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
	cachedAdjSrc    *mat.Dense
	cachedAdj       *mat.Dense
	hiddenBufs      []*mat.Dense
	dropoutMasks    []*mat.Dense
}

func NewGCN(nLayers, nFeatures, nHidden, nClasses int, dropout, lr float64) *GCN {
	layers := make([]*Layer, nLayers)
	for i := range layers {
		if i == 0 {
			layers[i] = NewLayer(nFeatures, nHidden)
		} else if i == nLayers-1 {
			layers[i] = NewLayer(nHidden, nClasses)
		} else {
			layers[i] = NewLayer(nHidden, nHidden)
		}
		layers[i].learningRate = lr
	}
	return &GCN{
		Layers:          layers,
		NLayers:         nLayers,
		NFeatures:       nFeatures,
		NHidden:         nHidden,
		NClasses:        nClasses,
		Dropout:         dropout,
		internalDropout: dropout,
		hiddenBufs:      make([]*mat.Dense, nLayers-1),
		dropoutMasks:    make([]*mat.Dense, nLayers-1),
	}
}

func (g *GCN) Train() {
	g.Dropout = g.internalDropout
}

func (g *GCN) Eval() {
	g.Dropout = 0
}

func (g *GCN) Forward(x, adj *mat.Dense) *mat.Dense {
	normAdj := g.normalizedAdj(adj)
	out := x
	for i, layer := range g.Layers {
		out = layer.Forward(out, normAdj)
		if i < g.NLayers-1 {
			out = g.hiddenActivation(i, out)
			g.applyDropout(i, out)
		}
	}
	return utils.Softmax(out, 1)
}

func (g *GCN) Backward(gradOutput *mat.Dense) {

	gradients := gradOutput

	for i := g.NLayers - 1; i >= 0; i-- {
		g.Layers[i].Backward(gradients)
		gradients = g.Layers[i].dH
		if i > 0 {
			gradients.MulElem(gradients, g.dropoutMasks[i-1])
			gradients.MulElem(gradients, utils.ReluDerivative(g.Layers[i-1].Z))
		}
	}
}

func (g *GCN) hiddenActivation(i int, input *mat.Dense) *mat.Dense {
	r, c := input.Dims()
	if g.hiddenBufs[i] == nil || !dimsMatch(g.hiddenBufs[i], r, c) {
		g.hiddenBufs[i] = mat.NewDense(r, c, nil)
	}
	g.hiddenBufs[i].Copy(input)
	return utils.ReluInPlace(g.hiddenBufs[i])
}

func (g *GCN) applyDropout(i int, input *mat.Dense) {
	r, c := input.Dims()
	if g.dropoutMasks[i] == nil || !dimsMatch(g.dropoutMasks[i], r, c) {
		g.dropoutMasks[i] = mat.NewDense(r, c, nil)
	}
	mask := g.dropoutMasks[i]
	if g.Dropout == 0 {
		mask.Apply(func(_, _ int, _ float64) float64 { return 1 }, mask)
		return
	}
	keepScale := 1 / (1 - g.Dropout)
	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			if utils.RandFloat64() < g.Dropout {
				mask.Set(row, col, 0)
			} else {
				mask.Set(row, col, keepScale)
			}
		}
	}
	input.MulElem(input, mask)
}

func (gcn *GCN) SGDUpdateWeights(learningRate float64) {
	for _, layer := range gcn.Layers {
		layer.SGDUpdate(learningRate)
	}
}

func (g *GCN) normalizedAdj(adj *mat.Dense) *mat.Dense {
	if g.cachedAdjSrc == adj && g.cachedAdj != nil {
		return g.cachedAdj
	}
	g.cachedAdj = data.NormalizeAdjacencyMatrix(adj)
	g.cachedAdjSrc = adj
	return g.cachedAdj
}
