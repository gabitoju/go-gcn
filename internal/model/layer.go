package model

import (
	"math"

	"github.com/gabitoju/go-gcn/internal/utils"
)

type Layer struct {
	InFeatures   int
	OutFeatures  int
	Weights      [][]float64
	Bias         []float64
	dW           [][]float64
	dB           []float64
	dH           [][]float64
	H            [][]float64
	Z            [][]float64
	learningRate float64
	mW           [][]float64
	vW           [][]float64
	mB           []float64
	vB           []float64
	t            int
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
			l.Weights[i][j] = utils.RandFloat64()
		}
	}

	for i := range l.Bias {
		l.Bias[i] = utils.RandFloat64()
	}
}

func (l *Layer) Forward(input [][]float64, adj [][]float64) [][]float64 {
	l.H = input
	support := utils.MatMul(input, l.Weights)
	output := utils.MatMul(adj, support)

	output = utils.MatAddBroadcast(output, l.Bias)

	l.Z = output

	return output
}

func (l *Layer) Backward(gradOutput [][]float64) {

	reluGrad := utils.ReluDerivative(l.Z)

	gradZ := utils.MatElementWiseMul(gradOutput, reluGrad)

	l.dW = utils.MatMul(utils.Transpose(l.H), gradZ)

	l.dB = ComputeBiasGradient(gradZ)

	l.dH = utils.MatMul(gradZ, utils.Transpose(l.Weights))
}

func ComputeBiasGradient(gradZ [][]float64) []float64 {
	biasGradient := make([]float64, len(gradZ[0]))

	for i := 0; i < len(gradZ); i++ {
		for j := 0; j < len(gradZ[i]); j++ {
			biasGradient[j] += gradZ[i][j]
		}
	}

	return biasGradient
}

func (l *Layer) SGDUpdate(learningRate float64) {
	for i := 0; i < len(l.Weights); i++ {
		for j := 0; j < len(l.Weights[i]); j++ {
			l.Weights[i][j] -= learningRate * l.dW[i][j]
		}
	}

	for i := 0; i < len(l.Bias); i++ {
		l.Bias[i] -= learningRate * l.dB[i]
	}
}

func (l *Layer) AdamUpdate(beta1, beta2, epsilon float64) {
	l.t++

	if l.mW == nil {
		l.mW = make([][]float64, len(l.Weights))
		l.vW = make([][]float64, len(l.Weights))
		for i := range l.Weights {
			l.mW[i] = make([]float64, len(l.Weights[i]))
			l.vW[i] = make([]float64, len(l.Weights[i]))
		}
	}

	if l.mB == nil {
		l.mB = make([]float64, len(l.Bias))
		l.vB = make([]float64, len(l.Bias))
	}

	for i := 0; i < len(l.Weights); i++ {
		for j := 0; j < len(l.Weights[i]); j++ {
			l.mW[i][j] = beta1*l.mW[i][j] + (1-beta1)*l.dW[i][j]
			l.vW[i][j] = beta2*l.vW[i][j] + (1-beta2)*l.dW[i][j]*l.dW[i][j]

			mHat := l.mW[i][j] / (1 - math.Pow(beta1, float64(l.t)))
			vHat := l.vW[i][j] / (1 - math.Pow(beta2, float64(l.t)))

			l.Weights[i][j] -= l.learningRate * mHat / (math.Sqrt(vHat) + epsilon)
		}
	}

	for i := 0; i < len(l.Bias); i++ {
		l.mB[i] = beta1*l.mB[i] + (1-beta1)*l.dB[i]
		l.vB[i] = beta2*l.vB[i] + (1-beta2)*l.dB[i]*l.dB[i]

		mHat := l.mB[i] / (1 - math.Pow(beta1, float64(l.t)))
		vHat := l.vB[i] / (1 - math.Pow(beta2, float64(l.t)))

		l.Bias[i] -= l.learningRate * mHat / (math.Sqrt(vHat) + epsilon)
	}
}
