package model

import (
	"math"

	"gonum.org/v1/gonum/mat"

	"github.com/gabitoju/go-gcn/internal/utils"
)

type Layer struct {
	InFeatures   int
	OutFeatures  int
	Weights      *mat.Dense
	Bias         *mat.VecDense
	dW           *mat.Dense
	dB           *mat.VecDense
	dH           *mat.Dense
	H            *mat.Dense
	Z            *mat.Dense
	adj          *mat.Dense
	supportBuf   *mat.Dense
	outputBuf    *mat.Dense
	gradZBuf     *mat.Dense
	gradSupport  *mat.Dense
	learningRate float64
	mW           *mat.Dense
	vW           *mat.Dense
	mB           *mat.VecDense
	vB           *mat.VecDense
	t            int
}

func NewLayer(inFeatures, outFeatures int) *Layer {
	layer := &Layer{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     mat.NewDense(inFeatures, outFeatures, nil),
		Bias:        mat.NewVecDense(outFeatures, nil),
	}
	layer.ResetWeightsAndBias()
	return layer
}

func NewLayerFromWeightsAndBias(weights *mat.Dense, bias *mat.VecDense) *Layer {
	r, c := weights.Dims()
	layer := &Layer{
		InFeatures:  r,
		OutFeatures: c,
		Weights:     weights,
		Bias:        bias,
	}
	return layer
}

func (l *Layer) ResetWeightsAndBias() {
	r, c := l.Weights.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			l.Weights.Set(i, j, utils.RandFloat64())
		}
	}
	for i := 0; i < l.Bias.Len(); i++ {
		l.Bias.SetVec(i, utils.RandFloat64())
	}
}

func (l *Layer) Forward(input, adj *mat.Dense) *mat.Dense {
	l.H = input
	l.adj = adj
	rows, _ := input.Dims()

	if l.supportBuf == nil || !dimsMatch(l.supportBuf, rows, l.OutFeatures) {
		l.supportBuf = mat.NewDense(rows, l.OutFeatures, nil)
	}
	l.supportBuf.Mul(input, l.Weights)

	if l.outputBuf == nil || !dimsMatch(l.outputBuf, rows, l.OutFeatures) {
		l.outputBuf = mat.NewDense(rows, l.OutFeatures, nil)
	}
	l.outputBuf.Mul(adj, l.supportBuf)

	addBias(l.outputBuf, l.Bias)
	l.Z = l.outputBuf
	return l.outputBuf
}

func addBias(out *mat.Dense, bias *mat.VecDense) {
	rows, _ := out.Dims()
	for i := 0; i < rows; i++ {
		row := mat.NewVecDense(bias.Len(), out.RawRowView(i))
		row.AddVec(row, bias)
	}
}

func (l *Layer) Backward(gradOutput *mat.Dense) {
	rows, cols := gradOutput.Dims()

	if l.gradZBuf == nil || !dimsMatch(l.gradZBuf, rows, cols) {
		l.gradZBuf = mat.NewDense(rows, cols, nil)
	}
	l.gradZBuf.Copy(gradOutput)

	if l.gradSupport == nil || !dimsMatch(l.gradSupport, rows, cols) {
		l.gradSupport = mat.NewDense(rows, cols, nil)
	}
	l.gradSupport.Mul(l.adj.T(), l.gradZBuf)

	if l.dW == nil || !dimsMatch(l.dW, l.InFeatures, l.OutFeatures) {
		l.dW = mat.NewDense(l.InFeatures, l.OutFeatures, nil)
	}
	l.dW.Mul(l.H.T(), l.gradSupport)

	l.dB = ComputeBiasGradient(l.gradZBuf)

	if l.dH == nil || !dimsMatch(l.dH, rows, l.InFeatures) {
		l.dH = mat.NewDense(rows, l.InFeatures, nil)
	}
	l.dH.Mul(l.gradSupport, l.Weights.T())
}

func dimsMatch(m *mat.Dense, r, c int) bool {
	if m == nil {
		return false
	}
	rows, cols := m.Dims()
	return rows == r && cols == c
}

func ComputeBiasGradient(gradZ *mat.Dense) *mat.VecDense {
	_, cols := gradZ.Dims()
	grad := mat.NewVecDense(cols, nil)
	rows, _ := gradZ.Dims()
	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += gradZ.At(i, j)
		}
		grad.SetVec(j, sum)
	}
	return grad
}

func (l *Layer) SGDUpdate(learningRate float64) {
	wRows, wCols := l.Weights.Dims()
	for i := 0; i < wRows; i++ {
		for j := 0; j < wCols; j++ {
			l.Weights.Set(i, j, l.Weights.At(i, j)-learningRate*l.dW.At(i, j))
		}
	}
	for i := 0; i < l.Bias.Len(); i++ {
		l.Bias.SetVec(i, l.Bias.AtVec(i)-learningRate*l.dB.AtVec(i))
	}
}

func (l *Layer) AdamUpdate(beta1, beta2, epsilon, weightDecay float64) {
	l.t++
	wRows, wCols := l.Weights.Dims()

	if l.mW == nil {
		l.mW = mat.NewDense(wRows, wCols, nil)
		l.vW = mat.NewDense(wRows, wCols, nil)
	}
	if l.mB == nil {
		l.mB = mat.NewVecDense(l.OutFeatures, nil)
		l.vB = mat.NewVecDense(l.OutFeatures, nil)
	}

	for i := 0; i < wRows; i++ {
		for j := 0; j < wCols; j++ {
			grad := l.dW.At(i, j)
			mPrev := l.mW.At(i, j)
			vPrev := l.vW.At(i, j)

			mVal := beta1*mPrev + (1-beta1)*grad
			vVal := beta2*vPrev + (1-beta2)*grad*grad

			l.mW.Set(i, j, mVal)
			l.vW.Set(i, j, vVal)

			mHat := mVal / (1 - math.Pow(beta1, float64(l.t)))
			vHat := vVal / (1 - math.Pow(beta2, float64(l.t)))

			update := mHat / (math.Sqrt(vHat) + epsilon)
			weight := l.Weights.At(i, j) - l.learningRate*update
			weight -= l.learningRate * weightDecay * weight
			l.Weights.Set(i, j, weight)
		}
	}

	for i := 0; i < l.Bias.Len(); i++ {
		grad := l.dB.AtVec(i)
		mPrev := l.mB.AtVec(i)
		vPrev := l.vB.AtVec(i)

		mVal := beta1*mPrev + (1-beta1)*grad
		vVal := beta2*vPrev + (1-beta2)*grad*grad

		l.mB.SetVec(i, mVal)
		l.vB.SetVec(i, vVal)

		mHat := mVal / (1 - math.Pow(beta1, float64(l.t)))
		vHat := vVal / (1 - math.Pow(beta2, float64(l.t)))

		update := mHat / (math.Sqrt(vHat) + epsilon)
		l.Bias.SetVec(i, l.Bias.AtVec(i)-l.learningRate*update)
	}
}
