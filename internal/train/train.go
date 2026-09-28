package train

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"github.com/gabitoju/go-gcn/internal/data"
	"github.com/gabitoju/go-gcn/internal/model"
	"github.com/gabitoju/go-gcn/internal/utils"
)

type TrainConfig struct {
	Epochs       int
	Labels       []int32
	TrainMask    []int
	ValidMask    []int
	TestMask     []int
	TrainLabels  []int32
	ValidLabels  []int32
	TestLabels   []int32
	LearningRate float64
	WeightDecay  float64
}

func (t *TrainConfig) Train(gcn *model.GCN, features, adj *mat.Dense) {
	t.train(gcn, features, func() *mat.Dense { return gcn.Forward(features, adj) })
}

func (t *TrainConfig) TrainSparse(gcn *model.GCN, features *mat.Dense, adj *data.SparseMatrix) {
	t.train(gcn, features, func() *mat.Dense { return gcn.ForwardSparse(features, adj) })
}

func (t *TrainConfig) train(gcn *model.GCN, features *mat.Dense, forward func() *mat.Dense) {

	trnLabels := make([]int32, len(t.TrainMask))
	validLabels := make([]int32, len(t.ValidMask))
	testLabels := make([]int32, len(t.TestMask))

	for i := range trnLabels {
		trnLabels[i] = t.Labels[t.TrainMask[i]]
	}

	for i := range validLabels {
		validLabels[i] = t.Labels[t.ValidMask[i]]
	}
	for i := range testLabels {
		testLabels[i] = t.Labels[t.TestMask[i]]
	}

	t.TrainLabels = trnLabels
	t.ValidLabels = validLabels
	t.TestLabels = testLabels

	for epoch := 1; epoch <= t.Epochs; epoch++ {
		t.trainEpoch(gcn, forward, epoch)
	}

	if len(t.TestMask) > 0 {
		gcn.Eval()
		output := forward()
		outputTest := selectRows(output, t.TestMask)
		fmt.Printf("Test Loss: %.4f Test Accuracy: %.4f\n", utils.CrossEntropyLoss(outputTest, t.TestLabels), utils.Accuracy(outputTest, t.TestLabels))
	}
}

func (t *TrainConfig) trainEpoch(gcn *model.GCN, forward func() *mat.Dense, epoch int) {

	gcn.Train()

	output := forward()

	outputTrn := selectRows(output, t.TrainMask)
	loss := utils.CrossEntropyLoss(outputTrn, t.TrainLabels)
	trainAcc := utils.Accuracy(outputTrn, t.TrainLabels)

	grad := utils.CrossEntropyLossDerivative(output, t.TrainLabels, t.TrainMask)

	gcn.Backward(grad)

	for _, layer := range gcn.Layers {
		layer.AdamUpdate(0.9, 0.999, 1e-8, t.WeightDecay)
	}

	gcn.Eval()
	output = forward()

	outputValid := selectRows(output, t.ValidMask)

	validLoss := utils.CrossEntropyLoss(outputValid, t.ValidLabels)
	validAcc := utils.Accuracy(outputValid, t.ValidLabels)

	fmt.Printf("Epoch: %d, Loss: %.4f, Accuracy: %.4f, Validation Loss: %.4f Validation Accuracy: %.4f\n", epoch, loss, trainAcc, validLoss, validAcc)

}

func selectRows(src *mat.Dense, indices []int) *mat.Dense {
	if len(indices) == 0 {
		return mat.NewDense(0, 0, nil)
	}
	_, cols := src.Dims()
	out := mat.NewDense(len(indices), cols, nil)
	for i, idx := range indices {
		for j := 0; j < cols; j++ {
			out.Set(i, j, src.At(idx, j))
		}
	}
	return out
}
