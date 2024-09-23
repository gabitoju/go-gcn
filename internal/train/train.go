package train

import (
	"fmt"

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

func (t *TrainConfig) Train(gcn *model.GCN, features [][]float64, adj [][]float64) {

	trnLabels := make([]int32, len(t.TrainMask))
	validLabels := make([]int32, len(t.ValidMask))

	for i := range trnLabels {
		trnLabels[i] = t.Labels[t.TrainMask[i]]
	}

	for i := range validLabels {
		validLabels[i] = t.Labels[t.ValidMask[i]]
	}

	t.TrainLabels = trnLabels
	t.ValidLabels = validLabels

	for epoch := 1; epoch <= t.Epochs; epoch++ {
		t.trainEpoch(gcn, features, adj, epoch)
	}
}

func (t *TrainConfig) trainEpoch(gcn *model.GCN, features [][]float64, adj [][]float64, epoch int) {

	gcn.Train()

	output := gcn.Forward(features, adj)

	outputTrn := make([][]float64, len(t.TrainMask))
	outputValid := make([][]float64, len(t.ValidMask))

	for i, idx := range t.TrainMask {
		outputTrn[i] = output[idx]
	}

	loss := utils.CrossEntropyLoss(outputTrn, t.TrainLabels)
	trainAcc := utils.Accuracy(outputTrn, t.TrainLabels)

	grad := utils.CrossEntropyLossDerivative(output, t.TrainLabels)

	gcn.Backward(grad)

	for _, layer := range gcn.Layers {
		layer.AdamUpdate(0.9, 0.999, 1e-8, t.WeightDecay)
	}

	gcn.Eval()
	output = gcn.Forward(features, adj)

	for i, idx := range t.ValidMask {
		outputValid[i] = output[idx]
	}

	validLoss := utils.CrossEntropyLoss(outputValid, t.ValidLabels)
	validAcc := utils.Accuracy(outputValid, t.ValidLabels)

	fmt.Printf("Epoch: %d, Loss: %.4f, Accuracy: %.4f, Validation Loss: %.4f Validation Accuracy: %.4f\n", epoch, loss, trainAcc, validLoss, validAcc)

}
