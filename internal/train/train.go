package train

import (
	"fmt"

	"github.com/gabitoju/go-gcn/internal/model"
	"github.com/gabitoju/go-gcn/internal/utils"
)

func Train(gcn *model.GCN, features [][]float64, adj [][]float64, labels []int32, epochs int, trainMask []int) {

	gcn.Train()

	trnLabels := make([]int32, len(trainMask))
	for i := range trnLabels {
		trnLabels[i] = labels[trainMask[i]]
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		trainEpoch(gcn, features, adj, trnLabels, trainMask, epoch)
	}
}

func trainEpoch(gcn *model.GCN, features [][]float64, adj [][]float64, labels []int32, trainMask []int, epoch int) {
	output := gcn.Forward(features, adj)

	outputTrn := make([][]float64, len(trainMask))

	for i, idx := range trainMask {
		outputTrn[i] = output[idx]
	}

	loss := utils.CrossEntropyLoss(outputTrn, labels)
	grad := utils.CrossEntropyLossDerivative(output, labels)

	gcn.Backward(grad)

	for _, layer := range gcn.Layers {
		layer.AdamUpdate(0.9, 0.999, 1e-8)
	}

	fmt.Printf("Epoch: %d, Loss: %.4f (%v)\n", epoch, loss, loss)

}
