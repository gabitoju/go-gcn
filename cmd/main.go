package main

import (
	"github.com/gabitoju/go-gcn/internal/data"
	"github.com/gabitoju/go-gcn/internal/model"
	"github.com/gabitoju/go-gcn/internal/train"
	"github.com/gabitoju/go-gcn/internal/utils"
)

func main() {

	utils.InitializeRand(42)

	features, adj, labels := data.LoadData("../datasets/cora", "cora")

	trn, valid, test := data.CreateDataSplit(140, 500, 1000, len(labels))

	t := train.TrainConfig{
		Epochs:       1000,
		Labels:       labels,
		TrainMask:    trn,
		ValidMask:    valid,
		TestMask:     test,
		LearningRate: 0.001,
		WeightDecay:  5e-4,
	}

	gcn := model.NewGCN(2, len(features[0]), 16, 7, 0.5, t.LearningRate)
	t.Train(gcn, features, adj)

}
