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

	trn, _, _ := data.CreateDataSplit(140, 500, 1000, len(labels))

	gcn := model.NewGCN(2, len(features[0]), 16, 7, 0.5)

	train.Train(gcn, features, adj, labels, 1000, trn)

}
