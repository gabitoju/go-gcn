# Go-GCN

Go-GCN is an implementation of Graph Convolutional Networks (GCN) in Go. It allows for semi-supervised learning on graph-structured data, such as citation networks. The project is designed with modular components for easy customization and experimentation, including support for dropout, Adam optimizer, and accuracy evaluation.

This implementation is based on the paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) by Thomas Kipf and Max Welling.

## Features

- Graph Convolutional Network (GCN) architecture
- Supports both SGD and Adam optimizers
- Dropout regularization to prevent overfitting
- Accuracy evaluation metrics
- Utility functions for matrix operations, activation functions, and loss calculations
- Data loading and preprocessing for the Cora dataset


## Installation

1. Clone the repository:

```bash
git clone https://github.com/gabitoju/go-gcn.git
cd go-gcn
```

2. Install the dependencies:

```bash
go mod tidy
```

## Usage

### Running the Model

The project is configured to run on the Cora dataset, located in the `datasets/cora` directory. Use the following command to run the model:

```bash
go run cmd/main.go
```
This will load the Cora dataset, create a GCN model, train it, and output the accuracy and loss during training.

### Example code

```go
package main

import (
    "github.com/gabitoju/go-gcn/internal/data"
    "github.com/gabitoju/go-gcn/internal/model"
    "github.com/gabitoju/go-gcn/internal/train"
    "github.com/gabitoju/go-gcn/internal/utils"
)

func main() {
    utils.InitializeRand(42) // Initialize random seed

    // Load Cora dataset
    features, adj, labels := data.LoadData("../datasets/cora", "cora")

    // Create train, validation, and test splits
    trn, valid, test := data.CreateDataSplit(140, 500, 1000, len(labels))

    // Configure training settings
    t := train.TrainConfig{
        Epochs:       1000,
        Labels:       labels,
        TrainMask:    trn,
        ValidMask:    valid,
        TestMask:     test,
        LearningRate: 0.001,
        WeightDecay:  5e-4,
    }

    // Initialize the GCN model
    gcn := model.NewGCN(2, len(features[0]), 16, 7, 0.5, t.LearningRate)

    // Train the model
    t.Train(gcn, features, adj)
}
```

## Dataset

The project uses the Cora dataset, a common benchmark for graph-based learning tasks. The dataset is located in the `datasets/cora` directory:

 * `cora.content`: Contains the node features and labels.
 * `cora.cites`: Contains the citation graph edges between nodes.

### Data Loading

The `LoadData` function in `internal/data/load.go` is responsible for loading and processing the dataset into adjacency matrices and feature matrices.

## Model Structure

### GCN

The GCN model is implemented in `internal/model/gcn.go`. It supports multiple layers, dropout regularization, and training with either the Adam or SGD optimizer.

### Layers

Each layer in the GCN is implemented in `internal/model/layer.go`, with support for forward and backward passes, and weight updates using either Adam or SGD.

## Training

The training loop is implemented in `internal/train/train.go`. It supports accuracy evaluation and outputs the loss and accuracy metrics for both training and validation datasets during each epoch. You can configure the number of epochs, learning rate, and weight decay in the `TrainConfig` struct.

## Evaluation

Accuracy is calculated using the `Accuracy` function in `internal/utils/evaluation.go`. After each epoch, the model's accuracy and loss on both the training and validation datasets are printed.

## Future Improvements

* Additional Evaluation Metrics: Add precision, recall, and F1-score evaluation metrics.
* More Datasets: Support for loading and training on additional graph-based datasets.
* Batch Training: Implement batch training for larger datasets.
* Hyperparameter Tuning: Add functionality for automatic hyperparameter tuning.
* CUDA and Metal Support: Add GPU support through CUDA for Nvidia GPUs and Metal for Apple devices to accelerate training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
