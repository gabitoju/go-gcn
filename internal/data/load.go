package data

import (
	"encoding/csv"
	"os"
	"strconv"
)

func LoadData(path, dataset string) ([][]int32, [][]float64, []int32) {

	contentPath := path + "/" + dataset + ".content"
	edgePath := path + "/" + dataset + ".cites"

	contentFile, err := os.Open(contentPath)
	if err != nil {
		panic(err)
	}
	defer contentFile.Close()

	csvReader := csv.NewReader(contentFile)
	csvReader.Comma = '\t'

	labels := make([]string, 0)
	indices := make(map[int]int)
	features := make([][]float64, 0)

	for {
		record, err := csvReader.Read()
		if err != nil {
			break
		}
		ix, _ := strconv.Atoi(record[0])
		indices[ix] = len(indices)
		labels = append(labels, record[len(record)-1])
		nodeFeatures := record[1 : len(record)-1]
		nodeFeaturesFloat := make([]float64, len(nodeFeatures))
		for i, f := range nodeFeatures {
			nodeFeaturesFloat[i], _ = strconv.ParseFloat(f, 64)
		}
		features = append(features, nodeFeaturesFloat)
	}

	encoded_labels := EncodeOneHot(labels)

	edgeFile, err := os.Open(edgePath)
	if err != nil {
		panic(err)
	}
	defer edgeFile.Close()

	csvReader = csv.NewReader(edgeFile)
	csvReader.Comma = '\t'

	adj := make([][]int32, len(indices))
	for i := range adj {
		adj[i] = make([]int32, len(indices))
	}
	for {
		record, err := csvReader.Read()
		if err != nil {
			break
		}
		id1, _ := strconv.Atoi(record[0])
		id2, _ := strconv.Atoi(record[1])

		ix1 := indices[id1]
		ix2 := indices[id2]

		adj[ix1][ix2] = 1
		adj[ix2][ix1] = 1
	}

	return adj, features, encoded_labels
}

func EncodeOneHot(labels []string) []int32 {
	classMap := make(map[string]int)
	ix := 0
	totalRecords := len(labels)
	for _, label := range labels {
		if _, ok := classMap[label]; !ok {
			classMap[label] = ix
			ix += 1
		}
	}

	oneHotLabels := make([]int32, totalRecords)
	for i, label := range labels {
		oneHotLabels[i] = int32(classMap[label])
	}

	return oneHotLabels
}
