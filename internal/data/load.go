package data

import (
	"encoding/csv"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"

	"github.com/gabitoju/go-gcn/internal/utils"
)

func LoadData(path, dataset string) (*mat.Dense, *SparseMatrix, []int32) {

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
	var featuresData []float64
	featureLen := 0

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
		if featureLen == 0 {
			featureLen = len(nodeFeaturesFloat)
		}
		featuresData = append(featuresData, nodeFeaturesFloat...)
	}

	encoded_labels := EncodeOneHot(labels)

	edgeFile, err := os.Open(edgePath)
	if err != nil {
		panic(err)
	}
	defer edgeFile.Close()

	csvReader = csv.NewReader(edgeFile)
	csvReader.Comma = '\t'

	edges := make([][2]int, 0)
	for {
		record, err := csvReader.Read()
		if err != nil {
			break
		}
		id1, _ := strconv.Atoi(record[0])
		id2, _ := strconv.Atoi(record[1])

		ix1 := indices[id1]
		ix2 := indices[id2]

		edges = append(edges, [2]int{ix1, ix2}, [2]int{ix2, ix1})
	}

	featuresMat := mat.NewDense(len(labels), featureLen, featuresData)

	return featuresMat, NewSparseMatrix(len(indices), len(indices), edges), encoded_labels
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

func CreateDataSplit(trainSize, validationSize, testSize, size int) ([]int, []int, []int) {
	indices := make([]int, size)
	for i := 0; i < size; i++ {
		indices[i] = i
	}

	indices = utils.ShuffleInts(size, indices)

	trainIndices := indices[:trainSize]
	validationIndices := indices[trainSize : trainSize+validationSize]
	testIndices := indices[trainSize+validationSize : trainSize+validationSize+testSize]

	return trainIndices, validationIndices, testIndices
}
