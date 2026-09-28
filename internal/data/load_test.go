package data

import (
	"testing"

	"github.com/gabitoju/go-gcn/internal/utils"
)

func TestEncodeOneHot(t *testing.T) {

	tests := []struct {
		name     string
		labels   []string
		expected []int32
	}{
		{
			labels: []string{"Neural_Networks", "Rule_Learning", "Reinforcement_Learning", "Reinforcement_Learning", "Reinforcement_Learning", "Probabilistic_Methods", "Probabilistic_Methods", "Theory", "Neural_Networks"},
			expected: []int32{
				0, 1, 2, 2, 2, 3, 3, 4, 0,
			},
		}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := EncodeOneHot(test.labels)
			for i := range actual {
				if actual[i] != test.expected[i] {
					t.Errorf("EncodeOneHot(%v) = %v; want %v", test.labels, actual, test.expected)
				}
			}
		})
	}
}

func TestCreateDataSplit(t *testing.T) {
	utils.InitializeRand(42)
	tests := []struct {
		name       string
		trainSize  int
		valSize    int
		testSize   int
		total_size int
	}{
		{
			name:       "simple_split",
			trainSize:  140,
			valSize:    500,
			testSize:   1000,
			total_size: 2708,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			train, val, tst := CreateDataSplit(test.trainSize, test.valSize, test.testSize, test.total_size)
			if len(train) != test.trainSize {
				t.Errorf("CreateDataSplit(%v, %v, %v, %v) = %v; want %v", test.trainSize, test.valSize, test.testSize, test.total_size, len(train), test.trainSize)
			}
			if len(val) != test.valSize {
				t.Errorf("CreateDataSplit(%v, %v, %v, %v) = %v; want %v", test.trainSize, test.valSize, test.testSize, test.total_size, len(val), test.valSize)
			}
			if len(tst) != test.testSize {
				t.Errorf("CreateDataSplit(%v, %v, %v, %v) = %v; want %v", test.trainSize, test.valSize, test.testSize, test.total_size, len(tst), test.testSize)
			}
		})
	}
}
