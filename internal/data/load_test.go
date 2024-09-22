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
		train_size int
		test_size  int
		val_size   int
		total_size int
	}{
		{
			name:       "simple_split",
			train_size: 140,
			test_size:  500,
			val_size:   1000,
			total_size: 2708,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			train, tst, val := CreateDataSplit(test.train_size, test.test_size, test.val_size, test.total_size)
			if len(train) != test.train_size {
				t.Errorf("CreateDataSplit(%v, %v, %v, %v) = %v; want %v", test.train_size, test.test_size, test.val_size, test.total_size, len(train), test.train_size)
			}
			if len(tst) != test.test_size {
				t.Errorf("CreateDataSplit(%v, %v, %v, %v) = %v; want %v", test.train_size, test.test_size, test.val_size, test.total_size, len(tst), test.test_size)
			}
			if len(val) != test.val_size {
				t.Errorf("CreateDataSplit(%v, %v, %v, %v) = %v; want %v", test.train_size, test.test_size, test.val_size, test.total_size, len(val), test.val_size)
			}
		})
	}
}
