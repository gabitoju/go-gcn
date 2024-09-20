package data

import (
	"testing"
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
