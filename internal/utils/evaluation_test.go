package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAccuracy(t *testing.T) {
	tests := []struct {
		name   string
		yPred  *mat.Dense
		yTrue  []int32
		expect float64
	}{
		{
			name: "perfect_match",
			yPred: mat.NewDense(2, 2, []float64{
				0.1, 0.9,
				0.8, 0.2,
			}),
			yTrue:  []int32{1, 0},
			expect: 1,
		},
		{
			name: "multi_class",
			yPred: mat.NewDense(3, 4, []float64{
				0.1, 0.01, 0.07, 0.82,
				0.8, 0.1, 0.05, 0.05,
				0.02, 0.9, 0.07, 0.01,
			}),
			yTrue:  []int32{3, 0, 1},
			expect: 1,
		},
		{
			name: "half_correct",
			yPred: mat.NewDense(2, 2, []float64{
				0.1, 0.9,
				0.8, 0.2,
			}),
			yTrue:  []int32{0, 0},
			expect: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Accuracy(tt.yPred, tt.yTrue)
			if got != tt.expect {
				t.Fatalf("Accuracy() = %v, want %v", got, tt.expect)
			}
		})
	}
}
