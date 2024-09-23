package utils

import "testing"

func TestAccuracy(t *testing.T) {
	tests := []struct {
		name   string
		y_pred [][]float64
		y_true []int32
		want   float64
	}{
		{
			name: "test_accuracy_1",
			y_pred: [][]float64{
				{0.1, 0.9},
				{0.8, 0.2},
			},
			y_true: []int32{1, 0},
			want:   1,
		},
		{
			name: "test_accuracy_2",
			y_pred: [][]float64{
				{0.1, 0.01, 0.07, 0.82},
				{0.8, 0.1, 0.05, 0.05},
				{0.02, 0.9, 0.07, 0.01},
			},
			y_true: []int32{3, 0, 1},
			want:   1,
		},
		{
			name: "test_accuracy_3",
			y_pred: [][]float64{
				{0.1, 0.9},
				{0.8, 0.2},
			},
			y_true: []int32{0, 0},
			want:   0.5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Accuracy(tt.y_pred, tt.y_true); got != tt.want {
				t.Errorf("Accuracy() = %v, want %v", got, tt.want)
			}
		})
	}
}
