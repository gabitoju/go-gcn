package utils

import (
	"fmt"
	"testing"
)

func TestShape(t *testing.T) {
	tests := []struct {
		name string
		x    interface{}
		want string
	}{
		{
			name: "2x3",
			x:    [][]float64{{1, 2, 3}, {4, 5, 6}},
			want: "[2 3]",
		},
		{
			name: "3x2",
			x:    [][]float64{{1, 2}, {3, 4}, {5, 6}},
			want: "[3 2]",
		},
		{
			name: "3x3x3",
			x: [][][]float64{
				{
					{255, 0, 0}, {0, 255, 0}, {0, 0, 255},
				},
				{
					{255, 255, 0}, {0, 255, 255}, {255, 0, 255},
				},
				{
					{192, 192, 192}, {128, 128, 128}, {64, 64, 64},
				}},
			want: "[3 3 3]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Shape(tt.x); fmt.Sprint(got) != tt.want {
				t.Errorf("Shape() = %v, want %v", got, tt.want)
			}
		})
	}
}
