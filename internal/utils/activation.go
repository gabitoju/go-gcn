package utils

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func relu_derivative(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}
