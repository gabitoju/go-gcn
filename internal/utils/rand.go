package utils

import "math/rand"

var r *rand.Rand

func InitializeRand(seed int64) {
	r = rand.New(rand.NewSource(seed))
}

func RandFloat64() float64 {
	return r.Float64()
}

func ShuffleInts(size int, a []int) []int {
	r.Shuffle(size, func(i, j int) {
		a[i], a[j] = a[j], a[i]
	})
	return a
}
