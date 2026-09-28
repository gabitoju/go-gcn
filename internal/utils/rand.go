package utils

import (
	"math/rand"
	"time"
)

var r *rand.Rand

func InitializeRand(seed int64) {
	r = rand.New(rand.NewSource(seed))
}

func ensureRand() {
	if r == nil {
		InitializeRand(time.Now().UnixNano())
	}
}

func RandFloat64() float64 {
	ensureRand()
	return r.Float64()
}

func ShuffleInts(size int, a []int) []int {
	ensureRand()
	r.Shuffle(size, func(i, j int) {
		a[i], a[j] = a[j], a[i]
	})
	return a
}
