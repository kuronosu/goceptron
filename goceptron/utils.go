package goceptron

import "math/rand"

func FandFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
