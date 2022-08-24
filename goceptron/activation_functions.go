package goceptron

import "math"

type AF struct {
	F func(float64) float64
	D func(float64) float64
}

func SigmoidF(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidD(x float64) float64 {
	_f := SigmoidF(x)
	return _f * (1.0 - _f)
}

func ReLUF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func ReLUD(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func StepF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func StepD(x float64) float64 {
	return 0
}

var Sigmoid = AF{
	F: SigmoidF,
	D: SigmoidD,
}

var ReLU = AF{
	F: ReLUF,
	D: ReLUD,
}

var Step = AF{
	F: StepF,
	D: StepD,
}
