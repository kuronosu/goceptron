package goceptron

func SE(output, target float64) float64 {
	return (output - target) * (output - target)
}

func MSE(output, target []float64) float64 {
	if len(output) != len(target) {
		return 0
	}
	var s float64
	for i := range output {
		s += SE(output[i], target[i])
	}
	return 1.0 / float64(len(output)) * s
}
