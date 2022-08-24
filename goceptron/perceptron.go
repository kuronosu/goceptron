package goceptron

import (
	"fmt"
)

var DefaultAF = Step

type Perceptron struct {
	weights    []float64
	bias       float64
	Activation AF
	Lambda     float64
}

func (p *Perceptron) Train(inputs [][]float64, output []float64) error {
	if len(inputs) == 0 {
		return fmt.Errorf("inputs is empty")
	}
	if len(inputs) != len(output) {
		return fmt.Errorf("input size %d does not match output size %d", len(inputs), len(output))
	}
	for _i := 0; _i < 1000; _i++ {
		for idx, input := range inputs {
			result, err := p.Predict(input)
			if err != nil {
				return err
			}
			if err := p.optimize(input, output[idx], result); err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *Perceptron) optimize(inputs []float64, expected, output float64) error {
	if len(inputs) != len(p.weights) {
		return fmt.Errorf("input size %d does not match weights size %d", len(inputs), len(p.weights))
	}
	for i := range inputs {
		p.weights[i] += p.Lambda * inputs[i] * (expected - output)
	}
	p.bias += p.Lambda * (expected - output)
	return nil
}

func (p *Perceptron) Predict(input []float64) (float64, error) {
	output, err := p.transfer(input)
	if err != nil {
		return 0, err
	}
	return p.Activation.F(output), nil
}

func (p *Perceptron) transfer(input []float64) (float64, error) {
	if len(input) != len(p.weights) {
		return 0, fmt.Errorf("input size %d does not match weights size %d", len(input), len(p.weights))
	}
	sum := 0.0
	for i := range input {
		sum += input[i] * p.weights[i]
	}
	sum += p.bias
	return sum, nil
}

func (p *Perceptron) randomizeWeights() {
	for i := range p.weights {
		p.weights[i] = randFloat(-1.0, 1.0)
	}
}

func (p *Perceptron) InitWeights(weights []float64) error {
	if weights == nil {
		p.randomizeWeights()
		return nil
	}
	if len(weights) != len(p.weights) {
		return fmt.Errorf("weights size %d does not match perceptron size %d", len(weights), len(p.weights))
	}
	p.weights = weights
	return nil
}

func (p *Perceptron) InitBias(bias float64) {
	p.bias = bias
}

func (p *Perceptron) Weights() []float64 {
	return p.weights
}

func (p *Perceptron) Bias() float64 {
	return p.bias
}

func NewPerceptron(size int, bias *float64, weights []float64, activation *AF) (*Perceptron, error) {
	_a := activation
	if _a == nil {
		_a = &DefaultAF
	}
	p := &Perceptron{
		weights:    make([]float64, size),
		bias:       0.0,
		Activation: *_a,
		Lambda:     0.5,
	}
	if bias != nil {
		p.InitBias(*bias)
	}
	if err := p.InitWeights(weights); err != nil {
		return nil, err
	}
	return p, nil
}
