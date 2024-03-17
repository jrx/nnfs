package MachineLearning

import (
	t "gorgonia.org/tensor"
)

func AddBiases(inputs t.Tensor, biases []float64) (t.Tensor, error) {

	newShape := inputs.Shape()

	nB := make([]float64, newShape[0]*newShape[1])
	for i := range nB {
		nB[i] = biases[i%len(biases)]
	}

	b := t.New(t.WithShape(newShape...), t.WithBacking(nB))
	return t.Add(inputs, b)
}
