package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

func main() {

	rawInputs := []float64{
		1, 2, 3, 2.5,
		2.0, 5.0, -1.0, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}
	inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))

	rawWeights := []float64{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}
	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))

	biases := []float64{2.0, 3.0, 0.5}

	weights.T()

	dotProduct, err := t.Dot(inputs, weights)
	if err != nil {
		fmt.Println(err)
	}

	layerOutputs, err := AddBiases(dotProduct, biases)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("output: %v\n", layerOutputs)
}
