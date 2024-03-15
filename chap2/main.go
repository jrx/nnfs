package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

func main() {

	inputs := t.New(t.WithBacking([]float64{1, 2, 3, 2.5}))

	rawWeights := []float64{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}

	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))

	biases := t.New(t.WithBacking([]float64{2.0, 3.0, 0.5}))

	dotProduct, err := t.Dot(weights, inputs)
	if err != nil {
		fmt.Println(err)
	}

	layerOutputs, err := t.Add(dotProduct, biases)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("%v\n", layerOutputs)
}
