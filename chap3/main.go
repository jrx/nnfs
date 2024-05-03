package main

import (
	"fmt"

	ml "github.com/jrx/nnfs/chap3/MachineLearning"
	p "github.com/jrx/nnfs/chap3/Plotting"
	t "gorgonia.org/tensor"
)

func main() {

	// Inputs
	rawInputs := []float64{
		1, 2, 3, 2.5,
		2.0, 5.0, -1.0, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}
	inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))

	// Layer 1
	rawWeights := []float64{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}
	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))

	biases := []float64{2.0, 3.0, 0.5}

	weights.T()

	// Layer 2
	rawWeights2 := []float64{
		0.1, -0.14, 0.5,
		-0.5, 0.12, -0.33,
		-0.44, 0.73, -0.13,
	}
	weights2 := t.New(t.WithShape(3, 3), t.WithBacking(rawWeights2))

	biases2 := []float64{-1, 2, -0.5}

	weights2.T()

	// Forward pass 1
	dotProduct, err := t.Dot(inputs, weights)
	if err != nil {
		fmt.Println(err)
	}

	layerOutputs1, err := ml.AddBiases(dotProduct, biases)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("Output Layer 1:\n %v\n", layerOutputs1)

	// Forward pass 2
	dotProduct2, err := t.Dot(layerOutputs1, weights2)
	if err != nil {
		fmt.Println(err)
	}

	layerOutputs2, err := ml.AddBiases(dotProduct2, biases2)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("Output Layer 2:\n %v\n", layerOutputs2)

	// Plotting
	p.PlotData(p.X, "plot.png", p.Y, p.CMap)
}
