package main

import "fmt"

func main() {

	inputs := []float64{1, 2, 3, 2.5}

	weights := [][]float64{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}

	bias := []float64{2.0, 3.0, 0.5}

	layerOutputs := make([]float64, len(weights))

	for i := 0; i < len(weights); i++ {

		for j := 0; j < len(inputs); j++ {
			layerOutputs[i] += inputs[j] * weights[i][j]
		}

		layerOutputs[i] += bias[i]
	}

	fmt.Printf("%v\n", layerOutputs)

}
