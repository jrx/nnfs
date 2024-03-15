package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

func main() {

	inputs := t.New(t.WithBacking([]float64{1, 2, 3, 2.5}))
	weights := t.New(t.WithBacking([]float64{0.2, 0.8, -0.5, 1.0}))
	bias := t.New(t.WithBacking([]float64{2.0}))

	dotProduct, err := t.Dot(inputs, weights)
	if err != nil {
		fmt.Println(err)
	}

	output, err := t.Add(dotProduct, bias)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("%v\n", output)

}
