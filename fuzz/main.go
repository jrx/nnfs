package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

func main() {

	rawA := []float64{1, 2, 3}
	rawB := []float64{2, 3, 4}

	a := t.New(t.WithShape(1, 3), t.WithBacking(rawA))
	b := t.New(t.WithShape(1, 3), t.WithBacking(rawB))

	b.T()

	dotProduct, err := t.Dot(a, b)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("%v\n", dotProduct)
}
