package MachineLearning

import (
	t "gorgonia.org/tensor"
)

type ActivationReLU struct {
	Output t.Tensor
}

func NewActivationReLU() ActivationReLU {
	return ActivationReLU{Output: t.New(t.Of(t.Float64))}
}

func (r *ActivationReLU) Forward(inputs t.Tensor) {

	zeros := t.New(t.WithShape(inputs.Shape()...), t.Of(t.Float64))

	output, err := t.MaxBetween(inputs, zeros)
	handleErr(err)

	r.Output = output
}
