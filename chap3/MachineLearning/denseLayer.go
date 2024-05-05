package MachineLearning

import t "gorgonia.org/tensor"

type LayerDense struct {
	Weights t.Tensor
	Biases  []float64

	Output t.Tensor
}

func NewLayerDense(nInputs, nNeurons int) LayerDense {
	weights := t.New(
		t.WithShape(nInputs, nNeurons),
		t.WithBacking(t.Random(t.Float64, nInputs*nNeurons)),
	)
	weights, err := weights.MulScalar(0.01, false)
	handleErr(err)

	biases := make([]float64, nNeurons)
	return LayerDense{weights, biases, nil}
}

func (l *LayerDense) Forward(inputs t.Tensor) {
	dp, err := t.Dot(inputs, l.Weights)
	handleErr(err)

	if len(dp.Shape()) == 1 {
		err = dp.Reshape(dp.Shape()[0], 1)
		handleErr(err)
	}

	l.Output, err = AddBiases(dp, l.Biases)
	handleErr(err)
}

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}
