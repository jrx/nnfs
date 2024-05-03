package main

import (
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

func main() {
	plotData(X, "plot.png")
}

func plotData(data [][]float64, path string) {
	p := plot.New()

	xys := make(plotter.XYs, len(data))

	for i := range xys {
		xys[i].X = data[i][0]
		xys[i].Y = data[i][1]
	}

	scatter, err := plotter.NewScatter(xys)
	handleErr(err)

	p.Add(scatter)

	wt, err := p.WriterTo(300, 300, "png")
	handleErr(err)

	f, err := os.Create(path)
	handleErr(err)

	defer func() {
		err := f.Close()
		handleErr(err)
	}()

	wt.WriteTo(f)
}

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}
