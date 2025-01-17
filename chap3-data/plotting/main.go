package main

import (
	"image/color"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

func main() {
	plotData(X, "plot.png", Y, CMap)
}

func plotData(data [][]float64, path string, col []float64, cmap map[float64]color.Color) {
	p := plot.New()

	for key, element := range cmap {
		xys := plotter.XYs{}
		for i := range data {
			if col[i] == key {
				xys = append(xys, plotter.XY{X: data[i][0], Y: data[i][1]})
			}
		}

		scatter, err := plotter.NewScatter(xys)
		handleErr(err)

		scatter.Color = element

		p.Add(scatter)
	}

	wt, err := p.WriterTo(450, 400, "png")
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
