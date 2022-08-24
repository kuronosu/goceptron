package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"

	"github.com/kuronosu/goceptron/goceptron"
)

func main() {
	X, Y, err := GetIrisDataSet()
	if err != nil {
		log.Fatalln(err)
	}
	rand.Seed(time.Now().UTC().UnixNano())
	p, err := goceptron.NewPerceptron(2, nil, nil, &goceptron.Step)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println("Initial weights:", p.Weights(), p.Bias())
	err = p.Train(X, Y)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println("Trained weights:", p.Weights(), p.Bias())
	res0, _ := p.Predict([]float64{1.0, 0.75})
	res1, _ := p.Predict([]float64{4.5, 1.25})
	fmt.Println("Prediction setosa {1.5, 0.25}:", res0)
	fmt.Println("Prediction versicolor {4.5, 1.25}:", res1)

	PlotPoints(X, Y)
	PlotPredictions(*p)
}

func IrisPoints(nx, ny int, p goceptron.Perceptron) (plotter.XYs, plotter.XYs) {
	ptsSetosa := make(plotter.XYs, 0)
	ptsVersicolor := make(plotter.XYs, 0)
	dx := 6 / float64(nx)
	dy := 2 / float64(ny)
	for x := 0.0; x < 6; x += dx {
		for y := 0.0; y < 2; y += dy {
			_x := float64(x)
			_y := float64(y)
			res, _ := p.Predict([]float64{_x, _y})
			if res == 0 {
				ptsSetosa = append(ptsSetosa, plotter.XY{X: _x, Y: _y})
			} else {
				ptsVersicolor = append(ptsVersicolor, plotter.XY{X: _x, Y: _y})
			}
		}
	}
	return ptsSetosa, ptsVersicolor
}

func PlotPoints(X [][]float64, Y []float64) {
	setosa := make(plotter.XYs, 0)
	versicolor := make(plotter.XYs, 0)

	for idx, label := range Y {
		if label == 0 {
			setosa = append(setosa, plotter.XY{X: X[idx][0], Y: X[idx][1]})
		} else {
			versicolor = append(versicolor, plotter.XY{X: X[idx][0], Y: X[idx][1]})
		}
	}

	plt := plot.New()
	plt.Title.Text = "Iris Data Set"
	plt.X.Label.Text = "Petal length"
	plt.Y.Label.Text = "Petal width"
	err := plotutil.AddScatters(plt,
		"Setosa", setosa,
		"Versicolor", versicolor,
	)
	if err != nil {
		panic(err)
	}
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, "iris.png"); err != nil {
		panic(err)
	}
}

func PlotPredictions(p goceptron.Perceptron) {
	plt := plot.New()
	plt.Title.Text = "Iris Data Set"
	plt.X.Label.Text = "Petal length"
	plt.Y.Label.Text = "Petal width"
	points := 50
	setosa, versicolor := IrisPoints(points, points, p)
	err := plotutil.AddScatters(plt,
		"Setosa", setosa,
		"Versicolor", versicolor,
	)
	if err != nil {
		panic(err)
	}
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, "predictions.png"); err != nil {
		panic(err)
	}
}

func SavePredictions(p goceptron.Perceptron) {
	csvFile, err := os.Create("results.csv")

	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	defer csvFile.Close()
	csvwriter := csv.NewWriter(csvFile)
	for y := 0.0; y <= 2.00; y += 0.01 {
		for x := 0.0; x <= 6.00; x += 0.01 {
			t, err := p.Predict([]float64{x, y})
			if err != nil {
				continue
			}
			// data = append(data, Iris{x, y, t})
			data := []string{
				fmt.Sprintf("%.2f", x),
				fmt.Sprintf("%.2f", y),
				fmt.Sprintf("%.0f", t),
			}
			csvwriter.Write(data)
		}
	}
	csvwriter.Flush()
}

func GetIrisDataSet() ([][]float64, []float64, error) {
	response, err := http.Get("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
	if err != nil {
		return nil, nil, err
	}
	defer response.Body.Close()
	reader := csv.NewReader(response.Body)
	var X [][]float64
	var Y []float64
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if record[4] != "Versicolor" && record[4] != "Setosa" {
			continue
		}
		var _x1, _x2 float64
		if _x1, err = strconv.ParseFloat(record[2], 64); err != nil {
			continue
		}
		if _x2, err = strconv.ParseFloat(record[3], 64); err != nil {
			continue
		}
		X = append(X, []float64{_x1, _x2})
		if record[4] == "Setosa" {
			Y = append(Y, 0)
		} else {
			Y = append(Y, 1)
		}
	}
	return X, Y, err
}
