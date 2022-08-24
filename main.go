package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"

	"github.com/kuronosu/goceptron/goceptron"
)

// Function to generate random cirular data set
func GetCirucularData(cx, cy, r, d float64, n int) [][]float64 {
	var X [][]float64
	dr := 360 / float64(n)
	for i := 0.0; i < 360; i += dr {
		_r := r + goceptron.FandFloat(-d, d)
		x := math.Cos(i)*_r + cx
		y := math.Sin(i)*_r + cy
		X = append(X, []float64{x, y})
	}
	return X
}

func GetCircularDataSet(n int) ([][]float64, []float64) {
	X1 := GetCirucularData(0, 0, 1, 0.1, n)
	X2 := GetCirucularData(0, 0, 2, 0.1, n)
	Y1 := make([]float64, 0)
	Y2 := make([]float64, 0)
	for range X1 {
		Y1 = append(Y1, 0)
	}
	for range X2 {
		Y2 = append(Y2, 1)
	}
	X := append(X1, X2...)
	Y := append(Y1, Y2...)
	return X, Y
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	fmt.Println("Ejemplo entradas no lineales (circular)")
	circular()
	fmt.Println("Ejemplo entradas lineales (iris)")
	iris()
}

func convertCirularDataToInputs(X []float64) []float64 {
	if len(X) != 2 {
		panic(fmt.Sprintf("Invalid data length: %d", len(X)))
	}
	return []float64{X[0], X[1], X[0] * X[0], X[0] * X[1], X[1] * X[1]}
}

func circular() {
	X, Y := GetCircularDataSet(360)
	TX := make([][]float64, 0)
	for _, x := range X {
		_x := convertCirularDataToInputs(x)
		TX = append(TX, _x)
	}
	p, err := goceptron.NewPerceptron(5, nil, nil, &goceptron.Step)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println("Initial weights:", p.Weights(), p.Bias())
	err = p.Train(TX, Y)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Initial weights:", p.Weights(), p.Bias())
	PlotPoints(X, Y, "Circular", "X", "Y", "0", "1", "circular.png")
	PlotCircularPrediction(*p)
}

func PlotCircularPrediction(p goceptron.Perceptron) {
	xi := -3.0
	xf := 3.0
	yi := -3.0
	yf := 3.0
	points := BinaryLabelPoints(50, 50, xi, xf, yi, yf, p, convertCirularDataToInputs)
	pts0 := points[0]
	pts1 := points[1]
	// pts := append(pts0, pts1...)

	plt := plot.New()
	plt.Title.Text = "Iris Data Set"
	plt.X.Label.Text = "Petal length"
	plt.Y.Label.Text = "Petal width"
	err := plotutil.AddScatters(plt,
		"0", pts0,
		"1", pts1,
	)
	if err != nil {
		panic(err)
	}
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, "curcular_predictions.png"); err != nil {
		panic(err)
	}

	// plotutil.AddScatters(p, plotter.XYs{pts})
	// p.Save(10*vg.Centimeter, 10*vg.Centimeter, "circular.png")
}

func iris() {
	X, Y, err := GetIrisDataSet()
	if err != nil {
		log.Fatalln(err)
	}
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

	PlotPoints(X, Y, "Iris", "Petal length", "Petal width", "Setosa", "Versicolor", "iris.png")
	PlotIrisPredictions(*p)
}

func BinaryLabelPoints(nx, ny int, xi, xf, yi, yf float64, p goceptron.Perceptron, transformer func([]float64) []float64) map[int]plotter.XYs {
	pts0 := make(plotter.XYs, 0)
	pts1 := make(plotter.XYs, 0)
	dx := (xf - xi) / float64(nx)
	dy := (yf - yi) / float64(ny)
	for x := xi; x <= xf; x += dx {
		for y := yi; y <= yf; y += dy {
			_x := float64(x)
			_y := float64(y)
			res, _ := p.Predict(transformer([]float64{_x, _y}))
			if res == 0 {
				pts0 = append(pts0, plotter.XY{X: _x, Y: _y})
			} else {
				pts1 = append(pts1, plotter.XY{X: _x, Y: _y})
			}
		}
	}
	points := make(map[int]plotter.XYs, 0)
	points[0] = pts0
	points[1] = pts1
	return points
}

func PlotPoints(X [][]float64, Y []float64, title, labelX, labelY, labelT0, labelT1, filename string) {
	p0 := make(plotter.XYs, 0)
	p1 := make(plotter.XYs, 0)

	for idx, label := range Y {
		if label == 0 {
			p0 = append(p0, plotter.XY{X: X[idx][0], Y: X[idx][1]})
		} else {
			p1 = append(p1, plotter.XY{X: X[idx][0], Y: X[idx][1]})
		}
	}

	plt := plot.New()
	plt.Title.Text = title
	plt.X.Label.Text = labelX
	plt.Y.Label.Text = labelY
	err := plotutil.AddScatters(plt,
		labelT0, p0,
		labelT1, p1,
	)
	if err != nil {
		panic(err)
	}
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, filename); err != nil {
		panic(err)
	}
}

func PlotIrisPredictions(p goceptron.Perceptron) {
	plt := plot.New()
	plt.Title.Text = "Iris Data Set"
	plt.X.Label.Text = "Petal length"
	plt.Y.Label.Text = "Petal width"
	points := BinaryLabelPoints(50, 50, 0, 6, 0, 2, p, func(x []float64) []float64 { return x })
	setosa := points[0]
	versicolor := points[1]
	err := plotutil.AddScatters(plt,
		"Setosa", setosa,
		"Versicolor", versicolor,
	)
	if err != nil {
		panic(err)
	}
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, "iris_predictions.png"); err != nil {
		panic(err)
	}
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
