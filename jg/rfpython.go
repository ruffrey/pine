package main

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
)

type Dataset struct {
	Coords []float32
	Val    string
}

// Load a CSV File
// last column is a string value
func load_csv(filename string) (dataset []*Dataset) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	rows := strings.Split(string(buf), "\n")
	for r := 0; r < len(rows); r++ {
		row := rows[r]
		cols := strings.Split(row, ",")
		parsedRow := make([]float32, len(cols)-1)
		for c := 0; c < len(cols)-1; c++ {
			f64, err := strconv.ParseFloat(cols[c], 32)
			if err != nil {
				panic(err)
			}
			parsedRow[c] = float32(f64)
		}
		d := &Dataset{Coords: parsedRow, Val: cols[len(cols)-1]}
		dataset = append(dataset, d)
	}
	return dataset
}

func main() {
	data := load_csv("sonar.all-data.csv")
	fmt.Println(data[0])
}
