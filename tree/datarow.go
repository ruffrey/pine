package main

import (
	"fmt"
	"strconv"
	"strings"
)

type datarow []float32

func parseRow(row string, rowIndex int) (dr datarow) {
	cols := strings.Split(row, ",")
	if len(cols) == 0 { // blank lines ignored
		return nil
	}
	dr = make(datarow, columnsPerRow)
	// last column is the thing the previous column features predict
	for i := 0; i < lastColumnIndex; i++ {
		nc, err := strconv.ParseFloat(cols[i], 32)
		if err != nil {
			fmt.Println("row=", rowIndex, "col=", i)
			panic(err)
		}
		dr[i] = float32(nc)
	}
	prediction := cols[lastColumnIndex]
	if _, existsYet := variables[prediction]; !existsYet {
		indexedVariables = append(indexedVariables, prediction)
		newIndex := len(indexedVariables) - 1
		variables[prediction] = float32(newIndex)
	}
	dr[lastColumnIndex] = variables[prediction]

	return dr
}
