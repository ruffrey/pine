package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"strconv"
	"strings"
)

// Load a CSV File - last column is a string value
func load_csv(filename string) (dataset []DataRow) {
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
			if err != nil { panic(err) }
			parsedRow[c] = float32(f64)
		}
		d := DataRow{Coords: parsedRow, Val: cols[len(cols)-1]}
		dataset = append(dataset, d)
	}
	return dataset
}

// Split a dataset into k folds
func cross_validation_split(dataset []DataRow, n_folds int) (dataset_split []DataRow) {
	dataset_copy := dataset[0:]
	fold_size := len(dataset) / n_folds
	for i := 0; i < n_folds; i++ {
		var fold []DataRow
		for {
			// take a random index from the dataset, remove it, and add it to our
			// fold list, which gets appended to the dataset_split
			index := rand.Intn(len(dataset_copy))
			item := dataset_copy[index]
			dataset_copy = append(dataset_copy[:index], dataset_copy[index+1:]...) // delete
			fold = append(fold, item)
			if len(fold) < fold_size {
				break
			}
		}
		dataset_split = append(dataset_split, fold...)
	}
	return dataset_split
}

// Calculate accuracy percentage
func accuracy_metric(actual string, predicted string) float32 {
	var correct float32 = 0.0
	lenActual := len(actual)
	for i := 0; i < lenActual; i++ {
		if actual[i] == predicted[i] {
			correct++
		}
	}
	return 100 * correct / float32(lenActual)
}

//Evaluate an algorithm using a cross validation split
func evaluate_algorithm(dataset []DataRow, n_folds int, args ...interface{}) {
	folds := cross_validation_split(dataset, n_folds)
	var scores []DataRow
	for ix, fold := range folds {
		train_set := append(folds[:ix], folds[ix+1:]...) // all but this fold
		train_set = sum(train_set) // summed each value so there is one item per DataRow
		var test_set []DataRow
		for _, row := range fold.Coords {
			row_copy := row
			test_set = append(test_set, row_copy)
		}
		predicted := random_forest(train_set, test_set)
		actual := [row[-1] for row in fold] // last value of last rows in fold
		lastFoldCols := make([]string)
		for _, row := range fold.Coords {
			lastFoldCols = append(row.)
		}
		actual := lastFoldRow[len(lastFoldRow)-1]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	}
}

func random_forest(dataset []DataRow, test_set []DataRow) (prediction []string) {

	return prediction
}

// metaparams
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
n_trees = 10

func main() {
	data := load_csv("sonar.all-data.csv")
	fmt.Println(data[0])
}
