package main

import (
	"encoding/gob"
	"math/rand"
	"os"
)

func lastColumn(dataSubset []datarow) (lastColList []float32) {
	for _, row := range dataSubset {
		lastColList = append(lastColList, row[4])
	}
	return lastColList
}

// maxCount returns whichever item in the list is most frequent
func maxCount(list []float32) (highestFreqIndex float32) {
	seen := make(map[float32]float32)
	for _, variableIndex := range list {
		if _, exists := seen[variableIndex]; !exists {
			seen[variableIndex] = 0
		}
		seen[variableIndex]++
	}
	var highestSeen float32
	for variableIndex, count := range seen {
		if count > highestSeen {
			highestSeen = count
			highestFreqIndex = variableIndex
		}
	}
	return highestFreqIndex
}

func getTrainingCaseSubset(data []datarow) (subset []datarow) {
	dataLen := len(data)
	twoThirds := (dataLen * 2) / 3
	for len(subset) < twoThirds {
		subset = append(subset, data[rand.Intn(dataLen)])
	}
	return subset
}

/*
func getVariablesSubset(Variables []string) (subset []string) {
	varLen := len(Variables)
	m := int(math.Sqrt(float32(varLen)))
	for len(subset) < m {
		s := Variables[rand.Intn(varLen)]
		if !includes(subset, s) {
			subset = append(subset, s)
		}
	}
	return subset
}
*/

func includes(arr []int, compare int) (doesInclude bool) {
	for _, af := range arr {
		if af == compare {
			doesInclude = true
			break
		}
	}
	return doesInclude
}

/*
splitIntoParts is a utility for doing cross validation. it splits the dataset
into nFolds parts so they may be tested with one fold as the control
group later.
It samples with replacement from the fold.
cross_validation_split
*/
func splitIntoParts(dataset []datarow) (datasetSplit [][]datarow) {
	dataset_copy := dataset[0:]
	fold_size := len(dataset) / n_folds
	for i := 0; i < n_folds; i++ {
		var fold []datarow
		for len(fold) < fold_size {
			// take a random index from the dataset, remove it, and add it to our
			// fold list, which gets appended to the dataset_split
			// sampling without replacement
			index := rand.Intn(len(dataset_copy))
			item := dataset_copy[index]
			dataset_copy = append(dataset_copy[:index], dataset_copy[index+1:]...) // delete
			fold = append(fold, item)
		}
		datasetSplit = append(datasetSplit, fold)
	}
	return datasetSplit
}

/* Saving */

type saveFormat struct {
	Trees              []*Tree
	IndexedVariables   []string
	Variables          map[string]float32
}

// Encode via Gob to file
func save(path string, object *saveFormat) error {
	file, err := os.Create(path)
	if err == nil {
		encoder := gob.NewEncoder(file)
		encoder.Encode(object)
	}
	file.Close()
	return err
}

// Decode Gob file
func load(path string, object *saveFormat) error {
	file, err := os.Open(path)
	if err == nil {
		decoder := gob.NewDecoder(file)
		err = decoder.Decode(object)
	}
	file.Close()
	return err
}
