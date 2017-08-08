package main

import (
	"math"
	"math/rand"
	"strings"
)

var trainingData = "Hey there my name is Jeff. What is up? How are you. Hi there dude."
var totalTrees = 5
var trainingRounds = 10
var M_bags = 5
var indexedVariables []string // index to character
var variables map[string]int  // character to index
// first 4 are considered predictors, last one is the letter index to be predicted
var trainingCases [][5]int
var maxDepth = 5

func main() {
	variables = make(map[string]int)
	allChars := strings.Split(trainingData, "")
	var c string
	for i := 0; i < len(allChars); i++ {
		c = allChars[i]
		if _, existsYet := variables[c]; !existsYet {
			indexedVariables = append(indexedVariables, c)
			newIndex := len(indexedVariables) - 1
			variables[c] = newIndex
		}
	}
	for i := 0; i < len(allChars)-4; i++ {
		nextCase := [5]int{
			variables[allChars[i]],
			variables[allChars[i+1]],
			variables[allChars[i+2]],
			variables[allChars[i+3]],
			variables[allChars[i+4]],
		}
		trainingCases = append(trainingCases, nextCase)
	}

	train()
}

/*
Tree, aka a node, has a left and a right branch.

When evaluating for an input row, take the input row and get the value
at the VariableIndex in the input row. If it is less than the ValueIndex,
go left (which might terminate). Otherwise, go right (which also might
terminate).
- if leftSamples or rightSamples, we still need to predict down the chain
- if
*/
type Tree struct {
	VariableIndex int
	ValueIndex    int // the value that this tree predicts (?)
	Left          *Tree
	leftSamples   [][5]int // test cases for left group
	Right         *Tree
	rightSamples  [][5]int // test cases for right group
}

/*
train

1. sample N cases at random with replacement to create a subset of the data (see top layer of figure above). The subset should be about 66% of the total set.
*/
func train() {
	var trees []*Tree

	// initial trees
	for i := 0; i < totalTrees; i++ {
		// create a tree
		t := &Tree{}
		trees = append(trees, t)
	}

	for i := 0; i < trainingRounds; i++ {
		var y_OutputValues []int // list of variable indexes predicted by each tree
		for _, tree := range trees {
			dataSubsetBag := getTrainingCaseSubset(trainingCases)

			variablesSubset := getVariablesSubset(indexedVariables)

			tree.split(dataSubsetBag, variablesSubset)
		}
	}
}

// getSplit selects the best split point for a dataset
func getSplit(dataSubset [][5]int, variablesSubset []int) (t *Tree) {
	var bestVariableIndex int
	var bestValueIndex int
	var bestLeft [][5]int
	var bestRight [][5]int
	var bestGini float64 = 9999

	// The goal here seems to be to split the subsets of data on random variables,
	// and see which one best predicts the row of data. That gets turned into a
	// new tree
	for _, varIndex := range variablesSubset {
		for _, row := range dataSubset {
			// create a test split
			left, right := splitOnIndex(varIndex, row[varIndex], dataSubset)
			gini := calcGiniOnSplit(left, right, lastColumn(dataSubset))
			if gini < bestGini { // lowest gini is lowest error in predicting
				bestVariableIndex = varIndex
				bestValueIndex = row[varIndex]
				bestGini = gini
				bestLeft = left
				bestRight = right
			}
		}
	}
	t = &Tree{
		VariableIndex: bestVariableIndex,
		ValueIndex:    bestValueIndex,
		leftSamples:   bestLeft,
		rightSamples:  bestRight,
	}
	return t
}

func lastColumn(dataSubset [][5]int) (lastColList []int) {
	for _, row := range dataSubset {
		lastColList = append(lastColList, row[4])
	}
	return lastColList
}

/*
splitOnIndex splits a dataset based on an attribute and an attribute value

test_split
*/
func splitOnIndex(index int, value int, dataSubset [][5]int) (left, right [][5]int) {
	for _, row := range dataSubset {
		if row[index] < value {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}
	return left, right
}

/*
calcGiniOnSplit calculates the error of the split dataset

classValues are the last column (to be predicted by preceeding columns)
of the training sets that were split into left and right. So we look at
the left and right split, and how well each one predicts the expected
output values.
*/
func calcGiniOnSplit(left, right [][5]int, classValues []int) (gini float64) {
	// how many of the items in the split predict the class value?
	for _, classVariableIndex := range classValues {
		var size float64
		var proportion float64
		// left
		if len(left) != 0 {
			size = float64(len(left))
			proportion = withValue(left, classVariableIndex) / size
			gini += proportion * (1 - proportion)
		}
		// right is the same code
		if len(right) != 0 {
			size = float64(len(right))
			proportion = withValue(right, classVariableIndex) / size
			gini += proportion * (1 - proportion)
		}
	}
	return gini
}

func withValue(splitGroup [][5]int, value int) (count float64) {
	for _, varIndex := range splitGroup {
		if varIndex[4] == value {
			count++
		}
	}
	return count
}

/*
split creates child splits for a node or makes terminals. This gives
structure to the new tree created by getSplit()
*/
func (t *Tree) split(dataSubset [][5]int, variablesSubset []int, depth int) {
	// check for a no-split
	// a perfect split in one direction, so make a terminal out of it.
	// toTerminal will pick the most frequent variable index
	if len(t.leftSamples) == 0 || len(t.rightSamples) == 0 {
		if len(t.leftSamples) > 0 {
			t.Left = toTerminal(t.leftSamples)
			t.Right = toTerminal(t.leftSamples)
		} else {
			t.Left = toTerminal(t.rightSamples)
			t.Right = toTerminal(t.rightSamples)
		}
		return
	}
	// check for max depth
	// too deep - go ahead and do like we did above, let the toTerminal
	// function choose the most frequent variable index to be the value on
	// each side. the split index will determine which way to go when an
	// input row comes in
	if depth >= maxDepth {
		t.Left = toTerminal(t.leftSamples)
		t.Right = toTerminal(t.rightSamples)
		return
	}

	// process left
	if len(t.leftSamples) <= 1 { // only one row left (?)
		t.Left = toTerminal(t.leftSamples)
	} else {
		t.Left = getSplit(t.leftSamples, variablesSubset)
		t.Left.split(t.leftSamples, variablesSubset, depth+1)
	}

	// process right
	if len(t.leftSamples) <= 1 { // only one row left (?)
		t.Right = toTerminal(t.rightSamples)
	} else {
		t.Right = getSplit(t.rightSamples, variablesSubset)
		t.Right.split(t.rightSamples, variablesSubset, depth+1)
	}
}

// whatever is most represented
func toTerminal(dataSubset [][5]int) (t *Tree) {
	outcomes := make(map[int]int)
	for _, row := range dataSubset {
		if _, exists := outcomes[row[4]]; !exists {
			outcomes[row[4]] = 1
		} else {
			outcomes[row[4]]++
		}
	}
	var highestFreq int
	var highestFreqVariableIndex int
	for varIndex, count := range outcomes {
		if count > highestFreq {
			highestFreq = count
			highestFreqVariableIndex = varIndex
		}
	}
	return &Tree{
		ValueIndex: highestFreqVariableIndex,
	}
}

// predict takes a list of variable indexes and predicts a single
// variable index as the output
func (t *Tree) predict(rowOfVarIndexes [5]int) (predictionIndex int) {

}

//
// utilities
//

func getTrainingCaseSubset(data [][5]int) (subset [][5]int) {
	dataLen := len(data)
	twoThirds := (dataLen * 2) / 3
	for len(subset) < twoThirds {
		subset = append(subset, data[rand.Intn(dataLen)])
	}
	return subset
}

func getVariablesSubset(variables []string) (subset []string) {
	varLen := len(variables)
	m := int(math.Sqrt(float64(varLen)))
	for len(subset) < m {
		s := variables[rand.Intn(varLen)]
		if !includes(subset, s) {
			subset = append(subset, s)
		}
	}
	return subset
}

func includes(a []string, f string) (doesInclude bool) {
	for _, af := range a {
		if af == f {
			doesInclude = true
			break
		}
	}
	return doesInclude
}

// splitIntoParts is a utility for doing cross validation. it splits the dataset
// into nFolds parts so they may be tested with one fold as the control
// group later (?)
// cross_validation_split
func splitIntoParts(dataset [][5]int, nFolds int) (datasetSplit [][][5]int) {
	foldSize := nFolds / len(dataset)
	for i := 0; i < nFolds; i++ {
		datasetSplit = append(datasetSplit, dataset[nFolds*i:nFolds*i+1])
	}
	return datasetSplit
}
