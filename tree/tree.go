package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

var trainingData string
var totalTrees = 5
var indexedVariables []string // index to character
var variables map[string]int  // character to index
var allVariableIndexes []int  // int is the same as the index
// first 4 are considered predictors, last one is the letter index to be predicted
var trainingCases [][5]int
var maxDepth = 5
var n_folds = 5    // how many folds of the dataset for cross-validation
var n_features int // Little `m`, will get rounded down

var charMode = false

//var dataFile = "/Users/jpx/apollo.txt"
var dataFile = "../jg/iris.csv"

func main() {
	rand.Seed(time.Now().UnixNano())
	buf, err := ioutil.ReadFile(dataFile)
	if err != nil {
		panic(err)
	}
	trainingData = string(buf)
	variables = make(map[string]int)

	if charMode {
		allChars := strings.Split(trainingData, "")
		var c string
		for i := 0; i < len(allChars); i++ {
			c = allChars[i]
			if _, existsYet := variables[c]; !existsYet {
				indexedVariables = append(indexedVariables, c)
				newIndex := len(indexedVariables) - 1
				allVariableIndexes = append(allVariableIndexes, newIndex)
				variables[c] = newIndex
			}
		}

		n_features = int(math.Sqrt(5)) // there are 5 items per case

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

	} else { // NOT character prediction mode
		rows := strings.Split(trainingData, "\n")
		for rowIndex, row := range rows {
			nextCase := [5]int{}
			cols := strings.Split(row, ",")
			for i := 0; i < 4; i++ {
				_flt, err := strconv.ParseFloat(cols[i], 64)
				nextCase[i] = int(_flt) // we lose a lot of percision here, but later..better
				if err != nil {
					fmt.Println("row=", rowIndex, "col=", i)
					panic(err)
				}
			}
			prediction := cols[4]
			if _, existsYet := variables[prediction]; !existsYet {
				indexedVariables = append(indexedVariables, prediction)
				newIndex := len(indexedVariables) - 1
				allVariableIndexes = append(allVariableIndexes, newIndex)
				variables[prediction] = newIndex
			}
			nextCase[4] = variables[prediction]
			trainingCases = append(trainingCases, nextCase)
		}
	}

	for _, n_trees := range []int{1, 5, 10, 25, 100} {
		scores := evaluateAlgorithm()
		fmt.Println("Trees:", n_trees)
		fmt.Println("  Scores:", scores)
		fmt.Println("  Mean Accuracy:", sum(scores)/float64(len(scores)), "%")
	}
}

func evaluateAlgorithm() (scores []float64) {
	folds := splitIntoParts(trainingCases)
	for foldIx, testSet := range folds {
		// train on all except the fold `testSet`
		var trainSet [][5]int
		for i := 0; i < len(folds); i++ {
			if i != foldIx {
				trainSet = append(trainSet, folds[i]...)
			}
		}
		predicted := randomForest(trainSet, testSet)
		actual := lastColumn(testSet)
		accuracy := accuracyMetric(actual, predicted)
		scores = append(scores, accuracy)
	}

	return scores
}

// predict takes a list of variable indexes (an input row) and predicts a single
// variable index as the output.
func (t *Tree) predict(row [5]int) (prediction int) {
	if row[t.VariableIndex] < t.ValueIndex {
		if t.LeftNode != nil {
			return t.LeftNode.predict(row)
		}
		return t.LeftTerminal
	}
	if t.RightNode != nil {
		return t.RightNode.predict(row)
	}
	return t.RightTerminal
}

// baggingPredict returns the most frequent variable index in the list of predictions
func baggingPredict(trees []*Tree, row [5]int) (mostFreqVariable int) {
	var predictions []int
	for _, tree := range trees {
		predictions = append(predictions, tree.predict(row))
	}
	mostFreqVariable = maxCount(predictions)
	return mostFreqVariable
}

func randomForest(trainSet [][5]int, testSet [][5]int) (predictions []int) {
	var allTrees []*Tree
	for i := 0; i < totalTrees; i++ {
		sample := getTrainingCaseSubset(trainSet)
		tree := getSplit(sample)
		tree.split(sample, 1)
		allTrees = append(allTrees, tree)
	}
	for _, row := range testSet {
		predictions = append(predictions, baggingPredict(allTrees, row))
	}
	return predictions
}

func accuracyMetric(actual []int, predicted []int) (accuracy float64) {
	correct := 0.0
	lenActual := len(actual)
	for i := 0; i < lenActual; i++ {
		if actual[i] == predicted[i] {
			correct += 1
		}
	}
	accuracy = 100 * correct / float64(lenActual)
	return accuracy
}

func sum(scores []float64) (s float64) {
	for _, f := range scores {
		s += f
	}
	return s
}

/*
Tree, for left and right, either has a Node or a terminal value

When evaluating for an input row, take the input row and get the value
at the VariableIndex in the input row. If it is less than the ValueIndex,
go left (which might terminate). Otherwise, go right (which also might
terminate).
*/
type Tree struct {
	VariableIndex int // the variable that this tree splits on (?) (Index)
	ValueIndex    int // the split value of this node
	LeftNode      *Tree
	RightNode     *Tree
	LeftTerminal  int // index of a variable that this predicts
	RightTerminal int // index of a variable that this predicts

	leftSamples  [][5]int // temp test cases for left group
	rightSamples [][5]int // temp test cases for right group
}

// getSplit selects the best split point for a dataset
func getSplit(dataSubset [][5]int) (t *Tree) {
	var bestVariableIndex int
	var bestValueIndex int
	var bestLeft [][5]int
	var bestRight [][5]int
	var bestGini float64 = 9999

	var features []int // index of
	for len(features) < n_features {
		index := rand.Intn(5) // total cases per input
		if !includes(features, index) {
			features = append(features, index)
		}
	}

	// The goal here seems to be to split the subsets of data on random variables,
	// and see which one best predicts the row of data. That gets turned into a
	// new tree
	for _, varIndex := range features {
		for _, row := range dataSubset {
			// create a test split
			left, right := splitOnIndex(varIndex, row[varIndex], dataSubset)
			// last column is the features
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
split creates child splits for a t or makes terminals. This gives
structure to the new tree created by getSplit()
*/
func (t *Tree) split(dataSubset [][5]int, depth int) {
	defer (func() { t.leftSamples = nil; t.rightSamples = nil })()
	// check for a no-split
	// a perfect split in one direction, so make a terminal out of it.
	// toTerminal will pick the most frequent variable index
	if len(t.leftSamples) == 0 || len(t.rightSamples) == 0 {
		if len(t.leftSamples) > 0 {
			t.LeftTerminal = toTerminal(t.leftSamples)
			t.LeftTerminal = toTerminal(t.leftSamples)
		} else {
			t.RightTerminal = toTerminal(t.rightSamples)
			t.RightTerminal = toTerminal(t.rightSamples)
		}
		return
	}
	// check for max depth
	// too deep - go ahead and do like we did above, let the toTerminal
	// function choose the most frequent variable index to be the value on
	// each side. the split index will determine which way to go when an
	// input row comes in
	if depth >= maxDepth {
		t.LeftTerminal = toTerminal(t.leftSamples)
		t.RightTerminal = toTerminal(t.rightSamples)
		return
	}

	// process left
	if len(t.leftSamples) <= 1 { // only one row left (?)
		t.LeftTerminal = toTerminal(t.leftSamples)
	} else {
		t.LeftNode = getSplit(t.leftSamples)
		t.LeftNode.split(t.leftSamples, depth+1)
	}

	// process right
	if len(t.leftSamples) <= 1 { // only one row left (?)
		t.RightTerminal = toTerminal(t.rightSamples)
	} else {
		t.RightNode = getSplit(t.rightSamples)
		t.RightNode.split(t.rightSamples, depth+1)
	}
}

// whatever is most represented
func toTerminal(dataSubset [][5]int) (value int) {
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
	return highestFreqVariableIndex
}

//
//
// utilities
//
//

func lastColumn(dataSubset [][5]int) (lastColList []int) {
	for _, row := range dataSubset {
		lastColList = append(lastColList, row[4])
	}
	return lastColList
}

// maxCount returns whichever item in the list is most frequent
func maxCount(list []int) (highestFreqIndex int) {
	seen := make(map[int]int)
	for _, variableIndex := range list {
		if _, exists := seen[variableIndex]; !exists {
			seen[variableIndex] = 0
		}
		seen[variableIndex]++
	}
	highestSeen := 0
	for variableIndex, count := range seen {
		if count > highestSeen {
			highestSeen = count
			highestFreqIndex = variableIndex
		}
	}
	return highestFreqIndex
}

func getTrainingCaseSubset(data [][5]int) (subset [][5]int) {
	dataLen := len(data)
	twoThirds := (dataLen * 2) / 3
	for len(subset) < twoThirds {
		subset = append(subset, data[rand.Intn(dataLen)])
	}
	return subset
}

/*
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
func splitIntoParts(dataset [][5]int) (datasetSplit [][][5]int) {
	dataset_copy := dataset[0:]
	fold_size := len(dataset) / n_folds
	for i := 0; i < n_folds; i++ {
		var fold [][5]int
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
		datasetSplit = append(datasetSplit, fold)
	}
	return datasetSplit
}
