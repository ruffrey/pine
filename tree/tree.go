package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

/*
Notes:
- Input data should be csv, where the last column is the thing being predicted.
- The other input rows should each be a value for a feature, compabible with being
parsed into float32.
- Every input show should have the same number of columns.
- Throughout the codebase, indexes are float32 instead of int when stored. This is
to be able to store the last column as an index to the variable string it represents.
*/

var treeSizesToTest = []int{5, 10, 100}
var trainingData string
var totalTrees = 5
var indexedVariables []string    // index to character
var variables map[string]float32 // character to index
var allVariableIndexes []float32 // int is the same as the index
// first 4 are considered predictors, last one is the letter index to be predicted
var trainingCases []datarow
var maxDepth = 10
var n_folds = 5       // how many folds of the dataset for cross-validation
var n_features int    // Little `m`, will get rounded down
var columnsPerRow int // how many total columns in a row. must be the same.

var charMode = false

//var dataFile = "/Users/jpx/apollo-short.txt"

var dataFile = "../iris.csv"

func main() {
	if os.Args[1] != "" {
		dataFile = os.Args[1]
	}
	rand.Seed(time.Now().Unix())
	fmt.Println("Reading data file", dataFile)
	buf, err := ioutil.ReadFile(dataFile)
	if err != nil {
		panic(err)
	}
	trainingData = string(buf)
	variables = make(map[string]float32)

	if charMode {
		columnsPerRow = 5
		allChars := strings.Split(trainingData, "")
		var c string
		for i := 0; i < len(allChars); i++ {
			c = allChars[i]
			if _, existsYet := variables[c]; !existsYet {
				indexedVariables = append(indexedVariables, c)
				newIndex := len(indexedVariables) - 1
				allVariableIndexes = append(allVariableIndexes, float32(newIndex))
				variables[c] = float32(newIndex)
			}
		}

		for i := 0; i < len(allChars)-4; i++ {
			nextCase := datarow{
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
		col1 := strings.Split(rows[0], ",")
		columnsPerRow = len(col1) // globally set
		for rowIndex, row := range rows {
			cols := strings.Split(row, ",")
			if len(cols) == 0 { // blank lines ignored
				continue
			}
			nextCase := make(datarow, columnsPerRow)
			// last column is the thing the previous column features predict
			for i := 0; i < columnsPerRow-1; i++ {
				nc, err := strconv.ParseFloat(cols[i], 32)
				if err != nil {
					fmt.Println("row=", rowIndex, "col=", i)
					panic(err)
				}
				nextCase[i] = float32(nc)
			}
			prediction := cols[columnsPerRow-1]
			if _, existsYet := variables[prediction]; !existsYet {
				indexedVariables = append(indexedVariables, prediction)
				newIndex := len(indexedVariables) - 1
				allVariableIndexes = append(allVariableIndexes, float32(newIndex))
				variables[prediction] = float32(newIndex)
			}
			nextCase[4] = variables[prediction]
			trainingCases = append(trainingCases, nextCase)
		}
	}

	// there are columnsPerRow items per data row
	n_features = int(math.Sqrt(float64(columnsPerRow)))

	fmt.Println("features:", columnsPerRow-1)
	fmt.Println("prediction categories:", variables)
	fmt.Println("feature split size (m):", n_features)

	// run the training testing various numbers of trees to see how many we need
	for _, n_trees := range treeSizesToTest {
		scores := evaluateAlgorithm()
		fmt.Println("\nTrees:", n_trees)
		fmt.Println("  Scores:", scores)
		fmt.Println("  Mean Accuracy:", sum(scores)/float32(len(scores)), "%")
	}
}

type datarow []float32

func evaluateAlgorithm() (scores []float32) {
	folds := splitIntoParts(trainingCases)
	for foldIx, testSet := range folds {
		// train on all except the fold `testSet`
		var trainSet []datarow
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
func (t *Tree) predict(row datarow) (prediction float32) {
	if row[int(t.VariableIndex)] < t.ValueIndex {
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
func baggingPredict(trees []*Tree, row datarow) (mostFreqVariable float32) {
	var predictions []float32
	for _, tree := range trees {
		predictions = append(predictions, tree.predict(row))
	}
	mostFreqVariable = maxCount(predictions)
	return mostFreqVariable
}

// unclear why training set is unused, but that matches the python implementation, and
// it increases accuracy hugely
func randomForest(trainSet []datarow, testSet []datarow) (predictions []float32) {
	var allTrees []*Tree

	for i := 0; i < totalTrees; i++ {
		//sample := getTrainingCaseSubset(trainSet)
		tree := getSplit(trainingCases)
		tree.split(1)
		allTrees = append(allTrees, tree)
	}
	for _, row := range testSet {
		pred := baggingPredict(allTrees, row)
		predictions = append(predictions, pred)
	}
	return predictions
}

var f_100 float32 = 100

func accuracyMetric(actual []float32, predicted []float32) (accuracy float32) {
	var correct float32
	lenActual := len(actual)
	for i := 0; i < lenActual; i++ {
		if actual[i] == predicted[i] {
			correct += 1
		}
	}
	accuracy = f_100 * correct / float32(lenActual)
	return accuracy
}

func sum(scores []float32) (s float32) {
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
	VariableIndex float32 // the variable that this tree splits on (?) (Index)
	ValueIndex    float32 // the split value of this node
	LeftNode      *Tree
	RightNode     *Tree
	LeftTerminal  float32 // index of a variable that this predicts
	RightTerminal float32 // index of a variable that this predicts

	leftSamples  []datarow // temp test cases for left group
	rightSamples []datarow // temp test cases for right group
}

func (t *Tree) String() string {
	return fmt.Sprintf("VariableIndex: %f, ValueIndex: %f, LeftNode: %+v, RightNode: %+v, LeftTerminal: %f, RightTerminal: %f",
		t.VariableIndex,
		t.ValueIndex,
		t.LeftNode,
		t.RightNode,
		t.LeftTerminal,
		t.RightTerminal)
}

// getSplit selects the best split point for a dataset
func getSplit(dataSubset []datarow) (t *Tree) {
	var bestVariableIndex float32
	var bestValueIndex float32
	var bestLeft []datarow
	var bestRight []datarow
	var bestGini float32 = 9999

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
			left, right := splitOnIndex(varIndex, row[int(varIndex)], dataSubset)
			// last column is the features
			gini := calcGiniOnSplit(left, right, lastColumn(dataSubset))
			if gini < bestGini { // lowest gini is lowest error in predicting
				bestVariableIndex = float32(varIndex)
				bestValueIndex = row[int(varIndex)]
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
func splitOnIndex(index int, value float32, dataSubset []datarow) (left, right []datarow) {
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
func calcGiniOnSplit(left, right []datarow, classValues []float32) (gini float32) {
	// how many of the items in the split predict the class value?
	for _, classVariableIndex := range classValues {
		var size float32
		var proportion float32
		// left
		if len(left) != 0 {
			size = float32(len(left))
			proportion = withValue(left, classVariableIndex) / size
			gini += proportion * (1 - proportion)
		}
		// right is the same code
		if len(right) != 0 {
			size = float32(len(right))
			proportion = withValue(right, classVariableIndex) / size
			gini += proportion * (1 - proportion)
		}
	}
	return gini
}

func withValue(splitGroup []datarow, value float32) (count float32) {
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
func (t *Tree) split(depth int) {
	//defer (func() { t.leftSamples = nil; t.rightSamples = nil })()

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
		t.LeftNode.split(depth + 1)
	}

	// process right
	if len(t.rightSamples) <= 1 { // only one row left (?)
		t.RightTerminal = toTerminal(t.rightSamples)
	} else {
		t.RightNode = getSplit(t.rightSamples)
		t.RightNode.split(depth + 1)
	}
}

// whatever is most represented
func toTerminal(dataSubset []datarow) (highestFreqVariableIndex float32) {
	outcomes := make(map[float32]int)
	for _, row := range dataSubset {
		if _, exists := outcomes[row[4]]; !exists {
			outcomes[row[4]] = 1
		} else {
			outcomes[row[4]]++
		}
	}
	var highestFreq int
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
func getVariablesSubset(variables []string) (subset []string) {
	varLen := len(variables)
	m := int(math.Sqrt(float32(varLen)))
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
