package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"
)

var trainingData string
var totalTrees = 5
var indexedVariables []string    // index to character
var variables map[string]float32 // character to index
var allVariableIndexes []float32 // int is the same as the index
// first 4 are considered predictors, last one is the letter index to be predicted
var trainingCases [][5]float32
var maxDepth = 10
var n_folds = 5    // how many folds of the dataset for cross-validation
var n_features int // Little `m`, will get rounded down

var charMode = true

var dataFile = "/Users/jpx/apollo.txt"

//var dataFile = "../iris.csv"

func main() {
	rand.Seed(time.Now().Unix())
	buf, err := ioutil.ReadFile(dataFile)
	if err != nil {
		panic(err)
	}
	trainingData = string(buf)
	variables = make(map[string]float32)
	n_features = int(math.Sqrt(5)) // there are 5 items per case

	if charMode {
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
			nextCase := [5]float32{
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
			nextCase := [5]float32{}
			cols := strings.Split(row, ",")
			for i := 0; i < 4; i++ {
				nc, err := strconv.ParseFloat(cols[i], 32)
				if err != nil {
					fmt.Println("row=", rowIndex, "col=", i)
					panic(err)
				}
				nextCase[i] = float32(nc)
			}
			prediction := cols[4]
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

	for _, n_trees := range []int{1, 5, 10, 25} {
		scores := evaluateAlgorithm()
		fmt.Println("Trees:", n_trees)
		fmt.Println("  Scores:", scores)
		fmt.Println("  Mean Accuracy:", sum(scores)/float32(len(scores)), "%")
	}
}

func evaluateAlgorithm() (scores []float32) {
	folds := splitIntoParts(trainingCases)
	for foldIx, testSet := range folds {
		// train on all except the fold `testSet`
		var trainSet [][5]float32
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
func (t *Tree) predict(row [5]float32) (prediction float32) {
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
func baggingPredict(trees []*Tree, row [5]float32) (mostFreqVariable float32) {
	var predictions []float32
	for _, tree := range trees {
		predictions = append(predictions, tree.predict(row))
	}
	mostFreqVariable = maxCount(predictions)
	return mostFreqVariable
}

// unclear why training set is unused, but that matches the python implementation, and
// it increases accuracy hugely
func randomForest(trainSet [][5]float32, testSet [][5]float32) (predictions []float32) {
	var allTrees []*Tree
	var wg sync.WaitGroup
	var mux sync.Mutex
	wg.Add(totalTrees)

	for i := 0; i < totalTrees; i++ {
		//sample := getTrainingCaseSubset(trainSet)
		go (func() {
			tree := getSplit(trainingCases)
			tree.split(1)
			mux.Lock()
			allTrees = append(allTrees, tree)
			mux.Unlock()
			wg.Done()
		})()
	}
	wg.Wait()
	wg.Add(len(testSet))
	for _, row := range testSet {
		go (func(r [5]float32) {
			pred := baggingPredict(allTrees, r)
			mux.Lock()
			predictions = append(predictions, pred)
			mux.Unlock()
			wg.Done()
		})(row)
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

	leftSamples  [][5]float32 // temp test cases for left group
	rightSamples [][5]float32 // temp test cases for right group
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
func getSplit(dataSubset [][5]float32) (t *Tree) {
	var bestVariableIndex float32
	var bestValueIndex float32
	var bestLeft [][5]float32
	var bestRight [][5]float32
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
func splitOnIndex(index int, value float32, dataSubset [][5]float32) (left, right [][5]float32) {
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
func calcGiniOnSplit(left, right [][5]float32, classValues []float32) (gini float32) {
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

func withValue(splitGroup [][5]float32, value float32) (count float32) {
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
func toTerminal(dataSubset [][5]float32) (highestFreqVariableIndex float32) {
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

func lastColumn(dataSubset [][5]float32) (lastColList []float32) {
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

func getTrainingCaseSubset(data [][5]float32) (subset [][5]float32) {
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
func splitIntoParts(dataset [][5]float32) (datasetSplit [][][5]float32) {
	dataset_copy := dataset[0:]
	fold_size := len(dataset) / n_folds
	for i := 0; i < n_folds; i++ {
		var fold [][5]float32
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
