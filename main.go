package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Pine - a random forest implementation that takes characters and predicts characters.

/*
Predrag Radenkovic, University of Belgrade:

Random forests is an ensemble classifier that consists of
many decision trees and outputs the class that is the mode
of the class's output by individual trees.

Decision trees are individual learners that are combined.



Why random forests? For many data sets, it produces a highly
accurate classifier. It runs efficiently on large databases.
It can handle thousands of input variables without variable
deletion. It gives estimates of what variables are important
in the classification. It generates an internal unbiased
estimate of the generalization error as the forest building
progresses.
*/

/*
Notes:
- this will implement a classifier, not regression.
- from a string we will predict the next string.
*/

var trainingData = "Hey there my name is Jeff. What is up? How are you. Hi there dude."
var trainingCases []string

// N is the number of training cases
var N int

// n is now many times to choose a training set (?)
var n int

// M is the total number of variables in the classifier
var M int

// m is the number of input variables (i.e. features, i.e. predictors)
// to be used to determine the decision at a node of the tree; m should
// be much less than M
var m int

var variables map[string]float32        // where float32 is an index
var indexedVariables map[float32]string // where string is a variable value

var totalTrees int = 1000

var nodeSize = 5

//var maxDepth = 10

func main() {
	variables = make(map[string]float32)
	allChars := strings.Split(trainingData, "")
	var c string
	for i := 0; i < len(allChars); i++ {
		c = allChars[i]
		if _, existsYet := variables[c]; !existsYet {
			newIndex := float32(len(indexedVariables) - 1)
			indexedVariables[newIndex] = c
			variables[c] = newIndex
		}
	}
	trainingCases = strings.Split(trainingData, ". ")

	// How to select M?
	// Try to recommend defaults, half of them and twice of them and pick the best
	M = len(variables)
	m = int(math.Sqrt(float64(M)))
	// How to select N?
	// Build trees until the error no longer decreases
	N = len(trainingCases)
	n = 2 * (N / 3) // .66 n

	fmt.Println("M=", M, ", m=", m, "N=", N, ", n=", n)

	rootTree := &Tree{
		Nodes: make([]*Tree, 0),
	}

	// Training

	for { // one round of training
		for t := 0; t < len(rootTree.Nodes); t++ {
			tree := rootTree.Nodes[t]
			tree.Run()
		}
	}
}

/*
Tree either has child nodes with list of child Nodes, or is a terminal with
a Value that is an index into the indexedVariables.
*/
type Tree struct {
	// Index is the index of this tree on the parent's Nodes
	Index int
	// Value is the feature; this Node will predict this Value. To look up the string
	// value, use indexedValues[tree.Value]
	Value float32
	Left  *Tree
	Right *Tree
	// Variables are all the features (predictors) this tree tries to predict.
	// Each tree will be responsible for a random subset of it's parent's
	// Variables. The string is the category (letter) and the float value
	// is the cutoff point.
	Variables map[string]float32
}

func NewTree(childIndex int, parentVariables map[string]float32) (t *Tree) {
	t = &Tree{
		Index:     childIndex,
		Value:     variables[randMapKey(parentVariables)], // the float index, not the string
		Variables: make(map[string]float32),               // all except Value
	}

	for s, ix := range parentVariables {
		if ix == t.Value {
			continue
		}
		t.Variables[s] = ix
	}

	return t
}

// allCases may be every training sample, or a subset when it is further down the tree
func Run(allCases []string) (t *Tree) {
	// Choose a training set for this tree by choosing `n` times
	// with replacement from all N available training cases (i.e.
	// take a bootstrap sample). Use the rest of the cases to
	// estimate the error of the tree, by predicting their classes.
	trainingSamples, testSamples := getRandomSubset(allCases)
	var error float32
	var bestSplitValue float32
	var bestSplitTreeIndex int
	var stopConditionHolds bool
	var features []float32 // subset of features by index
	var giniIndex float32

	// For each node of the tree, randomly choose `m` variables on
	// which to base the decision at that node.
	for len(features) < m {
		f := variables[randMapKey(variables)]
		if !includes(features, f) {
			features = append(features, f)
		}
	}

	// Calculate the best split based on these m variables in the
	// training set.
	// Splits are chosen according to a purity measure:
	// E.g. squared error (regression), Gini index or deviance (classification)

	for _, featureIndex := range features { // s is the feature string value
		for _, sample := range trainingSamples {
			left, right := testSplit(featureIndex, indexedVariables[featureIndex], trainingSamples)
			giniIndex = CalcGini(left, right, features)
		}
	}

	for n, node := range t.Nodes {

		// Estimating the importance of each predictor:
		// - Denote by ê the OOB estimate of the loss when using original training
		// set, D.
		// - For each predictor xp where p∈{1,..,k}
		// 		- Randomly permute pth predictor to generate a new set of
		// 		samples D' ={(y1,x'1),...,(yN,x'N)}
		// 		- Compute OOB estimate êk of prediction error with the new samples
		// - A measure of importance of predictor xp is êk – ê, the increase in
		// error due to random perturbation of pth predictor
		isBetter := giniIndex < bestSplitValue // gini is error
		if isBetter {
			bestSplitValue = giniIndex
			bestSplitTreeIndex = n
		}
	}

	return t
}

func includes(a []float32, f float32) (doesInclude bool) {
	for _, af := range a {
		if af == f {
			doesInclude = true
			break
		}
	}
	return doesInclude
}

// testSplit is used to split the dataset by a candidate split point
func testSplit(featureIndex int, value string, dataset []string) (left []string, right []string) {
	for _, row := range dataset {
		if string(row[index]) < value {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}
	return left, right
}

/*
CalcGini calculates the Gini impurity (aka gini index). An
alternative definition is the "expected error rate of the
system."

Wikipedia:

	Gini impurity is a measure of how often a randomly chosen
	element from the set would be incorrectly labeled if it
	was randomly labeled according to the distribution of
	labels in the subset. Gini impurity can be computed by
	summing the probability p_{i} of an item with label {i}
	being chosen times the probability 1-p_{i} of a mistake in
	categorizing that item. It reaches its minimum (zero) when
	all cases in the node fall into a single target category.
*/
func CalcGini(left, right []float32, subsetVariables []float32) (v float32) {
	var probabilities []float32

	var p float32
	for value, floatIndex := range variables {
		if len(left) != 0 {

		}
		if len(right) != 0 {

		}

		p = multiplyAll()
		probabilities = append(probabilities, p)
	}

	v = sumAll(probabilities)
	return v
}

func multiplyAll(vals []float32) (prod float32) {
	for i := 1; i < len(vals); i++ {
		prod = vals[i] * vals[i-1]
	}
	return prod
}

func sumAll(vals []float32) (s float32) {
	for i := 0; i < len(vals); i++ {
		s += vals[i]
	}
	return s
}

/*
Split will take a tree and make nodes underneath it.
*/
func (t *Tree) Split() {
	if len(t.Nodes) > 0 {
		fmt.Println("tree=", *t)
		panic("Cannot split tree that already has child nodes")
	}

}

/*
Grow is probably only done on the root tree (?)


- While growing forest, estimate test error from training samples.
- For each tree grown, 33-36% of samples are not selected in bootstrap,
called out of bootstrap (OOB) samples.
- Using OOB samples as input to the corresponding tree, predictions are
made as if they were novel test samples
- Through book-keeping, majority vote (classification), average
(regression) is computed for all OOB samples from all trees.
*/
func (t *Tree) Grow() {

}

func (t *Tree) Predict(parent *Tree) *Tree {

}

func getRandomSubset(cases []string) (trainingSamples []string, testSamples []string) {
	C := len(cases)
	indexesSeen := make([]int, C) // 0 means not seen, 1 means seen
	var i int
	var ix int

	// Get random samples, with replacement, basically it is ok to have dupes.
	// Little `n` is calculated above as a portion of big `N`
	for i = 0; i < n; i++ {
		ix = rand.Intn(C - 1)
		trainingSamples = append(trainingSamples, cases[ix])
		indexesSeen[ix] = 1
	}

	// get which samples were never put into the test
	for i = 0; i < C; i++ {
		if indexesSeen[i] == 0 {
			testSamples = append(testSamples, cases[i])
		}
	}

	return trainingSamples, testSamples
}

func randMapKey(m map[string]float32) (s string) {
	cursor := 0
	iterateUntil := rand.Intn(len(m) - 1)
	for key := range m {
		if cursor >= iterateUntil {
			s = key
			break
		}
		cursor++
	}

	return s
}
