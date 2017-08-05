package main

import (
	"strings"
)

// Pine - a random forest implementation.

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

var trainingData = "Hey there my name is Jeff. What is up? How are you. Hi there dude."
var trainingCases []string

// N is the number of training cases
var N int

// n is now many times to choose a training set (?)
var n int

// M is the total number of variables in the classifier
var M int

// m is the number  of input variables to be used to determine
// the decision at a node of the tree; m should be much less than M
var m int

var variables map[string]bool

var totalTrees int = 1000

func main() {
	uniqueChars := make(map[string]bool)
	allChars := strings.Split(trainingData, "")
	var c string
	for i := 0; i < len(allChars); i++ {
		c = allChars[i]
		if !uniqueChars[c] {
			uniqueChars[c] = true
		}
	}
	trainingCases = strings.Split(trainingData, ". ")

	variables = uniqueChars

	// How to select M?
	// Try to recommend defaults, half of them and twice of them and pick the best
	M = len(uniqueChars)
	// How to select N?
	// Build trees until the error no longer decreases
	N = len(trainingCases)
	rootTree := &Tree{
		Nodes: make([]Tree, 0),
	}

	// Training

	for { // one round of training
		for t := 0; t < len(rootTree.Nodes); t++ {
			tree := rootTree.Nodes[t]
			tree.Run()
		}
	}
}

type Tree struct {
	Nodes []Tree
}

func (t *Tree) Run() {
	// Choose a training set for this tree by choosing `n` times
	// with replacement from all N available training cases (i.e.
	// take a bootstrap sample). Use the rest of the cases to
	// estimate the error of the tree, by predicting their classes.
	var trainingSample string
	var otherSamples []string
	var error float32
	var bestSplitValue float32
	var bestSplitTreeIndex int
	var stopConditionHolds bool
	var giniIndex float32

	for {
		// For each node of the tree, randomly choose `m` variables on
		// which to base the decision at that node.
		for n := 0; n < len(t.Nodes); n++ {
			// Calculate the best split based on these m variables in the
			// training set.
			// Splits are chosen according to a purity measure:
			// E.g. squared error (regression), Gini index or devinace (classification)
			giniIndex = t.Nodes[n].CalcSplit()
			isBetter := giniIndex > bestSplitValue
			if isBetter {
				bestSplitValue = giniIndex
				bestSplitTreeIndex = n
			}
		}
		if stopConditionHolds {
			break
		}
	}

}

func (t *Tree) CalcSplit() (v float32) {
	return v
}

func (t *Tree) Split() {

}
