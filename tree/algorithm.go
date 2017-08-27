package main

import (
	"log"
	"math/rand"
	"sync"
)

func evaluateAlgorithm() (scores []float32, trees []*Tree) {
	folds := splitIntoParts(trainingCases)
	var treeLock sync.Mutex
	var scoreLock sync.Mutex
	var wg sync.WaitGroup
	wg.Add(len(folds))
	for fIx, tst := range folds {
		go (func(foldIx int, testSet []datarow) {
			// train on all except the fold `testSet`
			var trainSet []datarow
			for i := 0; i < len(folds); i++ {
				if i != foldIx {
					trainSet = append(trainSet, folds[i]...)
				}
			}
			log.Println("Fold start:", foldIx)
			predicted, treeSet := randomForest(trainSet, testSet)
			treeLock.Lock()
			trees = append(trees, treeSet...)
			treeLock.Unlock()
			actual := lastColumn(testSet)
			accuracy := accuracyMetric(actual, predicted)
			scoreLock.Lock()
			scores = append(scores, accuracy)
			scoreLock.Unlock()
			log.Println("Fold end:", foldIx)
			wg.Done()
		})(fIx, tst)
	}
	wg.Wait()

	return scores, trees
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

// Originally in the python implementation, the training subset was unused. That
// seems incorrect. It decreases accuracy to only be working with the 2/3 of the training
// subset (which was already n_folds-1/n_folds). Decreased accuracy on a single node
// might be better than high accuracy.
func randomForest(trainSet []datarow, testSet []datarow) (predictions []float32, allTrees []*Tree) {
	for i := 0; i < *treesPerFold; i++ {
		sample := getTrainingCaseSubset(trainSet)
		tree := getSplit(sample)
		tree.split(1)
		allTrees = append(allTrees, tree)
		log.Println("split tree", i, "/", *treesPerFold)
	}
	for _, row := range testSet {
		pred := baggingPredict(allTrees, row)
		predictions = append(predictions, pred)
	}
	return predictions, allTrees
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

// getSplit selects the best split point for a dataset, for a few features only,
// so this tree cares about only some features, not all of them.
func getSplit(dataSubset []datarow) (t *Tree) {
	var bestVariableIndex float32
	var bestValueIndex float32
	var bestLeft []datarow
	var bestRight []datarow
	var bestGini float32 = 9999

	// prevent many malloc events
	var leftLastCols []float32
	var rightLastCols []float32

	var features []int32 // index of
	for len(features) < n_features {
		// the following line is quite slow
		index := rand.Int31n(int32(columnsPerRow - 1)) // total cases per input
		if !includes(features, index) {
			features = append(features, index)
		}
	}

	// The goal is split the subsets of data on random Variables,
	// and see which one best predicts the row of data. That gets turned into a
	// new tree
	for _, varIndex := range features {
		for _, row := range dataSubset {
			// create a test split
			left, right := splitOnIndex(varIndex, row[int(varIndex)], dataSubset, leftLastCols, rightLastCols)
			// last column is the features
			gini := calcGiniOnSplit(leftLastCols, rightLastCols, lastColumn(dataSubset))
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

We also extract the last columns for the left and right splits, because they will
come in handy to speed up the really hot path in `withValue`.

test_split
*/
func splitOnIndex(index int32, value float32, dataSubset []datarow, leftLastCols []float32, rightLastCols []float32) (left []datarow, right []datarow) {
	// clear them, but keep garbage collection from cleaning them up,
	// without this it is actually slower to just get rid of them entirely
	// just overwrite leftLastCols and rightLastCols values where needed
	//leftLastCols = leftLastCols[:cap(leftLastCols)]
	//rightLastCols = rightLastCols[:cap(rightLastCols)]

	for _, row := range dataSubset {
		// last column has same index as the original row
		if row[index] < value {
			left = append(left, row)
			leftLastCols = append(leftLastCols, row[lastColumnIndex])
		} else {
			right = append(right, row)
			rightLastCols = append(rightLastCols, row[lastColumnIndex])
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
func calcGiniOnSplit(leftLastCols []float32, rightLastCols []float32, classValues []float32) (gini float32) {
	// how many of the items in the split predict the class value?
	for _, classVariableIndex := range classValues {
		var size float32
		var proportion float32
		// left
		if len(leftLastCols) != 0 {
			size = float32(len(leftLastCols))
			proportion = withValue(classVariableIndex, leftLastCols) / size
			gini += proportion * (1 - proportion)
		}
		// right is the same code
		if len(rightLastCols) != 0 {
			size = float32(len(rightLastCols))
			proportion = withValue(classVariableIndex, rightLastCols) / size
			gini += proportion * (1 - proportion)
		}
	}
	return gini
}

// this function takes up about 91 - 98% of cpu burn.
func withValue(value float32, splitGroupLastColumn []float32) (count float32) {
	splitGroupLen := len(splitGroupLastColumn)
	var prediction float32
	for i := 0; i < splitGroupLen; i++ {
		// this next line is really hot
		prediction = splitGroupLastColumn[i]
		if prediction == value {
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
