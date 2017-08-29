package main

import (
	"log"
	"math/rand"
	"sync"
	"time"
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
			log.Println("(", foldIx, ") Fold start")
			predicted, treeSet := randomForest(foldIx, trainSet, testSet)
			log.Println("(", foldIx, ") Fold done")
			treeLock.Lock()
			trees = append(trees, treeSet...)
			treeLock.Unlock()
			actual := lastColumn(testSet)
			accuracy := accuracyMetric(actual, predicted)
			scoreLock.Lock()
			scores = append(scores, accuracy)
			scoreLock.Unlock()
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

func treeWorker(jobs <-chan []datarow, results chan<- *Tree) {
	for trainSet := range jobs {
		sample := getTrainingCaseSubset(trainSet)
		tree := getSplit(sample)
		tree.split(1)
		results <- tree
	}
}

// Originally in the python implementation, the training subset was unused. That
// seems incorrect. It decreases accuracy to only be working with the 2/3 of the training
// subset (which was already n_folds-1/n_folds). Decreased accuracy on a single node
// might be better than high accuracy per node, because the nodes should be dissimilar
// but together they vote for the best answer.
func randomForest(foldIndex int, trainSet []datarow, testSet []datarow) (predictions []float32, allTrees []*Tree) {
	jobs := make(chan []datarow, parallelTrees)
	results := make(chan *Tree, *treesPerFold)

	// spawn worker pool
	for i := 0; i < parallelTrees; i++ {
		go treeWorker(jobs, results)
	}
	// send all jobs into the pool
	for i := 0; i < *treesPerFold; i++ {
		jobs <- trainSet
	}
	close(jobs) // disallow any more jobs to enter

	var lenAll int
	for tree := range results {
		allTrees = append(allTrees, tree)
		lenAll = len(allTrees)
		log.Println("(", foldIndex, ") Tree done", lenAll, "/", *treesPerFold)
		if lenAll >= *treesPerFold {
			close(results)
			break
		}
	}

	// worker pool done

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

var _logEvery int64 = 2000
var _sec = int64(time.Second)

// getSplit selects the best split point for a dataset, for a few features only,
// so this tree cares about only some features, not all of them.
func getSplit(dataSubset []datarow) (t *Tree) {
	var bestVariableIndex float32
	var bestValueIndex float32
	var bestLeft []datarow
	var bestRight []datarow
	var bestGini float32 = 9999

	// prevent many malloc and gc events by reusing these
	sc := splitCache{}

	// choose the features
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
	var index int64 = 0
	var n int64
	var perSplit int64
	var remaining int64
	lastLog := time.Now().UnixNano()
	totalIterations := int64(n_features * len(dataSubset))
	log.Println("totalIterations:", n_features, "*", len(dataSubset), "=", totalIterations)
	for _, varIndex := range features {
		for _, row := range dataSubset {
			// create a test split
			sc.splitOnIndex(varIndex, row[int(varIndex)], dataSubset)
			// last column is the features
			gini := calcGiniOnSplit(sc.leftLastCols, sc.rightLastCols, lastColumn(dataSubset))
			if gini <= bestGini { // lowest gini is lowest error in predicting
				bestVariableIndex = float32(varIndex)
				bestValueIndex = row[int(varIndex)]
				bestGini = gini
				bestLeft = sc.left
				bestRight = sc.right
			}
			index++
			if index%_logEvery == 0 {
				n = time.Now().UnixNano()
				perSplit = (n - lastLog) / _logEvery
				remaining = totalIterations - index
				log.Println(perSplit, "ns per row", (remaining*perSplit)/_sec,
					"secs left in split")
				lastLog = n
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
		prediction = splitGroupLastColumn[i]
		// this next line is really hot
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

	// NOT clear that this did any good, actually. Seems to prematurely
	// end the tree when it will normally gain more depth.
	// i.e. if it is a no-split, isn't that handled below already, on each
	// terminal?

	//if len(t.leftSamples) == 0 || len(t.rightSamples) == 0 {
	//	if len(t.leftSamples) > 0 {
	//		t.LeftTerminal = toTerminal(t.leftSamples)
	//		t.LeftTerminal = toTerminal(t.leftSamples)
	//	} else {
	//		t.RightTerminal = toTerminal(t.rightSamples)
	//		t.RightTerminal = toTerminal(t.rightSamples)
	//	}
	//	return
	//}

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
