package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strings"
	"time"

	"encoding/json"

	"os"

	"path/filepath"

	"runtime"

	"github.com/pkg/profile"
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

var trainingData string
var indexedVariables []string    // index to character
var variables map[string]float32 // character to index
// first len-1 are considered predictors, last one is the letter index to be predicted
var trainingCases []datarow

// maxDepth is the maximum depth of child nodes allowed from the root of a tree
var maxDepth = 8
var n_features int    // Little `m`, will get rounded down
var columnsPerRow int // how many total columns in a row. must be the same.
// how many inputs are fed into the network during a sample; similar to sequence length with neural networks
var lastColumnIndex int // columnsPerRow minus 1
var sequenceLength int  // for character mode
// how many trees to build at once (per fold)
var parallelTrees int = 1

// what to spilt on during -charmode. empty string will make very character
// an input, while splitting on space will use words
var charmodeSplitChar = " "

/* flags */
var treesPerFold *int
var n_folds *int // how many folds of the dataset for cross-validation
var charMode *bool
var dataFile *string
var modelFile *string
var saveTo *string
var seedText *string
var prof *string
var overrideFeatureSplitSize *int // override n_features
var overrideSequenceLength *int   // override sequenceLength

// in the dataset (minus 1 fold for cross-validation), how many samples
// should be taken from the dataset (with replacement) to train each tree?
// as a percent of dataset
var subsetSizePercent *float64

// skipSize is how many letters to skip during character mode when making
// test cases. A size of 1 means each and every inputted character will
// have a prediction case. skipSize of 3 means every 3rd will be predicted,
// and there will be 1/3 as many test cases.
var skipSize *int

/* end flags */

/* setColumnGlobals(int) MUST be called as soon as possible */

func main() {
	trn := flag.Bool("train", false, "Train a model")
	dataFile = flag.String("data", "", "Training data input file")
	saveTo = flag.String("save", "", "Where to save the model after training")
	treesPerFold = flag.Int("trees", 1, "How many decision trees to make per fold of the dataset")
	n_folds = flag.Int("folds", 5, "How many subdivisions of the dataset to make for cross-validation")

	pred := flag.Bool("pred", false, "Make a prediction")
	modelFile = flag.String("model", "", "Load a pretrained model for prediction")
	seedText = flag.String("seed", "", "Predict based on this string of data")
	charMode = flag.Bool("charmode", false, "Character prediction mode rather than numeric feature mode. This will create test cases by iterating through the data `skipSize` at a time, and making the previous `sequenceLength` items have higher weights based on the closeness to the current item being predicted.s")
	skipSize = flag.Int("skipsize", 3, "During -charmode, how many items to skip before making another training case")
	subsetSizePercent = flag.Float64("subsetpct", 0.6, "Percent of the dataset which should be used to train a tree (always minus 1 fold for cross-validation)")

	overrideFeatureSplitSize = flag.Int("m", 0, "Override calculation for feature split size (little m)")
	overrideSequenceLength = flag.Int("seqlen", 0, "Normally equal to the number of variables during -charmode, override for fewer previous look-behind-memory-variables in every input test cases")

	prof = flag.String("profile", "", "[cpu|mem] enable profiling")

	tojson := flag.Bool("tojson", false, "Convert a model to json")
	flag.Parse()

	if *prof == "mem" {
		defer profile.Start(profile.MemProfile).Stop()
	} else if *prof == "cpu" {
		defer profile.Start(profile.CPUProfile).Stop()
	}

	if *trn {
		if *dataFile == "" {
			fmt.Println("-data flag is required and should be a path to input data")
			return
		}
		if *saveTo == "" {
			fmt.Println("-save flag is required and should be a path for saving the model")
			return
		}
		if *skipSize < 1 {
			fmt.Println("-skipsize must be greater than 0")
			return
		}
		train()
		return
	}

	if *pred {
		if *modelFile == "" {
			fmt.Println("-model is required and should be a path for loading the pretrained model")
			return
		}
		if *seedText == "" {
			fmt.Println("-seed text is required")
			return
		}
		predict()
		return
	}

	if *tojson {
		if *modelFile == "" {
			fmt.Println("-model is required and should be a path for loading the pretrained model")
			return
		}
		gobToJson()
		return
	}

	usage()
}

func usage() {
	fmt.Println("Train or a random decision ensemble, or make a prediction from one.\n  tree [-train|-predict] [options]\n  Options:")
	flag.PrintDefaults()
}

func train() {
	// seed the random number generator
	rand.Seed(time.Now().Unix())

	fmt.Println("Reading data file", *dataFile)
	buf, err := ioutil.ReadFile(*dataFile)
	if err != nil {
		panic(err)
	}
	trainingData = string(buf)
	variables = make(map[string]float32)

	// setup variables from the data
	// we will get them, then shuffle the letters
	if *charMode {
		fmt.Println("Running in character mode - one hot encoding")
		allChars := getCharmodeInputText(trainingData)
		var c string
		// first find the unique letters
		var tempAllVars []string
		for i := 0; i < len(allChars); i++ {
			c = allChars[i]
			if _, existsYet := variables[c]; !existsYet {
				tempAllVars = append(tempAllVars, c) // it's in order here, but we will shuffle
				variables[c] = -1                    // track whether we have seen it before
			}
		}
		// now shuffle them - otherwise the left side will be artificially
		// favored because the more common letters favor the beginning of the array
		perm := rand.Perm(len(variables))
		for _, randIndex := range perm {
			// get a random letter
			c = tempAllVars[randIndex]
			// add the random letter into a tracked but now randomly ordered list
			indexedVariables = append(indexedVariables, c)
			// use that index to put it in the map of variables
			newIndex := len(indexedVariables) - 1
			variables[c] = float32(newIndex)
		}

		sequenceLength = *overrideSequenceLength
		if sequenceLength == 0 {
			sequenceLength = len(variables)
		}
		fmt.Println("sequence length=", sequenceLength)
		trainingCases = encodeLettersToCases(allChars)
	} else { // NOT character prediction mode
		rows := strings.Split(trainingData, "\n")
		col1 := strings.Split(rows[0], ",")
		setColumnGlobals(len(col1))
		for rowIndex, row := range rows {
			nextCase := parseRow(row, rowIndex)
			trainingCases = append(trainingCases, nextCase)
		}
	}

	// there are columnsPerRow items per data row

	n_features = *overrideFeatureSplitSize
	if n_features == 0 {
		// most likely we will want to use this, but during word mode we may limit
		// the feature size
		n_features = int(math.Sqrt(float64(columnsPerRow)))
	}

	fmt.Println("features:", lastColumnIndex)
	fmt.Println("data folds:", *n_folds)
	fmt.Println("trees per fold:", *treesPerFold)
	fmt.Println("prediction categories:", len(variables))
	fmt.Println("feature split size (m):", n_features)
	fmt.Println("training cases:", len(trainingCases))

	parallelTrees = int(math.Ceil(math.Max(2, float64(runtime.NumCPU())/float64(*n_folds))))
	fmt.Println("concurrent trees:", parallelTrees, "*", *n_folds, "=", parallelTrees*(*n_folds))

	// run the training testing various numbers of Trees to see how many we need
	var trees []*Tree
	var scores []float32
	saveNow := func() {
		s := &saveFormat{
			Trees:            trees,
			IndexedVariables: indexedVariables,
			Variables:        variables,
		}
		save(*saveTo, s)
		fmt.Println("\nSaved", len(trees), "trees and", len(indexedVariables), "variables to", *saveTo)
	}
	//var t *time.Ticker
	//go (func() {
	//	t = time.NewTicker(15 * time.Minute)
	//	for {
	//		select {
	//		case <-t.C:
	//			saveNow()
	//		}
	//	}
	//})()

	// this is the thing that begins running
	scores, trees = evaluateAlgorithm()

	//t.Stop() // prevent saving conflict top the save below

	fmt.Println("\nComplete.")
	fmt.Println("\nTrees per fold:", *treesPerFold)
	fmt.Println("  Fold Scores:", scores)
	fmt.Println("  Mean Accuracy:", sum(scores)/float32(len(scores)), "%")

	saveNow()
}

func setColumnGlobals(lenFirstRow int) {
	columnsPerRow = lenFirstRow
	lastColumnIndex = columnsPerRow - 1
}

func predict() {
	var loaded saveFormat
	err := load(*modelFile, &loaded)
	if err != nil {

		panic(err)
	}
	fmt.Println(len(loaded.Trees), "Trees loaded")

	variables = loaded.Variables
	sequenceLength = len(variables)
	indexedVariables = loaded.IndexedVariables

	var inputRows []datarow

	if *charMode {
		skipOne := 1
		skipSize = &skipOne // force this, to use all items

		seedChars := getCharmodeInputText(*seedText)
		if sequenceLength > len(seedChars) {
			sequenceLength = len(seedChars)
		}
		inputRows = encodeLettersToCases(seedChars)

		var mostFreqVar float32
		var lastPrediction string
		for _, irow := range inputRows {
			mostFreqVar = baggingPredict(loaded.Trees, irow)
			lastPrediction = indexedVariables[int(mostFreqVar)]
			fmt.Print(lastPrediction)
		}

		// now feed it back onto itself until stopping
		var irow datarow
		for {
			// make a row with only the last prediction in it
			irow = encodeLettersToCases([]string{lastPrediction})[0]
			mostFreqVar = baggingPredict(loaded.Trees, irow)
			lastPrediction = indexedVariables[int(mostFreqVar)]
			fmt.Print(lastPrediction)
		}
		// unreachable return
	}

	// not character mode

	// need to set this global first
	inputRow := strings.Split(*seedText, ",")
	setColumnGlobals(len(inputRow))

	inputRows = []datarow{parseRow(*seedText, 0)}
	for _, irow := range inputRows {
		mostFreqVar := baggingPredict(loaded.Trees, irow)
		fmt.Print(indexedVariables[int(mostFreqVar)])
	}

	fmt.Println()
}

func getCharmodeInputText(s string) (cases []string) {
	r := strings.Replace(s, "\n", " ", -1)
	cases = strings.Split(r, charmodeSplitChar)
	return cases
}

/*
encodeLettersToCases uses modified one hot encoding, so called "multi-hot encoding."

Each char is a feature in the row.

Each letter fires the next letter. Each character of a set with length `sequenceLength`
will get an equally increased amount in the training case. So in essence:

"hey u"
{"h", "e", "y", " ", "u"} 	   (but with their variable indexes)
{0.2, 0.4, 0.6, 0.8, 1.0}

The predicted letter is the last column.
*/
func encodeLettersToCases(allChars []string) (cases []datarow) {
	setColumnGlobals(len(indexedVariables) + 1)
	var letter string
	var sequenceWeight float32
	var indexDistance int
	// if there are less characters, at least
	ranOnce := false
	var nextEnd int // next end of the current sequence
	for letterIndex := sequenceLength; !ranOnce || letterIndex < len(allChars); letterIndex += *skipSize {
		ranOnce = true
		nextCase := make(datarow, columnsPerRow) // zero is default

		// many-hot encoding
		// each variable gets a different value, such that the most recent
		// in the sequence gets the highest possible value, and the
		// least recent in the sequence gets the lowest value
		nextEnd = int(math.Min(float64(len(allChars)), float64(letterIndex)))
		sequence := allChars[letterIndex-sequenceLength : nextEnd]

		// start i, the index of allChars, at the least recent sequence
		for i := 0; i < sequenceLength-1; i++ {
			letter = sequence[i]
			sequenceWeight = float32(i+1) / float32(sequenceLength)
			if sequenceWeight > 1 || sequenceWeight <= 0 {
				fmt.Println("i=", i, "letter=", letter, "indexDistance=", indexDistance, "sequenceWeight=", sequenceWeight)
				panic("sequence weight should be between 0 and 1.0")
			}
			nextCase[int(variables[letter])] = sequenceWeight
		}

		if letterIndex < len(allChars)-1 { // should always be true except during prediction
			letter = allChars[letterIndex]
			nextCase[lastColumnIndex] = variables[letter] // the variable index being predicted
		}
		cases = append(cases, nextCase)
		//fmt.Println(nextCase)
	}
	return cases
}

func gobToJson() {
	var loaded saveFormat
	err := load(*modelFile, &loaded)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(loaded.Trees), "Trees loaded")

	buf, err := json.Marshal(loaded)
	if err != nil {
		panic(err)
	}
	var outFile string
	if *saveTo != "" {
		outFile = *saveTo
	} else {
		base := filepath.Base(*modelFile)
		outFile = strings.Replace(base, filepath.Ext(base), ".json", 1)
	}
	err = ioutil.WriteFile(outFile, buf, os.ModePerm)
	if err != nil {
		panic(err)
	}
	fmt.Println("Wrote JSON to", outFile)
}
