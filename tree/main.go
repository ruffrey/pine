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
var treesPerFold *int
var indexedVariables []string    // index to character
var variables map[string]float32 // character to index
// first len-1 are considered predictors, last one is the letter index to be predicted
var trainingCases []datarow

// maxDepth is the maximum depth of child nodes allowed from the root of a tree
var maxDepth = 8
var n_folds *int      // how many folds of the dataset for cross-validation
var n_features int    // Little `m`, will get rounded down
var columnsPerRow int // how many total columns in a row. must be the same.
// how many inputs are fed into the network during a sample; similar to sequence length with neural networks
var lastColumnIndex int // columnsPerRow minus 1
var sequenceLength int  // for character mode
// in the dataset (minus 1 fold for cross-validation), how many samples
// should be taken from the dataset (with replacement) to train each tree?
// as a percent of dataset
var subsetSizePercent = 0.5

var charMode *bool

// flags
var dataFile *string
var modelFile *string
var saveTo *string
var seedText *string
var prof *string
var parallelTrees int = 1 // how many trees to build at once (per fold)

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
	charMode = flag.Bool("charmode", false, "Character prediction mode rather than numeric feature mode")

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
	if *charMode {
		fmt.Println("Running in character mode - one hot encoding")
		allChars := strings.Split(trainingData, "")
		var c string
		for i := 0; i < len(allChars); i++ {
			c = allChars[i]
			if _, existsYet := variables[c]; !existsYet {
				indexedVariables = append(indexedVariables, c)
				newIndex := len(indexedVariables) - 1
				variables[c] = float32(newIndex)
			}
		}
		sequenceLength = len(variables) * 2
		fmt.Println("sequence length=", sequenceLength)
		trainingCases = encodeLettersToCases(allChars, false)
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
	n_features = int(math.Sqrt(float64(columnsPerRow)))

	fmt.Println("features:", lastColumnIndex)
	fmt.Println("data folds:", *n_folds)
	fmt.Println("prediction categories:", variables)
	fmt.Println("feature split size (m):", n_features)
	fmt.Println("training cases:", len(trainingCases))

	parallelTrees = int(math.Max(2, float64(runtime.NumCPU())/float64(*n_folds)))
	fmt.Println("concurrent trees:", parallelTrees)

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
	indexedVariables = loaded.IndexedVariables

	var inputRows []datarow

	if *charMode {
		seedChars := strings.Split(*seedText, "")
		inputRows = encodeLettersToCases(seedChars, true)
	} else {
		// need to set this global first
		inputRow := strings.Split(*seedText, ",")
		setColumnGlobals(len(inputRow))

		inputRows = []datarow{parseRow(*seedText, 0)}
	}

	for _, irow := range inputRows {
		mostFreqVar := baggingPredict(loaded.Trees, irow)
		fmt.Print(indexedVariables[int(mostFreqVar)])
	}
	fmt.Println()
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
func encodeLettersToCases(allChars []string, isPrediction bool) (cases []datarow) {
	setColumnGlobals(len(indexedVariables) + 1)
	var letter string
	var sequenceWeight float32
	var indexDistance int

	// for predictions shorter than sequence length
	if isPrediction {
		// fill in whatever is missing in a full sequence with randomness
		var rem int
		if len(allChars) < sequenceLength {
			rem = sequenceLength - len(allChars)
		} else {
			rem = len(allChars) % sequenceLength
		}
		for i := 0; i < rem+1; i++ {
			allChars = append(allChars, indexedVariables[rand.Intn(len(variables))])
		}
	}

	for letterIndex := sequenceLength; letterIndex < len(allChars); letterIndex++ {

		nextCase := make(datarow, columnsPerRow) // zero is default

		// many-hot encoding
		// each variable gets a different value, such that the most recent
		// in the sequence gets the highest possible value, and the
		// least recent in the sequence gets the lowest value

		sequence := allChars[letterIndex-sequenceLength : letterIndex]

		// start i, the index of allChars, at the least recent sequence
		for i := 0; i < sequenceLength; i++ {
			letter = sequence[i]
			sequenceWeight = float32(i+1) / float32(sequenceLength)
			if sequenceWeight > 1 || sequenceWeight <= 0 {
				fmt.Println("i=", i, "letter=", letter, "indexDistance=", indexDistance, "sequenceWeight=", sequenceWeight)
				panic("sequence weight should be between 0 and 1.0")
			}
			nextCase[int(variables[letter])] = sequenceWeight
		}

		letter = allChars[letterIndex]
		nextCase[lastColumnIndex] = variables[letter] // the variable index being predicted
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
