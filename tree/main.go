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
var maxDepth = 10
var n_folds *int      // how many folds of the dataset for cross-validation
var n_features int    // Little `m`, will get rounded down
var columnsPerRow int // how many total columns in a row. must be the same.
// how many inputs are fed into the network during a sample; similar to sequence length with neural networks
var lastColumnIndex int     // columnsPerRow minus 1
var sequenceLength int = 25 // for character mode

var charMode *bool

// flags
var dataFile *string
var modelFile *string
var saveTo *string
var seedText *string
var prof *string

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
	rand.Seed(time.Now().Unix())
	fmt.Println("Reading data file", *dataFile)
	buf, err := ioutil.ReadFile(*dataFile)
	if err != nil {
		panic(err)
	}
	trainingData = string(buf)
	variables = make(map[string]float32)

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
		sequenceLength = len(variables) / 2
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

	// run the training testing various numbers of Trees to see how many we need
	var trees []*Tree
	var scores []float32

	scores, trees = evaluateAlgorithm()
	fmt.Println("\nTrees per fold:", *treesPerFold)
	fmt.Println("  Fold Scores:", scores)
	fmt.Println("  Mean Accuracy:", sum(scores)/float32(len(scores)), "%")

	s := &saveFormat{
		Trees:            trees,
		IndexedVariables: indexedVariables,
		Variables:        variables,
	}
	save(*saveTo, s)
	fmt.Println("\nSaved", len(trees), "trees and", len(indexedVariables), "variables to", *saveTo)
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
		for i := letterIndex - sequenceLength; i < letterIndex; i++ {
			letter = allChars[i]
			indexDistance = letterIndex - i
			sequenceWeight = 1 + (1-float32(indexDistance))/float32(sequenceLength)
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
	base := filepath.Base(*modelFile)
	outFile := strings.Replace(base, filepath.Ext(base), ".json", 1)
	err = ioutil.WriteFile(outFile, buf, os.ModePerm)
	if err != nil {
		panic(err)
	}
	fmt.Println("Wrote JSON to", outFile)
}
