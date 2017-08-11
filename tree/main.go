package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strings"
	"time"

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

var treeSizesToTest = []int{1}
var trainingData string
var totalTrees = 5
var indexedVariables []string    // index to character
var variables map[string]float32 // character to index
// first len-1 are considered predictors, last one is the letter index to be predicted
var trainingCases []datarow
var maxDepth = 10
var n_folds = 5       // how many folds of the dataset for cross-validation
var n_features int    // Little `m`, will get rounded down
var columnsPerRow int // how many total columns in a row. must be the same.

var charMode *bool

// flags
var dataFile *string
var modelFile *string
var saveTo *string
var seedText *string
var prof *string

func main() {
	trn := flag.Bool("train", false, "Train a model")
	dataFile = flag.String("data", "", "Training data input file")
	saveTo = flag.String("save", "", "Where to save the model after training")

	pred := flag.Bool("pred", false, "Make a prediction")
	modelFile = flag.String("model", "", "Load a pretrained model for prediction")
	seedText = flag.String("seed", "", "Predict based on this string of data")
	charMode = flag.Bool("charmode", false, "Character prediction mode rather than numeric feature mode")

	prof = flag.String("profile", "", "[cpu|mem] enable profiling")
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

		// Use one hot encoding. Each char is an feature in the row.
		// Each letter fires the next letter. Only one feature will
		// not be 0, the letter's index, and it will be 1. The predicted
		// letter is the last column.
		var prevLetter string
		var letter string
		columnsPerRow = len(indexedVariables) + 1
		for i := 1; i < len(allChars); i++ {
			nextCase := make(datarow, columnsPerRow)      // zero is default
			nextCase[int(variables[prevLetter])] = 1      // the one that is hot
			nextCase[columnsPerRow-1] = variables[letter] // the variable index being predicted
			trainingCases = append(trainingCases, nextCase)
		}

	} else { // NOT character prediction mode
		rows := strings.Split(trainingData, "\n")
		col1 := strings.Split(rows[0], ",")
		columnsPerRow = len(col1) // globally set
		for rowIndex, row := range rows {
			nextCase := parseRow(row, rowIndex)
			trainingCases = append(trainingCases, nextCase)
		}
	}

	// there are columnsPerRow items per data row
	n_features = int(math.Sqrt(float64(columnsPerRow)))

	fmt.Println("features:", columnsPerRow-1)
	fmt.Println("prediction categories:", variables)
	fmt.Println("feature split size (m):", n_features)

	// run the training testing various numbers of Trees to see how many we need
	var trees []*Tree
	var scores []float32
	for _, n_trees := range treeSizesToTest {
		scores, trees = evaluateAlgorithm()
		fmt.Println("\nTrees:", n_trees)
		fmt.Println("  Scores:", scores)
		fmt.Println("  Mean Accuracy:", sum(scores)/float32(len(scores)), "%")
	}
	s := &saveFormat{
		Trees:            trees,
		IndexedVariables: indexedVariables,
		Variables:        variables,
	}
	save(*saveTo, s)
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

	// need to set this global first
	cols := strings.Split(*seedText, ",")
	columnsPerRow = len(cols)

	inputRow := parseRow(*seedText, 0)
	mostFreqVar := baggingPredict(loaded.Trees, inputRow)
	fmt.Println("Prediction:", indexedVariables[int(mostFreqVar)])
}
