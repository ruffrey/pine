package main

import (
	"strings"
)

var trainingData = "Hey there my name is Jeff. What is up? How are you. Hi there dude."
var trainingCases []string

var N_totalTrainingCases int
var M_totalVarsInClassifier int

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

	N_totalTrainingCases = len(trainingCases)
	M_totalVarsInClassifier = len(uniqueChars)
}
