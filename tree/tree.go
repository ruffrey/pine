package main

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

	leftSamples  []datarow // temp test cases for left group
	rightSamples []datarow // temp test cases for right group
}
