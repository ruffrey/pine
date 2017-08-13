# pine

random forest attempt

# http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics

## One explanation:

1. We have many connected decision trees.
    - a decision tree can have many nodes
2. Each decision tree gets a random subset of all data.
3. At each node of the decision tree, choose some small random subset of variables from the data its tree got.
    - find a variable (and its value) which optimizes the split

## More detailed:

- N = total cases in a subset of all data
- M = total number of predictor variables
- m = number of predictor variables to select from all predictor variables on a node
    - three possible values for m: ½√m, sqrt(m), and 2√m
- cost = function that will evaluate whether a predictor variable had the best split on
    a parcicular node. this might be Gini, or something else

0. Enter an input into the forest. It will run down all the trees.
1. Sample 2/3 of the data, the total cases will be N, so every decision tree has a subset.
2. For every decision tree: for every node in the tree:
    a. take m predictor variables from the subset
    b. evaluate which predictor variable provides the "best split" - in other words, which
        one has the best value according to running the cost function
    c. do a binary split on the node with the best split from the last step
3. Take the output (y) of all the terminal nodes.
    - if it was numeric data, do an average or weighted average to determine the final output
    - if it was categorical data, take the voting majority


## Cross validation:

Does the forest predict better than just guessing (50%)?
Take each input sample, and hold out every one:
    
```go
var holdoutIndex int = -1
var holdoutTestSample sample
for i := 0; i < len(samples); i++ {
    holdoutIndex++
    holdoutTestSample = samples[holdoutIndex]
    samplesWithoutHoldout = append(samples[0:holdoutIndex], samples[holdoutIndex+1:]...)
}
```


----

# from wikipedia

## construct a multitude of decision trees

the output the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees

## random forest

given a training set  X which is an array of input data, with corresponding response array Y, do bagging B times (few hundred to a few thousand).

A bag selects a random training item from the set, and “fits trees” to the samples:

f_b is a decision or regression tree

```
for b := 0; b < B; b++ {
  index = randomIndex(X)
  sample = X[index]
  expected = Y[index]
  // train decision trees??

  majorityVote = getVote(trees) // ??
}
```
