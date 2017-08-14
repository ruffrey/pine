# pine

ensembles of random decision trees

![random decision tree ensembles training](decision-ensembles.png)

## how it works

Given a data set, rows of input features x, where the last column is the expected category y.
Often these are encoded in CSV format. The data should be encoded to float32 parseable values.

Train a set of random decison trees per bag:
- Given a training set (group of every bag/folds except one, see cross validation below) and a test training set
- Do the following to create however many trees you want in each set:
    - randomly select a sample that is 2/3 of the training set with replacement. There are likely duplicate values.
    - With the sample, determine the best split point of the data:
        - `M` is the total number of input features.
        - `m` is a subset of features from the total number of features that each tree will be responsible for caring about. In other words, each tree will try to best predict only `m` out of `M` total features. Ways to calculate `m` are like the square root of `M` or other ways to produce a smaller value.
        - So, randomly pick `m` features for this tree to care about.
        - For each feature, run through every possible split of the features of the input rows.
        - Track which predictions are correct and which are not. Use the split that has the most correct predictions on this data sample. This is also called minimizing the Gini error.  
    - Continue splitting the tree into nodes until reaching the maximum desired depth, or until naturally reaching the end of the tree.

Once you have a trained tree, a prediction is made by running a sample without the last column through every tree, and getting the mode (most frequent) prediction across the trees.

### Cross-validation

Cross-validation is a way minimize the out-of-bag error. In other words, we validate that samples *not* in the bags are still predicted correctly.
 
To do it, start by splitting the whole dataset into equal bags (or folds) without replacement, before creating the bags.

For example, say there are 20 samples and we want 4 folds. Each fold will have 5 samples, and none of the 20 samples will be repeated across all the folds. However, they need to be put randomly into the folds (random without replacement).

Next, loop through all the folds. The fold in the loop iteration will be the test set, so reserve it for later. Use all the other folds to train a set of decision trees. In our example above, that means on the first fold, we would use the last 3 for training, on the second, use the first fold and the last two for training, etc. For every training set, construct decision trees that best predicts it.
