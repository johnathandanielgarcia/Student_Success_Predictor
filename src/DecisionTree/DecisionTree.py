import numpy as np
from collections import Counter

# variables will be named by word1_word2 rather than word1Word2 to follow python conventions 

'''
TreeNode object:
self.feature: the feature the node was split on
self.threshold: the threshold for which we split the node
self.left: left child of the node
self.right: rigth child of the node
self.value: only applicable to leaf nodes, holds the decision, in this case it holds the 'GradeClass' or letter grade encoded as an int 
'''
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): # '*' means we HAVE to pass value by name which will help be able to tell which nodes are leaf nodes
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    '''
    returns whether a node is a leaf node by determining whether it has a a value
    '''
    def isLeaf(self):
        return self.value is not None


'''
Decision Tree object:
self.min_sample_split: a stopping condition in which we stop splitting if a node has less samples than this value 
self.max_depth: another stopping condition in which we stop splitting once our tree reaches this depth
self.num_features: specifies how many features we want to split by, this parameter lets users dictate if they want to split by less than the total number of features in a dataset
self.root: holds the root of the tree
'''
class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, num_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None


    '''
    Purpose: 
        Builds decision tree from training data.
    Inputs:
        X: numpy.ndarray, 2D array that must have shape (number of samples, number of features) and hold the data
        y: numpy.ndarray, 1D array that must have shape (number of samples, ) that has 'class labels' aka the target variable
    How It Works:
        1. Reassigns num_features to the total number of features if user did not specify how many to split by in constructor, otherwise it reassigns to the minimum of the number of features specified by user or in the data. 
        This ensures we dont try to access more features than we have.
        2. Then, call a helper function to grow the tree, aka split the tree until we hit a stopping condition 
    '''
    def fit(self, X, y):
        # makes sure we dont get error if user specifies more features than there are in the data
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)

        # call helper to grow tree
        self.root = self.grow_tree(X, y)


    '''
    Purpose:
        Recursively build the decision tree until we reach stopping conditions. 
    Inputs:
        X: nunpy.ndarray, 2D array that must have shape (number of samples, number of features) and hold the data
        y: numpy.ndarray, 1D array that must have shape (number of samples, ) that has 'class labels' aka the target variable
        depth: the depth we are currently at in the tree
    How It Works:
        1. Get the number of samples, features, and possible labels from the data to check whether we've reached any stopping conditions (max depth, pure node, too few samples). If so, return leaf node. 
        2. Randomly choose subset of features for splitting.
        3. Call a helper function, 'find_best_split' to find the best way to split the data at current depth (best defined as maximixing information gain aka minimizing entropy, way defined as feature and threshold)
        4. Split data into left and right child nodes based on best split.
        5. If split has no nodes in one direction, create leaf node with most commom label in the leaf. Else, build left and right subtrees recursively by calling 'grow_tree'
        6. Once all calls terminated, return TreeNode with its feature, thresholdm and child nodes. 
    '''
    def grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # need to check stopping criteria, if any are true then we return a leaf node
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_sample_split):
            leaf_value = self.most_common_label(y) # use helper function to get the value of our leaf node which is the most common value
            return TreeNode(value = leaf_value)

        features_idx = np.random.choice(num_features, self.num_features, replace = False) # when we select random group we only want unique features (replace = False)

        # find best split 
        best_feature, best_threshold = self.find_best_split(X, y, features_idx)

        # create new child nodes
        left_idxs, right_idxs = self.split(X[:, best_feature], best_threshold)

        # ADDED 
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self.most_common_label(y)
            return TreeNode(value = leaf_value)

        # recursively call grow_tree
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return TreeNode(best_feature, best_threshold, left, right)
    


    '''
    Purpose: 
        Find the most occuring value in a subset of target values 
    Inputs:
        y: numpy.ndarray, 1D array of target values in a given node 
    How It Works:
        1. Initialize a counter variable on y
        2. Access and store the label from the first (label, count) in the first most frequent value
    '''
    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    


    '''
    Purpose:
        Find the best feature and threshold for which to split a node, with 'best' defined as the highest information gain
    Inputs:
        X: nunpy.ndarray, 2D array that has feature values in a given node
        y: numpy.ndarray, 1D array that has 'class labels' aka the target variable in a given node
        feature_idxs: numpy.ndarray containing indices of features to iterate through to find the best split
    How It Works:
        1. Iterate through every feature assocoiated with the feature index in feature_idxs
        2. Get every unique value in the feature column and store them as potential thresholds for splitting 
        3. Iterate through every unique threshold
        4. Calculate the information gain for current threshold using helper function 'information_gain(...)'
        5. If current information gain is greater than best information gain, update it and store the threshold associated with it
    '''
    # find threshold for all possible splits to find best one
    def find_best_split(self, X, y, feature_idxs):
        best_gain = -1 # to keep track of best split
        split_idx, split_threshold = None, None # return values
        for feature_idx in feature_idxs:
            X_col = X[:, feature_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                # find information gain using helper function
                curr_gain = self.information_gain(y, X_col, threshold)

                if curr_gain > best_gain:
                    best_gain = curr_gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold
    


    '''
    Purpose:
        Calculate the information gain from splitting a node on a given threshold
    Inputs:
        y: numpy.ndarray, 1D array that has 'class labels' aka the target variable in a given node
        X_col: numpy.ndarray, 1D array containing all the values in a given feature column being considered
        threshold: a particular value, int or float, in the feature column being considered as the split condition of the node
    How It Works:
        1. Calculate the entropy of calling node by calling helper function 'entropy()'
        2. Split node and save the left and right indices by calling helper function 'split()'
        3. If one of the split nodes has no indices, then there is no information gain
        4. Otherwise, calculate the information gain, defined as -> parent entropy - weighted avg of children entropy after split
    '''
    def information_gain(self, y, X_col, threshold):
        # we need to find the parent entropy, then create children and find the weighted avg of their entropies, then subtract to find the information gain
        parent_entr = self.entropy(y)

        left_child_idxs, right_child_idxs = self.split(X_col, threshold)

        if len(left_child_idxs) == 0 or len(right_child_idxs) == 0: # if we split and get no idxs to go in a direction, then information gain is 0
            return 0
        
        num_samples = len(y)
        num_samples_left, num_samples_right = len(left_child_idxs), len(right_child_idxs)
        left_entr, right_entr = self.entropy(y[left_child_idxs]), self.entropy(y[right_child_idxs])
        children_entr = (num_samples_left / num_samples) * left_entr + (num_samples_right / num_samples) * right_entr

        information_gain = parent_entr - children_entr
        return information_gain
    


    '''
    Purpose:
        Calculate the entropy of a subset of target values
    Inputs:
        y: numpy.ndarray, 1D array containing target values
    How It Works:
        1. Get counts of every unique value in the target feature
        2. Calculate probability of every unique value in the target feature
        3. Calculate the entropy defined as (-)summation of p(x)*log(p(x)), ignoring probabilities of 0 to avoid undefined log
    '''
    def entropy(self, y):
        # entropy = -summation of p(x)*log(p(x)), where p(x) is the probability of each value
        hist = np.bincount(y) # bincount returns an array where the values corresponding to an idx represent how many times that idx number appears in the argument, can think of it like a histogram where the idxs are x-values and arr[idx] is the y value
        probabilities = hist / len(y) # creates the proportion or probability of every value
        return -np.sum([prob * np.log(prob) for prob in probabilities if prob>0]) # return entropy as calculated in first comment
    


    '''
    Purpose: 
        Split a node based on a given threshold
    Inputs:
        X_col: numpy.ndarray, 1D array containing all the values in a single feature column for all samples in the current node
        split_threshold: the threshold we are splitting the node by. All values less than or equal to this threshold should go left, all values greater than this threshold should go right
    How It Works:
        1. Call 'np.argwhere()' with X_col's values and the conditions <= / > to perform logical indexing
        2. Use '.flatten()' to convert the return value of 'np.argwhere()' into a 1D array
        3. Return the arrays of left and right indices
    '''
    def split(self, X_col, split_threshold):
        # need to find idxs that go in each direction when we split
        left_idxs = np.argwhere(X_col <= split_threshold).flatten() # argwhere is essentially logical index, will return values where condition is true, we use 'flatten' to have it return a 1D array instead of a 2D array
        right_idxs = np.argwhere(X_col > split_threshold).flatten()
        return left_idxs, right_idxs



    '''
    Purpose:
        Predict classifications of new data using fitted decision tree
    Inputs:
        X: numpy.ndarray, 2D array that must have shape (number of samples, number of features) and hold the data to be predicted on
    How It Works:
        1. Calls helper function 'traverse_tree()' for every datapoint in X and returns the values as a numpy.ndarray
    '''
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
    


    '''
    Purpose:
        Given a datapoint, traverse the fitted decision tree until a leaf node is reached and return the classification, or class label, at that leaf node
    Inputs:
        x: np.ndarray, a 1D array containing the data for one datapoint
        node: TreeNode, the node that the datapoint is traversing through in the current function call
    How It Works:
        1. If the current node is a leaf node, return the value at the node - this is the classification
        2. Otherwise, traverse the left subtree if x's value for the node's feature is <= the split threshold at the node
        3. Else, traverse the right subtree
    '''
    def traverse_tree(self, x, node):
        if node.isLeaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)





