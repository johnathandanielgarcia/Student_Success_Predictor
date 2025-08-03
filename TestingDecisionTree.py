import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.DecisionTree import DecisionTree

# helper to turn entire df into expected X, y for decision tree
def df_to_Xy(df, target_col):
    X = df.drop(columns = [target_col]).values
    y = df[target_col].values
    return X, y

df = pd.read_csv("data/Decision_Tree_formatted_final_dataset.csv")
X, y = df_to_Xy(df, "GradeClass")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

clf = DecisionTree.DecisionTree(min_sample_split=5, max_depth=10)
clf.fit(X_train, y_train)

# predict expects 2d array, reshape males this into 1 row for the 1 sample and infers the number of columns for the features 
x_test0_2d = X_test[0].reshape(1, -1)
print(clf.predict(x_test0_2d))



'''
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    predictions = clf.predict(X_test)
    return np.sum((y_test == y_pred) / len(y_test))

acc = accuracy(predictions, y_test)
print(acc)
'''



# iterating through different arguments to see which produces best accuracy
'''
for depth in [3, 5, 10, 20]:
    for min_split in [2, 5, 10]:
        clf = DecisionTree.DecisionTree(max_depth=depth, min_sample_split=min_split)
        clf.fit(X_train, y_train)
        acc = (clf.predict(X_test) == y_test).mean()
        print(f"Depth={depth}, MinSplit={min_split} -> Acc={acc:.3f}")
'''


