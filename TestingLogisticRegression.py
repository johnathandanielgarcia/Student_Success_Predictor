import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.LogisticRegressionModel.LogisticRegression import LogisticRegression

# helper to turn entire df into expected X, y for decision tree
def df_to_Xy(df, target_col):
    X = df.drop(columns = [target_col]).values
    y = df[target_col].values
    return X, y

#Load CSV data
df = pd.read_csv("data/LogReg_formatted_final_dataset.csv")

#Split the Data Frame into features and labels
# X, y = df_to_Xy(df, "GradeClass")
df["PassFail"] = df["GradeClass"].apply(lambda x: 1 if x <= 2 else 0) #convert to pass/fail

X, y = df_to_Xy(df.drop(columns=["GradeClass"]), "PassFail")


#Convert labels to binary; 1's stay 1 else become 0
# y = np.where(y == 1, 1, 0)

#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

# clf = LogisticRegression(learning_rate=0.1, max_it=1000)

#train model with training data
# clf.fit(X_train, y_train)

# Predict on the test set then calculate and pint accuracy of predictions
# predicted = clf.predict(X_test)
# accuracy = np.mean(predicted == y_test)
# print(f"Logistic Regression Accuracy: {accuracy:.3f}")

for learning_rate in [.01, .02, .03, .04, .001, .002, .003, .004]:
    for max_iter in [500, 750, 1000, 1250, 1500, 1750]:
        lr = LogisticRegression(learning_rate=learning_rate, max_it=max_iter)
        lr.fit(X_train, y_train)
        acc = (lr.predict(X_test) == y_test).mean()
        print(f"Learning Rate={learning_rate}, Max Iterations={max_iter} -> Acc={acc:.3f}")