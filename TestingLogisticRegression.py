import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.LogisticRegressionModel.LogisticRegression import LogisticRegression

# helper to turn entire df into expected X, y for decision tree
def df_to_Xy(df, target_col):
    X = df.drop(columns = [target_col]).values
    y = df[target_col].values
    return X, y

df = pd.read_csv("data/LogReg_formatted_final_dataset.csv")
X, y = df_to_Xy(df, "GradeClass")


#Maped to binary
y = np.where(y == 1, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

clf = LogisticRegression(learning_rate=0.1, max_it=1000)
clf.fit(X_train, y_train)

# predict expects 2d array, reshape males this into 1 row for the 1 sample and infers the number of columns for the features
predicted = clf.predict(X_test)
accuracy = np.mean(predicted == y_test)
print(f"Logistic Regression Accuracy: {accuracy:.3f}")