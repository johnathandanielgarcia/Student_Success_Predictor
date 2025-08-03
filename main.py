'''
Give user the option to:
1. train or fit each model
2. see how the model performs with different parameters
3. input custom parameters to see how model performs with them
4. input their own custom data and predict on it using the best performing model or model with their chosen parameters

If time allows:
- all features above deployed on the web using flask or some other web framework
- include visuals showing model performance across different parameters 
- visualize the models being fitted 
'''
from src.DecisionTree.DecisionTree import DecisionTree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def df_to_Xy(df, target_col):
    X = df.drop(columns = [target_col]).values
    y = df[target_col].values
    return X, y

def valid_option(value, acceptable_values):
    return True if value in acceptable_values else False

def prediction_to_grade(prediction):
    match prediction:
        case 0:
            return 'A'
        case 1:
            return 'B'
        case 2:
            return 'C'
        case 3:
            return 'D'
        case 4:
            return 'F'
        case _:
            print('Error')

def train_decision_tree(min_sample_split, max_depth):
    df = pd.read_csv("data/Decision_Tree_formatted_final_dataset.csv")
    X, y = df_to_Xy(df, "GradeClass")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

    dt = DecisionTree(min_sample_split, max_depth)
    dt.fit(X_train, y_train)
    return dt, X_test, y_test

def decision_tree_accuracy(dt, X_test, y_test):
    accuracy = (dt.predict(X_test) == y_test).mean()
    return accuracy

def user_trained_decision_tree():
    user_min_sample_split = int(input("Please enter the min_sample_split you'd like to use: "))
    user_max_depth = int(input("Please enter the max_depth you'd like to use: "))
    print("Training...")

    dt, X_test, y_test = train_decision_tree(user_min_sample_split, user_max_depth)
    acc = decision_tree_accuracy(dt, X_test, y_test)

    print(f"Decision Tree model trained with your parameters have achieved an accuracy of {acc:.4f}")
    digit = int(input("If you wish to train the model with your own choice of parameters again, please enter 5.\n" \
    "If you wish to predict on your own data, please enter 0.\n" \
    "If you wish to exit, please enter -1.\n"))
    return digit

def predict_decision_tree(which_tree):
    age = int(input("Please enter your age (15-18): \n"))
    gender = int(input("Please enter your gender according to the following mappings:\n" \
    "0: male\n" \
    "1: female\n"))
    ethnicity = int(input("Please enter your ethnicity according to the following mappings:\n" \
    "0: Caucasian\n" \
    "1: African American\n" \
    "2: Asian\n" \
    "3. Other\n"))
    parental_education = int(input("Please enter your parents education level according to the following mappings (Note: input the average of parents):\n" \
    "0: None\n" \
    "1: High School\n" \
    "2: Some College\n" \
    "3: Bachelors\n" \
    "4: Higher\n"))
    study_time_weekly = int(input("Please enter the amount of hours you spend studying weekly (range 0-20):\n"))
    absences = int(input("Please enter the number of ascences you have accrued for the school year (range 0-30):\n"))
    tutoring = int(input("Please enter whether you participate in tutoring according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    parental_support = int(input("Please enter the amount of parental support you receive in relation to school according to the following mappings:\n" \
    "0: None\n" \
    "1: Low\n" \
    "2: Moderate\n" \
    "3: High\n" \
    "4: Very High\n"))
    extracurricular = int(input("Please enter whether you participate in extracurricular activities outside of school according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    sports = int(input("Please enter whether you participate in sports outside of school according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    music = int(input("Please enter whether you practice music outside of school according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    volunteering = int(input("Please enter whether you participate in volunteering activities outside of school according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    gpa = int(input("Please enter your current GPA rounded to the closest whole number (2, 3, or 4):\n"))
    pursue_higher_education = int(input("Please enter whether you plan to pursue higher education according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    internet_access = int(input("Please enter whether you have internet access at home according to the following mappings:\n" \
    "0: No\n" \
    "1: Yes\n"))
    strength_familial_rel = int(input("Please enter the strength of your familial relationship according to the following mappings\n" \
    "0: Very Bad\n" \
    "1: Bad\n" \
    "2: Average\n" \
    "3: Good\n" \
    "4: Very Good\n"))
    freetime_after_school = int(input("Please enter how much freetime you have after school according to the following mappings:\n" \
    "0: None\n" \
    "1: Little\n" \
    "2: Average\n" \
    "3: Above Average\n" \
    "4: A lot\n"))
    health_status = int(input("Please enter your current health status according to the following mappings:\n" \
    "0: Very Sick\n" \
    "1: Sick\n" \
    "2: Normal\n" \
    "3: Healthy\n" \
    "4: Very Healthy\n"))

    data = np.array([age, gender, ethnicity, parental_education, study_time_weekly, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, 
                          gpa, pursue_higher_education, internet_access, strength_familial_rel, freetime_after_school, health_status]).reshape(1, -1)
    if which_tree == 0:
        dt,_,_ = train_decision_tree(5, 10)
    else:
        min_sample_split = int(input("Please enter the min_sample_split you'd like to use: "))
        max_depth = int(input("Please enter the max_depth you'd like to use: "))
        dt,_,_ = train_decision_tree(min_sample_split, max_depth)
    class_label = dt.predict(data)
    
    return class_label
    
    
def main():
    option = int(input("Welcome to Student Course Success Predictor!\n" \
    "Which model would you like to explore? First, the model will be trained and then you will have the option to predict using your own data!\n" \
    "\t 0. Decision Tree - trained to return predicted letter grade.\n" \
    "\t 1. Logistic Regression - trained to return predicted pass/fail.\n" \
    "Please enter either 0 or 1.\n"))

    while valid_option(option, [0, 1]) == False:
        option = int(input("Invalid option. Please enter 0 or 1.\n"))

    if option == 0:
        # initial training of decision tree 
        print("Training...")
        dt, X_test, y_test = train_decision_tree(5, 10)
        acc = decision_tree_accuracy(dt, X_test, y_test)

        # print training outcome, ask if user wants to train on custom params, predict with custom data, or exit
        print('Decision Tree model has been trained with the following parameters:\n' \
        'min_sample_split = 5\n' \
        'max_depth = 10\n' \
        f'These parameters have been proven to achieve the highest accuracy of {acc:.4f}.\n')
        digit = int(input("If you wish to train the model with your own choice of parameters, please enter 5. \n" \
        "If you wish to predict on your own data, please enter 0.\n" \
        "If you wish to exit, please enter -1.\n"))
        
        while digit != -1:
            while valid_option(digit, [-1, 0, 5]) == False:
                digit = int(input("Invalid. Please enter 5, 0, or -1.\n"))

            while digit == 5:
                digit = user_trained_decision_tree()

            while digit == 0:
                which_tree = int(input("If you would like to predict using the highest accuracy tree, please enter 0.\n" \
                "If you would like to use custom parameters, please enter 1.\n"))
                
                label = predict_decision_tree(which_tree)
                print(f"The model predicted a grade of {prediction_to_grade(label)}")

                digit = int(input("If you would like to predict on different data, please enter 0.\n" \
                "If you would like to train a model with custom parameters to check accuracy, please enter 5.\n" \
                "If you would like to exit, please enter -1.\n"))

main()

from TestingLogisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_logistic_regression(learning_rate=0.01, num_iter=1000):
    df = pd.read_csv("data/LogReg_formatted_final_dataset.csv")

    df["PassFail"] = df["GradeClass"].apply(lambda x: 1 if x <= 2 else 0) #convert to pass/fail
    X, y = df_to_Xy(df.drop(columns=["GradeClass"]), "PassFail")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    lr = LogisticRegression(learning_rate=learning_rate, num_iter=num_iter)
    lr.fit(X_train, y_train)
    return lr, X_test, y_test

#function to get metrics 
def logistic_regression_accuracy(lr, X_test, y_test):
    preds = lr.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    return acc, prec, rec

#user trains logistic regression with their own parameters 
def user_trained_logistic_regression():
    lr_rate = float(input("Please enter the learning rate you'd like to use (default 0.01): ") or 0.01)
    n_iter = int(input("Please enter the number of iterations you'd like to use (default 1000): ") or 1000)
    print("Training Logistic Regression model...")

    lr, X_test, y_test = train_logistic_regression(lr_rate, n_iter)
    acc, prec, rec = logistic_regression_accuracy(lr, X_test, y_test)

    print(f"Logistic Regression model metrics:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}")
    digit = int(input("If you wish to train the Logistic Regression again, enter 6.\n"                       "If you wish to predict on your own data, please enter 0.\n"                       "If you wish to exit, please enter -1.\n"))
    return digit

#user enters student data and predicts pass/fail
def predict_logistic_regression(model):
    print("Please enter your student data for prediction:")
    age = int(input("Age 15-18: "))
    gender = int(input("Gender 0=Male 1=Female: "))
    ethnicity = int(input("Ethnicity 0=Cauc 1=AfAm 2=Asian 3=Other: "))
    parental_education = int(input("Parent Edu 0-4: "))
    study_time = int(input("Study hrs/wk 0-20: "))
    absences = int(input("Absences 0-30: "))
    tutoring = int(input("Tutoring 0/1: "))
    parental_support = int(input("Parent Support 0-4: "))
    extracurricular = int(input("Extracurricular 0/1: "))
    sports = int(input("Sports 0/1: "))
    music = int(input("Music 0/1: "))
    volunteering = int(input("Volunteering 0/1: "))
    gpa = float(input("GPA 2.0-4.0: "))
    pursue_higher_ed = int(input("Pursue Higher Ed 0/1: "))
    internet_access = int(input("Internet Access 0/1: "))
    family_relation = int(input("Family Relationship 0-4: "))
    freetime = int(input("Free Time 0-4: "))
    health_status = int(input("Health 0-4: "))

    sample = np.array([[age, gender, ethnicity, parental_education, study_time, absences, tutoring,
                        parental_support, extracurricular, sports, music, volunteering, gpa,
                        pursue_higher_ed, internet_access, family_relation, freetime, health_status]])
    
    
    pred = model.predict(sample)[0]
    if pred == 1:
        print("Prediction: PASS (A/B/C)")
    else:
        print("Prediction: FAIL (D/F)")

if __name__ == "__main__":
    while True:
        print("\nChoose a model to run:")
        print("1. Train Decision Tree")
        print("2. Train Logistic Regression")
        print("0. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            next_step = user_trained_decision_tree()
            if next_step == 0:
                dt, X_test, y_test = train_decision_tree(2, 3)
                predict_decision_tree(dt)
            elif next_step == -1:
                break
        elif choice == "2":
            next_step = user_trained_logistic_regression()
            if next_step == 0:
                lr, X_test, y_test = train_logistic_regression()
                predict_logistic_regression(lr)
            elif next_step == -1:
                break
        elif choice == "0":
            break
        else:
            print("Invalid choice. Try again.")





