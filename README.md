# 🎓 Student Success Predictor  
**By:** Johnathan Garcia, Devina Tikkoo, and Hithisha Dubbaka  
**Demo Link:** https://www.youtube.com/watch?v=wzqMbxBsu0s  

---

## 🚀 Motivation  
Educational institutions tend to rely solely on past academic performance, in the form of standardized tests, GPA, final grades, etc., for future academic performance. While this tends to be a reliable indicator, it is reactive.  

That is, there is no way to determine whether a student needs extra help until they have already demonstrated poor performance.  

**What if there was a way to use data regarding the student's lifestyle and habits to predict whether the student is on pace for success or failure in a course?**

---

## 📖 Description  
**Student Success Predictor** is a way to predict the final letter grade a student is on pace to earn, and more generally whether they are on pace to pass or fail.  

We have implemented two basic machine learning algorithms — **decision tree** and **logistic regression** — **from scratch** (WITHOUT the use of external libraries).  

We trained the models on **over 200,000 datapoints** made up of student data.  

---

## 📂 File Tour  

### **data**  
Contains the initial, intermediate, and final datasets.  

### **src**  
Contains the source files for the decision tree and logistic regression models.  

### **data_exploration_and_analysis**  
Contains the Jupyter notebooks in which the datasets were explored and manipulated to be model-ready.  

### **No folder**  
- **main.py** : file that runs CLI.  
- **TestingDecisionTree.py** : ensures decision tree is working as expected and finds the optimal hyper-parameters.  
- **TestingLogisticRegression.py** : ensures logistic regression is working as expected and finds the optimal hyper-parameters.  

---

## 🛠️ How to Run  

There are very few dependencies to run this program and start predicting!  

- **numpy**  
- **pandas**  
- **matplotlib**  
- **scikit-learn**  

1. Ensure the dependencies listed above are downloaded on your device. Alternatively, you could create a virtual environment with the libraries downloaded within the environment as to not mess with any configurations you may have on your system currently.  
2. Run **main.py** and interact with the command line interface (CLI) to train the models with default and custom parameters, and predict on your own custom data.  
