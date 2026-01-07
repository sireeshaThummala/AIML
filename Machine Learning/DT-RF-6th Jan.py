# Decision Tree and Random Forest Implementation as supervised  Learning Models
# Its like az tree structure where each node represents a feature ( or attribute),
# each branch represents a decision rule, and each leaf node represents an outcome ( or target value/ class label).
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow as mlflow
import mlflow.sklearn

# Load dataset
iris = load_iris()

#split data into features and target
X = iris.data # features / column names are in iris.feature_names
y = iris.target # target labels / classes are in iris.target_names

# convert into data frame
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y
print(df.shape)
# print("Iris Dataset Head:\n", df.head())

# Split the dataset into training and testing sets (70% train, 30% test) but standard is 80-20.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# define and train Decision Tree Classifier
print("\n Decision Tree Classifier:")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train) # train the model
dt_pred = dt.predict(X_test) # make predictions


# evaluate the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)
print("Classification Report:\n", classification_report(y_test, dt_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree")
plt.show()


# Working with Random Forest Classifier
print("\n Random Forest Classifier:")
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train) # train the model
rf_pred = rf.predict(X_test) # make predictions

# evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:, rf_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, rf_pred))           

# Visualize one of the trees in the Random Forest

# Now integrating mlflow for experiment tracking
mlflow.set_experiment("Decision Tree and Random Forest Classifier Experiment") #1st creating set experiment

# Log Decision Tree model 
with mlflow.start_run(run_name="DT_RF_Classifier_Run"):   #2nd starting the run
    mlflow.log_param('model_type', 'Decision Tree')
    mlflow.log_param('max_depth', 3)
    mlflow.log_metric("dt_accuracy", dt_accuracy)
    mlflow.sklearn.log_model(dt, "decision_tree_model")

    #configuring auto logging
    mlflow.sklearn.autolog() # we enable auto logging for sklearn models instead of writing all the log statements manually .


