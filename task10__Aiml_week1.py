import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ----------------------------------
# Load Dataset
# ----------------------------------

df = pd.read_csv("heart.csv")

# ----------------------------------
# Basic Information
# ----------------------------------

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

# ----------------------------------
# Data Visualization
# ----------------------------------

df["target"].value_counts().plot(kind="bar")

plt.title("Heart Disease Count")

plt.show()

df["age"].hist()

plt.title("Age Distribution")

plt.show()

df["chol"].hist()

plt.title("Cholesterol Distribution")

plt.show()

# ----------------------------------
# Feature Selection
# ----------------------------------

X = df.drop("target", axis=1)

y = df["target"]

# ----------------------------------
# Train Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(

X,
y,

test_size=0.2,

random_state=42

)

# ----------------------------------
# Logistic Regression
# ----------------------------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# ----------------------------------
# Random Forest
# ----------------------------------

rf = RandomForestClassifier(

n_estimators=100,

random_state=42

)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# ----------------------------------
# Logistic Regression Results
# ----------------------------------

print("Logistic Regression Accuracy")

print(accuracy_score(y_test, lr_pred))

print(confusion_matrix(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

# ----------------------------------
# Random Forest Results
# ----------------------------------

print("Random Forest Accuracy")

print(accuracy_score(y_test, rf_pred))

print(confusion_matrix(y_test, rf_pred))

print(classification_report(y_test, rf_pred))