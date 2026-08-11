import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("titanic.csv")

# ---------------------------------------------------
# Basic EDA
# ---------------------------------------------------

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

# ---------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------

df["Age"].fillna(df["Age"].median(), inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df.drop(columns=["Cabin"], inplace=True)

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# ---------------------------------------------------
# Encode Categorical Variables
# ---------------------------------------------------

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

df["Embarked"] = encoder.fit_transform(df["Embarked"])

# ---------------------------------------------------
# Select Features
# ---------------------------------------------------

X = df[
[
"Pclass",
"Sex",
"Age",
"Fare",
"FamilySize",
"Embarked"
]
]

y = df["Survived"]

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

X,
y,

test_size=0.2,

random_state=42

)

# ---------------------------------------------------
# Logistic Regression
# ---------------------------------------------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test,lr_pred)

# ---------------------------------------------------
# Random Forest
# ---------------------------------------------------

rf = RandomForestClassifier(

n_estimators=100,

random_state=42

)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test,rf_pred)

# ---------------------------------------------------
# Results
# ---------------------------------------------------

print("Logistic Regression Accuracy")

print(lr_accuracy)

print()

print(classification_report(

y_test,

lr_pred

))

print()

print("Random Forest Accuracy")

print(rf_accuracy)

print()

print(classification_report(

y_test,

rf_pred

))