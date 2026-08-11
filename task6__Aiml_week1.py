import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(
    "sms.tsv",
    sep="\t",
    header=None,
    names=["Label","Message"]
)

print(df.head())

# ---------------------------------------------------
# Encode Labels
# ---------------------------------------------------

df["Label"] = df["Label"].map({
    "ham":0,
    "spam":1
})

# ---------------------------------------------------
# Features and Target
# ---------------------------------------------------

X = df["Message"]

y = df["Label"]

# ---------------------------------------------------
# TF-IDF
# ---------------------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train,X_test,y_train,y_test = train_test_split(

X,
y,

test_size=0.2,

random_state=42

)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

model = MultinomialNB()

model.fit(X_train,y_train)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

print("Accuracy")

print(accuracy_score(y_test,y_pred))

print()

print("Confusion Matrix")

print(confusion_matrix(y_test,y_pred))

print()

print("Classification Report")

print(classification_report(y_test,y_pred))