import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

digits = load_digits()

X = digits.data
y = digits.target

# ---------------------------------------------------
# Split Dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# Create Model
# ---------------------------------------------------

model = LogisticRegression(max_iter=5000)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------------------
# Training Accuracy
# ---------------------------------------------------

train_pred = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

print("Training Accuracy :", train_accuracy)

# ---------------------------------------------------
# Test Accuracy
# ---------------------------------------------------

test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print("Test Accuracy :", test_accuracy)

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_test, test_pred)

print("\nConfusion Matrix")

print(cm)

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------

print("\nClassification Report")

print(classification_report(
    y_test,
    test_pred
))

# ---------------------------------------------------
# Correct Predictions
# ---------------------------------------------------

correct = []

for i in range(len(y_test)):

    if y_test[i] == test_pred[i]:

        correct.append(i)

plt.figure(figsize=(12,5))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(
        X_test[correct[i]].reshape(8,8),
        cmap="gray"
    )

    plt.title(
        f"Pred:{test_pred[correct[i]]}"
    )

    plt.axis("off")

plt.suptitle("Correct Predictions")

plt.show()

# ---------------------------------------------------
# Incorrect Predictions
# ---------------------------------------------------

wrong = []

for i in range(len(y_test)):

    if y_test[i] != test_pred[i]:

        wrong.append(i)

plt.figure(figsize=(12,5))

for i in range(min(5,len(wrong))):

    plt.subplot(1,5,i+1)

    plt.imshow(
        X_test[wrong[i]].reshape(8,8),
        cmap="gray"
    )

    plt.title(
        f"T:{y_test[wrong[i]]}\nP:{test_pred[wrong[i]]}"
    )

    plt.axis("off")

plt.suptitle("Incorrect Predictions")

plt.show()