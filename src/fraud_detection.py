import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data["Class"].value_counts())

# Graph 1
sns.countplot(x="Class", data=data)
plt.title("Fraud vs Genuine Transactions")
plt.show()   # Close this graph window to continue

# Features and Target
X = data.drop("Class", axis=1)
y = data["Class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

print("Model trained!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Graph 2
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()
import joblib

joblib.dump(model, "fraud_model.pkl")

print("Model saved successfully!")