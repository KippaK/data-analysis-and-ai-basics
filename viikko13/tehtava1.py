import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Lue diabetes datasetti
diabetes_df = pd.read_csv("diabetes.csv")

# Erotetaan ominaisuudet (X) ja target (y)
X = diabetes_df.drop(columns=["Outcome"])
y = diabetes_df["Outcome"]

# Jaa data training ja test setteihin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaalaa data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_pred = logistic_model.predict(X_test_scaled)

# ANN
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann_model.fit(X_train_scaled, y_train)
ann_pred = ann_model.predict(X_test_scaled)

# Tulosta metriikat
print("Logistic Regression Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_pred))
print("Accuracy Score:", accuracy_score(y_test, logistic_pred))
print("Precision Score:", precision_score(y_test, logistic_pred))
print("Recall Score:", recall_score(y_test, logistic_pred))

print("\nANN Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, ann_pred))
print("Accuracy Score:", accuracy_score(y_test, ann_pred))
print("Precision Score:", precision_score(y_test, ann_pred))
print("Recall Score:", recall_score(y_test, ann_pred))

