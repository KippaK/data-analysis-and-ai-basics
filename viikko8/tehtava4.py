import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

diabetes_data = pd.read_csv('diabetes.csv')

X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print('Confusion Matrix:')
print(conf_matrix)
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)

new_patients = pd.read_csv('diabetes-new.csv')
new_patients_scaled = scaler.transform(new_patients)

predictions_new_patients = model.predict(new_patients_scaled)

print('Ennusteet uusille potilaille:')
print(predictions_new_patients)
