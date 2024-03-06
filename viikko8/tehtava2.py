import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import StandardScaler

data = pd.read_csv('titanic-class-age-gender-survived.csv')

data_with_dummies = pd.get_dummies(data, columns=['Gender'], drop_first=True)

X_new = data_with_dummies[['Age', 'Gender_male']]
y_new = data_with_dummies['Survived']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

scaler_new = StandardScaler()
X_train_scaled_new = scaler_new.fit_transform(X_train_new)
X_test_scaled_new = scaler_new.transform(X_test_new)

model_new = LogisticRegression()
model_new.fit(X_train_scaled_new, y_train_new)

y_pred_new = model_new.predict(X_test_scaled_new)

new_passengers_new = pd.DataFrame({'Age': [17, 17], 'Gender_male': [0, 1]})
new_passengers_scaled_new = scaler_new.transform(new_passengers_new)

predictions_new_passengers_new = model_new.predict(new_passengers_scaled_new)

conf_matrix_new = confusion_matrix(y_test_new, y_pred_new)
accuracy_new = accuracy_score(y_test_new, y_pred_new)
recall_new = recall_score(y_test_new, y_pred_new)
precision_new = precision_score(y_test_new, y_pred_new)

print('Confusion Matrix (with Gender):')
print(conf_matrix_new)
print('Accuracy (with Gender):', accuracy_new)
print('Recall (with Gender):', recall_new)
print('Precision (with Gender):', precision_new)