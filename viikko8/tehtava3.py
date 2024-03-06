import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('titanic-class-age-gender-survived.csv')
data_with_dummies = pd.get_dummies(data, columns=['Gender'], drop_first=True)

data_with_dummies_pclass = pd.get_dummies(data_with_dummies, columns=['PClass'], drop_first=True)

X_final = data_with_dummies_pclass[['Age', 'Gender_male', 'PClass_2nd', 'PClass_3rd']]
y_final = data_with_dummies_pclass['Survived']

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_final)
X_test_scaled_final = scaler_final.transform(X_test_final)

model_final = LogisticRegression()
model_final.fit(X_train_scaled_final, y_train_final)

y_pred_final = model_final.predict(X_test_scaled_final)

new_passengers_final = pd.DataFrame({'Age': [17, 17], 'Gender_male': [0, 1], 'PClass_2nd': [0, 0], 'PClass_3rd': [0, 1]})
new_passengers_scaled_final = scaler_final.transform(new_passengers_final)

predictions_new_passengers_final = model_final.predict(new_passengers_scaled_final)

conf_matrix_final = confusion_matrix(y_test_final, y_pred_final)
accuracy_final = accuracy_score(y_test_final, y_pred_final)
recall_final = recall_score(y_test_final, y_pred_final)
precision_final = precision_score(y_test_final, y_pred_final)

print('Confusion Matrix (with Gender and PClass):')
print(conf_matrix_final)
print('Accuracy (with Gender and PClass):', accuracy_final)
print('Recall (with Gender and PClass):', recall_final)
print('Precision (with Gender and PClass):', precision_final)