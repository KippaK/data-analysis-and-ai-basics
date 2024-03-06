import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

X = df.iloc[:, 0:4]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

dt_model = DecisionTreeClassifier(max_depth=4)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

cm_dt = confusion_matrix(y_test, y_pred_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)
cr_dt = classification_report(y_test, y_pred_dt)

print("Decision Tree Classifier:")
print(f"Accuracy: {acc_dt:.2f}")
print(f"Classification Report:\n{cr_dt}")

rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
cr_rf = classification_report(y_test, y_pred_rf)

print("\nRandom Forest Classifier:")
print(f"Accuracy: {acc_rf:.2f}")
print(f"Classification Report:\n{cr_rf}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_dt, annot=True, fmt='g', cbar=False, ax=axes[0])
axes[0].set_title(f'DT, acc: {acc_dt:.2f}')

sns.heatmap(cm_rf, annot=True, fmt='g', cbar=False, ax=axes[1])
axes[1].set_title(f'RF, acc: {acc_rf:.2f}')

plt.show()
