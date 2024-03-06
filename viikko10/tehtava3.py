import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz, DecisionTreeClassifier

df = pd.read_csv("titanic.csv")

df = pd.get_dummies(df, columns=['PClass', 'Gender'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']
columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

mfi = model.feature_importances_

y_pred = model.predict(X_test)
y_pred_pros = model.predict_log_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cr)

ax = plt.axes()
sns.heatmap(cm, annot=True, fmt='g', cbar=False)
ax.set_title(f'DT, acc:{acc:.02f}')
plt.show()

dot_data = export_graphviz(
    model,
    out_file = None,
    feature_names= columns,
    class_names= ['Survived', 'Dead'],
    filled=True,
    rounded=True)

graph = graphviz.Source(dot_data)

graph.render(filename='titanic', format='png') 

df_new = pd.read_csv('new-titanic.csv')
df_new = pd.get_dummies(df_new, columns=['PClass', 'Gender'], drop_first=True)

X_new = df_new.iloc[:, 0:3]

y_pred_new = model.predict(X_new)

df_new['Predicted'] = y_pred_new

print(df_new)
