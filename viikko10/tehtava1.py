import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz, DecisionTreeClassifier

df = pd.read_csv("iris.csv")

sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='Species', data=df)
plt.show()

sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='Species', data=df)
plt.show()

X = df.iloc[:, 0:4]
y = df.iloc[:, [4]]
columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = DecisionTreeClassifier(max_depth=4)
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
ax.set_title(f'DT, acc;{acc:.02f}')
plt.show()

dot_data = export_graphviz(
    model,
    out_file = None,
    feature_names= columns,
    class_names= df['Class'].unique(),
    filled=True,
    rounded=True)

graph = graphviz.Source(dot_data)

graph.render(filename='iris', format='png') 

df_new = pd.read_csv('new-iris.csv')

X_new = df_new.iloc[:, 0:4]

y_pred_new = model.predict(X_new)

df_new['Predicted_Species'] = y_pred_new

print(df_new)


