import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('titanic-class-age-gender-survived.csv')

X = df.loc[:, ['Age', 'Gender', 'PClass']]
y = df.loc[:,['Survived']]


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),
                                      ['Gender','PClass'])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.fit_transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_pros = model.predict_proba(X_test)


cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
pc = precision_score(y_test, y_pred)
rc = recall_score(y_test, y_pred)

print (f'cm: \n{cm}')
print (f'acc: {acc}')
print (f'pc: {pc}')
print (f'rc: {rc}')

tn, fp, fn, tp = cm.ravel()
ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, fmt='g', cbar=False)
ax.set_title(f'LogReg (acc: {acc:.02f}, recall: {rc:.02f}, precision: {pc:.02f}')
plt.show()

Xnew = pd.read_csv('titanic-new.csv')
# Xnew = Xnew.loc[:, ['Age', 'Gender']]
Xnew = ct.transform(Xnew)

y_pred_new = model.predict(Xnew)
y_pred_new_pros = model.predict_proba(Xnew)
print(y_pred_new_pros)