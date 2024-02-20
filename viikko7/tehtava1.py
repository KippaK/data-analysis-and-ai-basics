import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# lue data sekä jaa X ja y
df = pd.read_csv('startup.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, [-1]]

dummies = pd.get_dummies(df['State'], drop_first=True)

X_org = X

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)

model = LinearRegression()
model.fit(X_train, y_train)

coef = model.coef_

inter = model.intercept_

# y_pred = model.predict(X_test)
y_pred = y_scaler.inverse_transform(model.predict(X_test))

mae=mean_absolute_error(y_test, y_pred) 
r2=r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'r2:  {round(r2,4)}')
print(f'mae: {round(mae,4)}')
print(f'rmse: {round(rmse,4)}')

# Malli suoriutuu erittäin hyvin

# df_new_company = pd.read_csv('new_company.csv')
df_new_company = pd.read_csv('new_company_ct.csv')
df_new_company = ct.transform(df_new_company)
df_new_company = X_scaler.transform(df_new_company)
y_comp = y_scaler.inverse_transform(model.predict(df_new_company))

y_comp = model.predict(df_new_company)

print (f'Uuden yrityksen voitto: {y_comp[0]}, todellinen: ')
print (f'{df.iloc[0:1,-1].values[0]}')