import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Lue data ja jaa X ja y
df = pd.read_csv('startup.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, [-1]]

# Luo dummy-muuttujat
ct = ColumnTransformer(transformers=[('encoder', 
                                      OneHotEncoder(drop='first'), ['State'])], 
                       remainder='passthrough')
X = ct.fit_transform(X)

# Jaa data opetusdataan (80 %) ja testidataan (20 %)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Opeta malli opetusdatalla
model = LinearRegression()
model.fit(X_train, y_train)

# Tee ennusteet testidatalla
y_pred = model.predict(X_test)

# Laske metriikat (R2, MAE ja RMSE)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mea = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mea)

print(f'r2:  {round(r2, 4)}')
print(f'mae: {round(mae, 4)}')
print(f'rmse: {round(rmse, 4)}')

# Tallenna malli ja enkooderi pickle-tiedostoon
with open('startup_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    pickle.dump(ct, model_file)
