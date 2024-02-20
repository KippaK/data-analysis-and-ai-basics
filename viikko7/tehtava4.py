import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

# Read data
df = pd.read_csv('housing.csv')

# Select necessary columns
selected_columns = ['longitude', 'latitude', 'housing_median_age', 
                    'total_rooms', 'total_bedrooms', 'median_income', 'ocean_proximity']
X = df[selected_columns]
y = df['median_house_value']

# Create dummy variables
ct = ColumnTransformer(transformers=[('encoder', 
                                      OneHotEncoder(drop='first'), ['ocean_proximity'])], 
                       remainder='passthrough')
X = ct.fit_transform(X)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate metrics (R2, MAE, and RMSE)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mea = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mea)

print(f'r2:  {round(r2, 4)}')
print(f'mae: {round(mae, 4)}')
print(f'rmse: {round(rmse, 4)}')

# Save the model and encoder to a pickle file
with open('housing_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    pickle.dump(ct, model_file)
