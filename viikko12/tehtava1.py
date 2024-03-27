import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.externals import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Tehtävä 1: Lataa data ja käsittele se

# Lataa data
housing_data = pd.read_csv('housing.csv')

# Luo X ja y datasetit
X = housing_data.drop(columns=['medianHouseValue'])
y = housing_data['medianHouseValue']

# Tarkista ja korvaa mahdolliset puuttuvat arvot
X.fillna(0, inplace=True)

# Luo dummy-muuttujat oceanProximity-sarakkeesta
ocean_proximity_encoder = Pipeline([
    ('encoder', OneHotEncoder())
])
column_transformer = ColumnTransformer([
    ('ocean_proximity', ocean_proximity_encoder, ['oceanProximity'])
], remainder='passthrough')
X_encoded = column_transformer.fit_transform(X)

# Jaa data opetusdataan ja testidataan
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Skaalaa data StandardScalerin avulla
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Opetus usean muuttujan lineaarinen regressio
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Tee ennusteet testidatalla lineaarisella mallilla
linear_predictions = linear_reg.predict(X_test_scaled)

# Laske metriikat (R2, MAE ja RMSE)
linear_r2 = r2_score(y_test, linear_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = mean_squared_error(y_test, linear_predictions, squared=False)

print("Linear Regression Metrics:")
print("R2 Score:", linear_r2)
print("Mean Absolute Error:", linear_mae)
print("Root Mean Squared Error:", linear_rmse)

# Rakenna ja opeta ANN-malli
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
ann_model.compile(optimizer='adam', loss='mean_squared_error')
history = ann_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=64)

# Tee ennusteet testidatalla ANN-mallilla
ann_predictions = ann_model.predict(X_test_scaled)

# Laske metriikat ANN-mallille
ann_r2 = r2_score(y_test, ann_predictions)
ann_mae = mean_absolute_error(y_test, ann_predictions)
ann_rmse = mean_squared_error(y_test, ann_predictions, squared=False)

print("\nANN Regression Metrics:")
print("R2 Score:", ann_r2)
print("Mean Absolute Error:", ann_mae)
print("Root Mean Squared Error:", ann_rmse)

# Piirrä oppimiskäyrä
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Tallenna malli, dummy-enkooderi ja skaalain levylle
joblib.dump(linear_reg, 'linear_regression_model.pkl')
ann_model.save('ann_regression_model.h5')
joblib.dump(column_transformer, 'dummy_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Tehtävä 2: Lataa data ja käsittele se
# (Toteutetaan vasta kun ensimmäinen tehtävä on valmis)
# Lataa data ja käsittele se vastaavasti kuin ensimmäisessä tehtävässä
# Luo malli ja laske metriikat

# Vertaa tuloksia päätöspuulla ja random forestilla
# (Toteutetaan vasta kun ensimmäinen tehtävä on valmis)
# Käytä scikit-learnin DecisionTreeRegressoria ja RandomForestRegressoria
# Luo mallit, laske metriikat ja vertaa tuloksia

# Raportoi tulokset ja pohdi miksi jokin malli suoriutui paremmin kuin toinen

# Huom: Tämä koodipohja vaatii tiedostojen housing.csv ja new_house.csv olemassaolon samassa kansiossa.
# Tarvittavat kirjastot pitää myös varmistaa asennettuna.
