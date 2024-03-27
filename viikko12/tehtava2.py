# Tehtävä 2: Lataa data ja käsittele se
flights_data = pd.read_csv('flights.csv')

# Luo X ja y datasetit
X_flights = flights_data.drop(columns=['Price'])
y_flights = flights_data['Price']

# Tarkista ja korvaa mahdolliset puuttuvat arvot
X_flights.fillna(0, inplace=True)

# Luo dummy-muuttujat kategorisille ominaisuuksille tarvittaessa
# Tässä käytetään esimerkkinä OneHotEncoderia vain kategorisille ominaisuuksille
# Voit tarvittaessa käyttää muita tekniikoita tai mukautettuja ratkaisuja
categorical_features = ['Airline', 'Flight', 'Source City', 'Departure Time', 'Stops', 'Arrival Time', 'Destination City', 'Class']
ohe = OneHotEncoder()
X_flights_encoded = ohe.fit_transform(X_flights[categorical_features])

# Yhdistä kategoriset muuttujat ja numeeriset muuttujat
X_flights_final = pd.concat([X_flights.drop(columns=categorical_features), pd.DataFrame(X_flights_encoded.toarray())], axis=1)

# Jaa data opetusdataan ja testidataan
X_flights_train, X_flights_test, y_flights_train, y_flights_test = train_test_split(X_flights_final, y_flights, test_size=0.2, random_state=42)

# Skaalaa data tarvittaessa
# Riippuen käytetyistä algoritmeista ja siitä, tarvitaanko skaalausta, voi olla tarpeen soveltaa skaalausta

# Luo malli ja opeta se opetusdatalla
# Esimerkkinä päätöspuu
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor()
dt_model.fit(X_flights_train, y_flights_train)

# Tee ennusteet testidatalla
dt_predictions = dt_model.predict(X_flights_test)

# Laske metriikat päätöspuulle
dt_r2 = r2_score(y_flights_test, dt_predictions)
dt_mae = mean_absolute_error(y_flights_test, dt_predictions)
dt_rmse = mean_squared_error(y_flights_test, dt_predictions, squared=False)

print("Decision Tree Metrics:")
print("R2 Score:", dt_r2)
print("Mean Absolute Error:", dt_mae)
print("Root Mean Squared Error:", dt_rmse)

# Luo ja opeta Random Forest -malli
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_flights_train, y_flights_train)

# Tee ennusteet testidatalla
rf_predictions = rf_model.predict(X_flights_test)

# Laske metriikat Random Forestille
rf_r2 = r2_score(y_flights_test, rf_predictions)
rf_mae = mean_absolute_error(y_flights_test, rf_predictions)
rf_rmse = mean_squared_error(y_flights_test, rf_predictions, squared=False)

print("\nRandom Forest Metrics:")
print("R2 Score:", rf_r2)
print("Mean Absolute Error:", rf_mae)
print("Root Mean Squared Error:", rf_rmse)

# Raportoi tulokset ja pohdi miksi jokin malli suoriutui paremmin kuin toinen
# Voit verrata päätöspuun ja Random Forestin suorituskykyä ja pohjata vertailua esimerkiksi niiden monimutkaisuuteen,
# datan rakenteeseen ja parametreihin.
