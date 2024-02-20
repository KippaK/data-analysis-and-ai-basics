# Tarvittavat kirjastot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Vaihe 1: Lataa aineisto pandas dataframeen
data = pd.read_csv("housing.csv")

# Vaihe 2: Tutustu dataan ja visualisoi
plt.scatter(data['median_house_value'], data['median_income'] * 10000)
plt.title('Talon mediaaniarvo vs. Kotitalouden vuositulot')
plt.xlabel('Talon mediaaniarvo')
plt.ylabel('Kotitalouden vuositulot (10k $)')
plt.show()

# Vaihe 3: Jaa aineisto opetusdataan ja testidataan
X = data[['median_income']]
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vaihe 4: Opetetaan lineaarinen regressiomalli
model = LinearRegression()
model.fit(X_train, y_train)

# Vaihe 5: Tulosta suoran yhtälö
print(f"Suoran yhtälö: y = {model.coef_[0]}x + {model.intercept_}")

# Vaihe 6: Tee ennuste testidatalla
y_pred = model.predict(X_test)

# Vaihe 7: Visualisoi ennustettujen ja todellisten arvojen ero histogrammilla
plt.hist(y_test - y_pred, bins=50)
plt.title('Ennusteiden erot (Talon mediaaniarvo)')
plt.xlabel('Erot')
plt.ylabel('Lukumäärä')
plt.show()

# Vaihe 8: Arvioi mallia käyttäen metriikoita
r2_score = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(f"R^2: {r2_score}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Vaihe 9: Ennusta kuvitteellisen kotitalouden talon arvo, kun vuositulot on 30 000 dollaria
new_income = [[30000 / 10000]]  # Vuositulot jaetaan 10000:lla
predicted_house_value = model.predict(new_income)
print(f"Ennustettu talon mediaaniarvo: {predicted_house_value[0]}")

# Kiitos ChatGPT