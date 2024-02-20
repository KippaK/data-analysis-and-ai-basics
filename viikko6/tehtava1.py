import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Vaihe 1: Lataa aineisto pandas dataframeen
data = pd.read_csv("salary.csv")

# Vaihe 2: Tutustu dataan ja visualisoi
plt.scatter(data['YearsExperience'], data['Salary'])
plt.title('Palkka vs. Työkokemus')
plt.xlabel('Työkokemus (vuodet)')
plt.ylabel('Palkka')
plt.show()

# Korrelaatio ja p-arvo
correlation = data['YearsExperience'].corr(data['Salary'])
print(f"Korrelaatio: {correlation}")

# Heatmap korrelaatioista
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Vaihe 3: Jaa aineisto opetusdataan ja testidataan
X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vaihe 4: Opetetaan lineaarinen regressiomalli
model = LinearRegression()
model.fit(X_train, y_train)

# Vaihe 5: Tulosta suoran yhtälö
print(f"Suoran yhtälö: y = {model.coef_[0]}x + {model.intercept_}")

# Vaihe 6: Tee ennuste testidatalla
y_pred = model.predict(X_test)

# Vaihe 7: Visualisoi testiaineiston tulokset
plt.scatter(X_test, y_test, label='Todelliset palkat')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Ennusteet')
plt.title('Ennusteet vs. Todelliset palkat (Testidata)')
plt.xlabel('Työkokemus (vuodet)')
plt.ylabel('Palkka')
plt.legend()
plt.show()

# Vaihe 8: Luo seabornin regplot-visualisointi
sns.regplot(x=X_test['YearsExperience'], y=y_test, scatter_kws={'s': 20}, line_kws={'color': 'red'})
plt.title('Seaborn regplot')
plt.xlabel('Työkokemus (vuodet)')
plt.ylabel('Palkka')
plt.show()

# Vaihe 9: Arvioi mallia käyttäen metriikoita
r2_score = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(f"R^2: {r2_score}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Vaihe 10: Ennusta kuvitteellisen työntekijän palkka 7 vuoden kokemuksella
new_experience = [[7]]
predicted_salary = model.predict(new_experience)
print(f"Ennustettu palkka 7 vuoden kokemuksella: {predicted_salary[0]}")


# Kiitos ChatGPT