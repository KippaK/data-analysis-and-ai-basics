import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Ladataan data CSV tiedostosta
df = pd.read_csv('diabetes.csv')

# Jaetaan data opetus (X) ja tulos (y) dataan
# eli sarakkeet 0-7 opetusdataksi ja sarake 8 tulosdataksi
X = df.iloc[:, :-1] 
y = df.iloc[:, [-1]]

# onko null arvoja
print (f'Null values: \n{df.isnull().sum()}')

# muutetaan null nollaksi
X = X.fillna(0)

# Jaetaan data opetus- (80%) ja testi- (20%) datoihin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# skalataan data
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Luodaan malli neuroverkolle
model = Sequential()
model.add(Dense(50, input_dim=X.shape[1], activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

# Käännetään malli opettamista varten
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# binary_crossentropy
# Opetetaan malli datan avulla
history=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test,y_test))

# Visualisoidaan mallin oppiminen
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
# plt.ylim(bottom=0, top=5 * min(history.history['val_loss']))
plt.show()


# Tehdään testidatalla ennusteet
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5)


# Tutkitaan mallin metriikoita
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)

print ('cm:')
print(cm)
print(f'accuracy_score: {acc}\n')
print(f'recall_score: {rs}\n')
print(f'precision_score: {ps}\n')

# Visualivoidaan confusion matrix
tn, fp, fn, tp = cm.ravel() # ravel palauttaa litistetyn taulukon
ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, fmt='g')
ax.set_title(f'ANN (acc: {acc:.02f}, recall: {rs:.02f}, precision: {ps:.02f})')
plt.show()
