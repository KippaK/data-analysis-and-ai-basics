import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

# Lue iris datasetti
iris_df = pd.read_csv("iris.csv")

# Erotetaan ominaisuudet (X) ja target (y)
X = iris_df.drop(columns=["Species", "Class"])
y = iris_df["Class"]

# Kategorisoi target-muuttuja
y = to_categorical(y)

# Jaa data training ja test setteihin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaalaa data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rakenna ja opeta ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=4, verbose=1)

# Ennusta testidatalla
y_pred = model.predict(X_test_scaled)

# Tulosta confusion matrix
y_pred_classes = y_pred.argmax(axis=-1)
y_test_classes = y_test.argmax(axis=-1)
print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))


# Datan lukeminen
iris_data = pd.read_csv("iris.csv")

# Poistetaan tarpeettomat sarakkeet ja jaetaan ominaisuudet ja luokka
X = iris_data.drop(columns=['Species', 'Class'])
y = iris_data['Species']

# Datan jakaminen koulutus- ja testisetteihin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest -mallin luominen ja opettaminen
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Ennustaminen testidatalla
y_pred_rf = rf_classifier.predict(X_test)

# Confusion Matrix ja tarkkuus
cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Random Forest Confusion Matrix:")
print(cm_rf)
print("Accuracy Score (Random Forest):", accuracy_rf)
