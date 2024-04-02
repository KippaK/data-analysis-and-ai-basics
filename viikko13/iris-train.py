import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import pickle #save encoder
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical

#read data
df = pd.read_csv('iris.csv')

# Divide X and y
X = df.iloc[:, 0:4]
y = df.iloc[:, [4]]
y = to_categorical(y)

# Train and test data 80 % - 20 % 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Scale X
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Build and train ANN
model = Sequential()
model.add(Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')) #  input layer
model.add(Dense(25, activation='relu', kernel_initializer='normal')) # hidden layer
model.add(Dense(3, activation='softmax')) # 3 size output layer
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32,  verbose=1, validation_data=(X_test,y_test))
    
# visualize training
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# visualize training
print(history.history.keys())
# "Loss"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'\nTest results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

# Predict with test data
y_pred_proba = model.predict(X_test) 
y_pred = y_pred_proba.argmax(axis=1)
y_test = y_test.argmax(axis=1)
# Confusion Matrix and metrics
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='g')
plt.show()



