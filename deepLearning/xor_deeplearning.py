import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
epochs = 40
# model = keras.Sequential([
#     keras.layers.Dense(16, activation='relu', input_shape=(2,)),
#     keras.layers.Dense(14, activation='relu'),
#     keras.layers.Dense(12, activation='relu'),
#     keras.layers.Dense(1,activation='sigmoid')])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model = load_model('xor_model.keras')
# t_model = model.fit(X, y, epochs=epochs,verbose=0)
# model.save('xor_model.keras')

print('Training progress:')
print('='*20)

# for epoch in range(epochs):
#     print(f'Epoch: {epoch+1}/{epochs}   | Loss: {t_model.history["loss"][epoch]:.4f}')

y_pred = model.predict(np.array([[1,1]]),verbose=0)
print(1 if y_pred[0][0] > 0.5 else 0)