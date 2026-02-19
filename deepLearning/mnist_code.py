import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train,X_test = X_train/255.0,X_test/255.0

model = keras.Sequential([keras.layers.Dense(128,activation='relu',input_shape=(784,)),
                          keras.layers.Dropout(0.2),
                          keras.layers.Dense(64,activation='relu'),
                          keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


model_his = model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)

model.save('mnist_model.keras')

loss,accuracy = model.evaluate(X_test,y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy:.3f}")
