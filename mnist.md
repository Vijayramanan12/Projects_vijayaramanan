# Architecture

- 128->64 (hidden layer)
- dropout (20%)
- 10 (output layer)

# optimizer -> sparse_catogorical_crossentrop

- output of this optimizer.
Epoch 1/10
- accuracy: 0.9069 - loss: 0.3159 - val_accuracy: 0.9549 - val_loss: 0.1500
Epoch 2/10
- accuracy: 0.9550 - loss: 0.1491 - val_accuracy: 0.9689 - val_loss: 0.1091
Epoch 9/10
- accuracy: 0.9826 - loss: 0.0533 - val_accuracy: 0.9766 - val_loss: 0.0856
Epoch 10/10
- accuracy: 0.9830 - loss: 0.0499 - val_accuracy: 0.9771 - val_loss: 0.0882

Test loss: 0.07555829733610153
Test accuracy: 0.979

# optimizer -> SGD - learning_rate = 0.01

Epoch 1/10
- accuracy: 0.7637 - loss: 0.8078 - val_accuracy: 0.9021 - val_loss: 0.3514
Epoch 2/10
- accuracy: 0.8878 - loss: 0.3844 - val_accuracy: 0.9201 - val_loss: 0.2765
Epoch 9/10
- accuracy: 0.9494 - loss: 0.1715 - val_accuracy: 0.9592 - val_loss: 0.1399
Epoch 10/10
- accuracy: 0.9534 - loss: 0.1619 - val_accuracy: 0.9594 - val_loss: 0.1350

Test loss: 0.13156670331954956
Test accuracy: 0.961