import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(trainX, trainy), (testX, testy) = data.load_data()

# plt.imshow(trainX[1])
# # plt.imshow(trainX[1], cmap=plt.cm.binary)
# plt.show()

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

trainX = trainX/255
testX = testX/255

'''model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(trainX, trainy, epochs=12)

model.save("mnistDigits.h5")'''
model = keras.models.load_model("mnistDigits.h5")

test_loss, test_acc = model.evaluate(testX, testy)

print("Tested Acc: ", test_acc)

prediction = model.predict(testX)

for i in range(10):
    plt.grid(False)
    plt.imshow(testX[i], cmap=plt.cm.binary)
    plt.title("Prediction: " + classes[np.argmax(prediction[i])])
    plt.show()
