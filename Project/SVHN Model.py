#pip install tensorflow==2.0
#do this inside the virtual environment we created within anaconda to get the damn thing to work
#that still might not make it work I had to do a lot of fuckery to make it work and idrk what fixed it in the end

#pulled this code from https://www.tensorflow.org/datasets/keras_example just changed the dataset to svhn_cropped from mnist

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

(ds_train, ds_test), ds_info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(60000)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(10000)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

train_images, train_labels = next(iter(ds_train))
test_images, test_labels = next(iter(ds_test))

def show_batch(train_image, train_label):
  plt.figure(figsize=(32,32))
  for n in range(5):
      plt.imshow(train_images[n])
      plt.show()
      plt.title(train_labels[n+1])
# show_batch(train_images.numpy(), train_labels.numpy())

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

'''tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])
 model.compile(
   loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'])'''

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(ds_train, epochs=6, validation_data=ds_test)'''

model.fit(train_images, train_labels, batch_size=64, epochs=6)

model.save("SVHNModel.h5")
model = keras.models.load_model("SVHNModel.h5")

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ", test_acc)

prediction = model.predict(test_images)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + classes[test_labels[i]])
    plt.title("Prediction: " + classes[np.argmax(prediction[i])])
    plt.show()

