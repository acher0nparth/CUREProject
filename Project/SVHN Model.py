#pip install tensorflow==2.0
#do this inside the virtual environment we created within anaconda to get the damn thing to work
#that still might not make it work I had to do a lot of fuckery to make it work and idrk what fixed it in the end

#pulled this code from https://www.tensorflow.org/datasets/keras_example just changed the dataset to svhn_cropped from mnist

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

image_batch, label_batch = next(iter(ds_train))
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(32,32))
  for n in range(5):
      plt.imshow(image_batch[n])
      plt.show()
      plt.title(label_batch[n+1])
show_batch(image_batch.numpy(), label_batch.numpy())

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)


'''model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10)

# model.save("fashionModel.h5")
# model = keras.models.load_model("fashionModel.h5")

test_loss, test_acc = model.evaluate(ds_test, test_labels)

print("Tested Acc: ", test_acc)

prediction = model.predict(test_images)

for i in range(25):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()'''
