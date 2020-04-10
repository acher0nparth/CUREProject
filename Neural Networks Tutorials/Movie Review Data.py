import tensorflow as tf
from tensorflow import keras
import numpy as np

# loads data
data = keras.datasets.imdb

# only takes the words that are the most frequent
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

print(test_data[0])

# gives tuples of the words and integer values for word mapping
word_index = data.get_word_index()

# start at 3 to add values that check for validity
word_index = {k:(v+3) for k, v in word_index.items()}
# PAD makes all values the same length (padding) to have a known input shape and size (for input and output nodes)
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# swaps the values and the keys
reverse_word_index = dict([(value, key) for (key, value)in word_index.items()])

# turns data into a consistent form that the model can handle
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# decodes the word mapping
def decode_review(text):
    # if cant find value of "i" it will put a "?" in order not to crash
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# architecture of the model
'''model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))  # groups word vectors, defines a line in 16 dimensions
model.add(keras.layers.GlobalAveragePooling1D())  # averages value of each vector
model.add(keras.layers.Dense(16, activation="relu"))  # performs classification, passes dimensional data into 16 nodes
model.add(keras.layers.Dense(1, activation="sigmoid"))  # defines single output node (sigmoid = data between 0 and 1), classifies data based on patterns of words

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# validation data used to check how model is performing, uses different set of data to test accuracy every time
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = test_labels[:10000]
y_train = test_labels[10000:]

# batch_size = how many values loaded at once, verbose = how detailed output is (0 = nothing printed, 2 = accuracy and loss for each epoch)
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

# (binary data) h5 is the extension of a saved model in keras
model.save("model.h5")'''

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

# loads data
model = keras.models.load_model("model.h5")

# with open("text.txt", encoding="utf-8") as f:
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
#         encode = review_encode(nline)  # looks up mapping for words and returns encoded list
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict[0])


test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: ", str(predict[0]))
print("Actual: ", str(test_labels[0]))


