import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# KNN typically used for classification (essentially picks the closest data point and groups it to others)

data = pd.read_csv("car.data")

# preprocessing converts data into numeric values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# zip creates tuple objects corresponding to what is given
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# k known as hyper parameter = amount of neighbors being looked for
# k should be odd number to avoid comparing tied values
# calculations using euclidean distance = simple distance formula

model = KNeighborsClassifier(n_neighbors=9)

# fit function trains the model
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # data needs to be two dimensional for kneighbors method
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
