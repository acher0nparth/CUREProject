import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# linear regression used for when there is a linear trend

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# based on features in student data we want to deteremine the value of G3
predict = "G3"

# training attributes used to predict another value (G3)
X = np.array(data.drop([predict], 1)) # features
y = np.array(data[predict]) # label = attribute you are trying to predict

# splits data into 4 different arrays (test values test accuracy)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_) # the bigger the coefficient the more weight an attribute has
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'failures'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()