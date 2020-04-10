import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import eli5
from eli5.sklearn import PermutationImportance


'''SVM = support vector machine  
SVC is within SVM = support vector classification = classifier
creates a hyper plane - divides data linearly
determines closest points to line (different classes) and equates distances
best plane has the largest generated distance to maximize margin and make most accurate predictions
kernel function takes inputs and outputs a higher dimension to divide classes
kernel function is repeated until a hyper plane can be generated
soft margin allows for outlier points to exist to get a better classifier
hard margin doesnt allow for cross over of classes'''

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

classes = ['malignant', 'benign']

# various parameters available for SVC
# C is for margin (0 = hard, 1 = soft, 2 = softer, etc)
clf = svm.SVC(kernel="linear", C=3)
# clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# accuracy_score compares two lists for similarity (order doesnt matter)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

for x in range(len(x_test)):
    print("Actual: ", classes[y_test[x]], "\tPredicted: ", classes[y_pred[x]])


