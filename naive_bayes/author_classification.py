"""
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from time import time

from email_preprocess import preprocess
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from gaussian_naive_bayes import GaussianNB

features_train, features_test, labels_train, labels_test = preprocess()


clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print('Training time:', round(time() - t0, 3))

t0 = time()
labels_predicted = clf.predict(features_test)
score = accuracy_score(labels_test, labels_predicted)
print('Testing time:', round(time() - t0, 3))

print(score)
