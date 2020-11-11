import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
# clf = svm.SVC()
clf = svm.SVC(gamma=0.01, C=100)
X,y = digits.data[:-1], digits.target[:-1]
clf.fit(X,y)
print(clf.predict(digits.data[[-7]]))
plt.imshow(digits.images[-7], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
