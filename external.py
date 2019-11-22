from parse import image_parse, label_parse
from sklearn import datasets, neighbors, linear_model, metrics
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import matplotlib.pyplot as plt

X_test = image_parse("t10k-images-idx3-ubyte").reshape(10000, 784)
X_test = (255 - X_test) / 255.0
y_test = label_parse("t10k-labels-idx1-ubyte")

knn = load("knn.joblib")

#score = knn.score(X_test, y_test)
#print(score)


ax = []
fig = plt.figure(figsize=(20,20))
for num in range(100):
    img = X_test[num].reshape(28, 28)
    ax.append(fig.add_subplot(10, 10, num + 1))
    ax[-1].set_title(str(knn.predict(X_test[num].reshape(1, -1))) + " " + str(y_test[num]))
    ax[-1].axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
fig.tight_layout()
plt.axis('off')
plt.show()
