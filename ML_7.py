import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

digits = load_digits()

print(digits.data[0])

plt.matshow(digits.images[0])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

model.fit(X_train, y_train)
pr = model.predict([digits.data[23]])

print("Predicted value: ", pr)
print("Score: ", model.score(X_test, y_test))
