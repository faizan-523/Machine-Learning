import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

df = pd.read_excel('data.xlsx')

x = df[['Study Hours']]
y = df[['Marks']]

plt.scatter(x , y, color='blue', marker='o')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Linear Regression')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(x , y)

print('Coefficient: ', reg.coef_)
print('Intercept: ', reg.intercept_)

predicted_marks = reg.predict([[12]])
print('Predicted Marks for 12 Study Hours: ', predicted_marks)

plt.scatter(x , y, color='blue', marker='o')
plt.plot(x, reg.predict(x), color='red')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Linear Regression with Line of Best Fit')
plt.show()

