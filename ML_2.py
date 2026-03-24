import numpy as np 
import pandas as pd
from sklearn import linear_model

df = pd.read_excel('Linear_regression.xlsx')

reg = linear_model.LinearRegression()

x = df[['Study Hours', 'Sleep Hours', 'Practice Papers']]
y = df[['Marks']]

reg.fit(x , y)
print('Coefficient: ', reg.coef_)
print('Intercept: ', reg.intercept_)
predicted_marks = reg.predict(pd.DataFrame([[8, 8, 2]], columns=['Study Hours', 'Sleep Hours', 'Practice Papers']))
print('Predicted Marks for 8 Study Hours, 8 Sleep Hours, and 2 Practice Papers: ', predicted_marks)

