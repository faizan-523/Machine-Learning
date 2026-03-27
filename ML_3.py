import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

df = pd.read_excel('data.xlsx')

x = df[['Study Hours']]
y = df[['Marks']]

reg = linear_model.LinearRegression()
reg.fit(x , y)

print('Coefficient: ', reg.coef_)
print('Intercept: ', reg.intercept_)

predicted_marks = reg.predict(pd.DataFrame([[12]], columns=['Study Hours']))
print('Predicted Marks for 12 Study Hours: ', predicted_marks)

with open('model.pkl', 'wb') as file:
    pickle.dump(reg, file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

pm = model.predict(pd.DataFrame([[12]], columns=['Study Hours']))
print('Predicted Marks for 12 Study Hours using loaded model: ', pm)

joblib.dump(reg, 'model_joblib.pkl')
model_joblib = joblib.load('model_joblib.pkl')
result = model_joblib.predict(pd.DataFrame([[12]], columns=['Study Hours']))
print('Predicted Marks for 12 Study Hours using loaded model with joblib: ', result)