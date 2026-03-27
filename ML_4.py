import pandas as pd
from sklearn.linear_model import LinearRegression

model = LinearRegression()

df = pd.read_excel('data.xlsx')

d = pd.get_dummies(df.Names, dtype='int')
print(d)

merged = pd.concat([df, d], axis='columns')
print(merged)

final = merged.drop(['Names'], axis='columns')
print(final)

x = final.drop(['Marks'], axis='columns')
y = final.Marks

model.fit(x, y)
pred = model.predict([[1,7,2,8,0,0,0,0,0,0,0,1,0,0]])
print("Predicted Marks:", pred)

print(model.score(x, y))