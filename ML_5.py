import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

model = LinearRegression()

df = pd.read_excel('StudyHours.xlsx')

print(df.head())

plt.scatter(df.StudyHours, df.Marks)
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Study Hours vs Marks')
plt.show()

X = df[['StudyHours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted Marks:", y_pred)
print("Actual Marks:", y_test.values)
print("Score:", model.score(X_test, y_test))