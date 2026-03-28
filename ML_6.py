import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_excel('Logistic_Data.xlsx')
print(df.head())

plt.scatter(df['StudyHours'], df['Pass'])
plt.xlabel('Hours Studied')
plt.ylabel('Pass (1) / Fail (0)')
plt.title('Hours Studied vs Pass/Fail')
plt.show()

X = df[['StudyHours']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted values:", y_pred)
print("Actual values:", y_test.values)
print("Score:", model.score(X_test, y_test))

prob = model.predict_proba(X_test)
print("Predicted probabilities:", prob)
