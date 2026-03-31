import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder


df = pd.read_excel('decision_tree_dataset.xlsx')


input = df.drop('Play', axis='columns')
target = df['Play']

le = LabelEncoder()

for col in input.columns:
    input[col] = le.fit_transform(input[col])

model = tree.DecisionTreeClassifier()
model.fit(input, target)
mapping = {}
for col in input.columns:
    mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print(mapping)

print(model.predict([[0, 1, 0, 0]]))
print(model.predict([[1, 0, 1, 1]]))
print(model.score(input, target))
