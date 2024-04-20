import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import joblib

music_data = pd.read_csv("music.csv")
x = music_data.drop(columns=["genre"])
y = music_data["genre"]

age_entry = int(input("Enter your age: "))
gender_entry = int(input("Enter your Gendner (0-female & 1-male: "))
x.columns=["age","gender"]
user_input = pd.DataFrame({'age': [age_entry], 'gender': [gender_entry]})
model = DecisionTreeClassifier()
model.fit(x,y)
prediction = model.predict(user_input)
print(*prediction)
