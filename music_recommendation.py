import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.widget import Widget
from kivymd.uix.screenmanager import ScreenManager


class Musicrec(Screen):
    def recommend(self):
        age = int(self.ids.age.text)
        gender = int(self.ids.gender.text)
        music_data = pd.read_csv("music.csv")
        x = music_data.drop(columns=["genre"])
        y = music_data["genre"]
        x.columns=["age","gender"]
        user_input = pd.DataFrame({'age': [age], 'gender': [gender]})
        model = DecisionTreeClassifier()
        model.fit(x,y)
        prediction = model.predict(user_input)
        self.ids.prediction.text = f"{str(*prediction)}"

class Mu_rc(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        screen = Screen()
        mr = Musicrec()
        screen.add_widget(mr)
        return screen

if __name__ == "__main__":
    Mu_rc().run()
