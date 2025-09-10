import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from fixed_program import preprocess, train_model
from sklearn.metrics import accuracy_score



# --- These simulate importing from your main script --- #
def load_data():
    data = load_iris(as_frame=True)
    df = data.data.copy() 
    df['target'] = data.target
    return df

def preprocess(df):
    features = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]  
    X = features
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# --- Now the unit tests --- #
class TestFixedProgram(unittest.TestCase):

    def test_load_data_shape(self):
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        self.assertEqual(df.shape, (150, 5))

    def test_preprocess_output(self):
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        X_train, X_test, y_train, y_test = preprocess(df)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_model_accuracy(self):
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        X_train, X_test, y_train, y_test = preprocess(df)
        acc = train_model(X_train, X_test, y_train, y_test)
        self.assertGreater(acc, 0.5)

if __name__ == '__main__':
    unittest.main()
