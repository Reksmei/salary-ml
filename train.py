import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data():
    csv_path = pathlib.Path(__file__).parent / "Salary_Data.csv"
    df = pd.read_csv(csv_path)
    X = df[["years_experience"]]
    y = df["salary"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Model trained successfully. Coefficients: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")
    return model

# Split the data in training and testing
X_train, X_test, y_train, y_test = load_data()

# Train the model using the training data
trained_model = train_model(X_train, y_train)


    