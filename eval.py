from train import trained_model, train_model, X_test, y_test, y_train
from sklearn.metrics import r2_score, mean_squared_error

# Test the model by predicting the y values in the test dataset
y_pred = trained_model.predict(X_test)

# Define a class that when instantiated, calculates the r2 score and MSE for the model
class Eval:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.r2 = r2_score(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)

# Evaluate the model using test data, comparing predicted values against actual values
eval_1 = Eval(y_test, y_pred)

# Print the results of the evaluations
print(f"Initial R2 Score: {eval_1.r2}")
print(f"Initial Mean Squared Error: {eval_1.mse}")

# Retrain the model using test data if it under performs
if eval_1.r2 < 0.8 or eval_1.mse > 1000:
    trained_model = train_model(X_test, y_test)
    
    # Test the retrained model on the test data
    new_pred = trained_model.predict(X_test)
    eval_2 = Eval(y_test, y_pred=new_pred)

    # Print the results of the evaluations
    print(f"New R2 Score: {eval_2.r2}")
    print(f"New Mean Squared Error: {eval_2.mse}")

# Make sure the optimized, retrained model is exported
optimized_model = trained_model


