# Salary Predictor 💰

An end-to-end Machine Learning pipeline that predicts employee salaries based on years of experience using Linear Regression in Python and `scikit-learn`.

---

## 📌 Overview

This project provides an interactive salary prediction tool for Chilman's Organization. It trains a `LinearRegression` model on real historical salary data (`Salary_Data.csv`), evaluates model accuracy using $R^2$ Score and Mean Squared Error (MSE), applies a conditional retraining step, and provides an interactive terminal CLI for running predictions.

---

## 🚀 Key Features

* **Data Pipeline**: Loads and splits tabular salary data into training and test sets using `pandas` and `train_test_split`.
* **Supervised Learning**: Trains a `LinearRegression` model on employee years of experience vs. annual salary.
* **Model Evaluation**: Assesses baseline performance using:
  * **$R^2$ Score** (Coefficient of Determination)
  * **MSE** (Mean Squared Error)
* **Conditional Retraining**: Evaluates performance thresholds ($R^2 < 0.8$ or $\text{MSE} > 1000$) to trigger model retraining on test data and export an optimized model.
* **Interactive CLI**: Terminal interface with continuous input validation, error handling, formatted currency output, and exit controls.

---

## 📂 Project Structure

```text
.
├── Salary_Data.csv    # Dataset with YearsExperience and Salary columns
├── train.py           # Data loading, 80/20 train/test split, and model training
├── eval.py            # Model evaluation ($R^2$, MSE), conditional retraining, and export
├── main.py            # Interactive CLI application for running real-time predictions
├── requirements.txt   # Python package dependencies
└── README.md          # Project documentation
```

---

## 🛠️ Technologies Used

* **Python 3**
* **scikit-learn**
* **pandas**
* **numpy**

---

## ⚙️ Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3 installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

Launch the salary predictor CLI by running `main.py`:

```bash
python3 main.py
```

> **Note**: Running `main.py` automatically imports and executes `eval.py` and `train.py`, outputting the training coefficients, evaluation scores, and retraining metrics before opening the interactive prompt.

---

## 📊 Pipeline Workflow

1. **`train.py`**:
   - Reads `Salary_Data.csv`.
   - Splits features (`YearsExperience`) and target (`Salary`) into an 80/20 train-test split.
   - Fits an initial `LinearRegression` model on `X_train` and `y_train`.
2. **`eval.py`**:
   - Predicts salaries for `X_test` and computes initial $R^2$ and MSE.
   - If $R^2 < 0.8$ or $\text{MSE} > 1000$, triggers retraining on `X_test` and prints updated metrics.
   - Exports `optimized_model`.
3. **`main.py`**:
   - Imports `optimized_model`.
   - Starts an interactive terminal loop prompting the user for years of experience.
   - Formats and displays the predicted annual salary (e.g., `$65,000.00`).
   - Accepts `'quit'`, `'exit'`, or `'q'` to gracefully exit.