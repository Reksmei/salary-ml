import numpy as np
from eval import optimized_model

def main():
    print("""
===================================================
  Welcome to Reks' Org Salary Predictor!
  Predict your salary based on years of experience.
===================================================
Type 'quit' at any prompt to exit.
""")

    while True:
        raw_input = input("\nEnter your years of experience: ").strip()

        if raw_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            years = float(raw_input)
            if years < 0:
                print("Experience cannot be negative. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number (e.g. 3.5).")
            continue

        # Predict with 2D float array
        input_array = np.array([[years]])
        pred_salary = optimized_model.predict(input_array)[0]

        print(f"-> Predicted Salary: ${pred_salary:,.2f} per year ({years} years experience)")

if __name__ == "__main__":
    main()
