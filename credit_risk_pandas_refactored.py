"""
Credit Risk Analysis - Refactored for Testability

This script loads, cleans, preprocesses, and analyzes the credit risk dataset.
It also trains a Random Forest Classifier to predict loan defaults.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """
    Loads the credit risk dataset from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return None


def handle_missing_values(df):
    """
    Handles missing values in the DataFrame by imputing with the median.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    df_imputed = df.copy()
    # Impute with median for numerical columns
    df_imputed["person_emp_length"] = df_imputed["person_emp_length"].fillna(
        df_imputed["person_emp_length"].median()
    )
    df_imputed["loan_int_rate"] = df_imputed["loan_int_rate"].fillna(
        df_imputed["loan_int_rate"].median()
    )
    return df_imputed


def encode_categorical_variables(df):
    """
    Encodes categorical columns using LabelEncoder.

    Args:
        df (pd.DataFrame): The input DataFrame with categorical columns.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns encoded.
    """
    df_encoded = df.copy()
    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded


def prepare_data_for_ml(df):
    """
    A wrapper function to preprocess data for machine learning.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        tuple: A tuple containing features (X) and target (y) DataFrames.
    """
    df_clean = handle_missing_values(df)
    df_encoded = encode_categorical_variables(df_clean)

    X = df_encoded.drop("loan_status", axis=1)
    y = df_encoded["loan_status"]

    return X, y


def filter_high_interest_loans(df, interest_rate_threshold=15):
    """
    Filters the DataFrame for loans with an interest rate above a threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        interest_rate_threshold (float): The interest rate to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing only high-interest loans.
    """
    return df[df["loan_int_rate"] > interest_rate_threshold]


def group_by_loan_grade(df):
    """
    Groups the DataFrame by loan grade and calculates aggregate statistics.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with aggregated stats per loan grade.
    """
    grade_analysis = (
        df.groupby("loan_grade")
        .agg(
            {
                "loan_status": ["count", "mean"],
                "loan_int_rate": "mean",
                "loan_amnt": "mean",
            }
        )
        .round(3)
    )

    grade_analysis.columns = [
        "Total_Loans",
        "Default_Rate",
        "Avg_Interest_Rate",
        "Avg_Loan_Amount",
    ]
    return grade_analysis


def train_random_forest_model(X, y, test_size=0.2, random_state=42):
    """
    Trains a Random Forest Classifier and returns the model and its accuracy.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing the trained model and its accuracy score.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    rf_model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = rf_model.predict(X_test)
    accuracy = (y_test == y_pred).mean()  # equivalent to accuracy_score

    return rf_model, accuracy


# Main execution block (optional, for running the script directly)
if __name__ == "__main__":
    # 1. Load Data
    filepath = "data/credit_risk_dataset.csv"
    main_df = load_data(filepath)

    if main_df is not None:
        # 2. Filtering and Grouping
        high_risk = filter_high_interest_loans(main_df)
        print(f"\nFound {len(high_risk)} high-risk loans (>15% interest).")

        grade_summary = group_by_loan_grade(main_df)
        print("\nAnalysis by Loan Grade:")
        print(grade_summary)

        # 3. Machine Learning
        X_features, y_target = prepare_data_for_ml(main_df)
        model, model_accuracy = train_random_forest_model(X_features, y_target)

        print(f"\nML model trained successfully.")
        print(f"Model Accuracy: {model_accuracy:.3f}")
