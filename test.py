import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Import the functions from your refactored script
from credit_risk_pandas_refactored import (
    load_data,
    handle_missing_values,
    encode_categorical_variables,
    prepare_data_for_ml,
    filter_high_interest_loans,
    group_by_loan_grade,
    train_random_forest_model,
)


class TestCreditRiskAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up a small, controlled DataFrame for testing."""
        data = {
            "person_age": [22, 25, 28, 30],
            "person_income": [50000, 60000, 80000, 120000],
            "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT"],
            "person_emp_length": [2.0, 5.0, np.nan, 10.0],
            "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "PERSONAL"],
            "loan_grade": ["A", "B", "A", "C"],
            "loan_amnt": [5000, 10000, 8000, 15000],
            "loan_int_rate": [7.5, 12.2, np.nan, 16.0],
            "loan_status": [0, 1, 0, 1],
            "cb_person_default_on_file": ["N", "Y", "N", "N"],
        }
        self.df = pd.DataFrame(data)
        # Create a dummy CSV file for testing data loading
        self.test_csv_path = "test_dataset.csv"
        self.df.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Clean up created files after tests."""
        import os

        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_load_data(self):
        """Test that data is loaded correctly into a DataFrame."""
        loaded_df = load_data(self.test_csv_path)
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertFalse(loaded_df.empty)
        # Test for non-existent file
        self.assertIsNone(load_data("non_existent_file.csv"))

    def test_handle_missing_values(self):
        """Test that missing values in key columns are imputed."""
        processed_df = handle_missing_values(self.df)
        # Check that 'person_emp_length' and 'loan_int_rate' have no NaNs
        self.assertFalse(processed_df["person_emp_length"].isnull().any())
        self.assertFalse(processed_df["loan_int_rate"].isnull().any())
        # Check that other columns are unaffected
        self.assertEqual(
            self.df["person_age"].isnull().sum(),
            processed_df["person_age"].isnull().sum(),
        )

    def test_encode_categorical_variables(self):
        """Test that categorical columns are converted to numeric types."""
        encoded_df = encode_categorical_variables(self.df)
        categorical_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
        for col in categorical_cols:
            # Check if the dtype is now numeric (int or float)
            self.assertTrue(pd.api.types.is_numeric_dtype(encoded_df[col]))

    def test_prepare_data_for_ml(self):
        """Test the full data preparation pipeline."""
        X, y = prepare_data_for_ml(self.df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertNotIn("loan_status", X.columns)
        # After imputation, there should be no missing values
        self.assertEqual(X.isnull().sum().sum(), 0)

    def test_filter_high_interest_loans(self):
        """Test the filtering logic for high-interest loans."""
        filtered_df = filter_high_interest_loans(self.df, int_threshold=15)
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df["loan_int_rate"].iloc[0], 16.0)
        # Test edge case with no results
        empty_df = filter_high_interest_loans(self.df, int_threshold=20)
        self.assertTrue(empty_df.empty)

    def test_group_by_loan_grade(self):
        """Test the grouping and aggregation logic."""
        grouped_df = group_by_loan_grade(self.df)
        self.assertIn("A", grouped_df.index)
        self.assertIn("B", grouped_df.index)
        self.assertIn("C", grouped_df.index)
        # Check a specific calculated value (Grade A has 2 loans)
        self.assertEqual(grouped_df.loc["A"]["Total_Loans"], 2)
        # Check calculation of default rate for Grade A (one default / two loans)
        # Our sample has status [0, 1, 0, 1] for grades [A, B, A, C], so for A it is [0,0], mean is 0.
        self.assertEqual(grouped_df.loc["A"]["Default_Rate"], 0.0)

    def test_train_random_forest_model(self):
        """Test that the model training function returns a fitted model and an accuracy score."""
        X, y = prepare_data_for_ml(self.df)
        # Since the dataset is tiny, we'll use all of it for training and testing
        # to ensure stratify doesn't fail. In a real scenario, you need more data.
        model, accuracy = train_random_forest_model(X, y, test_size=0.5)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        # Check if the model is fitted by checking for an attribute like 'classes_'
        self.assertTrue(hasattr(model, "classes_"))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
