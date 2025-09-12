#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Analysis - Week 2 Mini Assignment
# 
# **Student**: Diwas Puri  
# **Dataset**: Credit Risk Dataset  
# **Algorithm**: Random Forest Classifier  
# 
# This analysis covers all Week 2 requirements:
# 1. Import Dataset
# 2. Inspect Data
# 3. Basic Filtering and Grouping
# 4. Machine Learning Algorithm Exploration
# 5. Visualization
# 
# #### Disclosure
# 
# Because of the author's lack of knowledge on Machine Learning algorithms, Claude Sonnet 4.0 was used to generate the relavent code snippets.
# 
# ---

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")


# Load the dataset
df = pd.read_csv('data/credit_risk_dataset.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ## 2. Inspect the Data
# 
# Understanding the structure, types, and basic statistics of our dataset.

# Display first 5 rows
print("First 5 rows:")
df.head()


# Basic dataset information
print("Dataset Info:")
df.info()

print("\nSummary Statistics:")
df.describe()

# Check for missing values and target distribution
print("Missing Values:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])

print("\nTarget Variable (loan_status) Distribution:")
print(df['loan_status'].value_counts())
print(f"Default rate: {df['loan_status'].mean():.2%}")


# ## 3. Basic Filtering and Grouping
# 
# Applying filters to extract meaningful subsets and using groupby for analysis.


# Filter 1: High-risk loans (interest rate > 15%)
high_risk_loans = df[df['loan_int_rate'] > 15]
print(f"High-risk loans (>15% interest): {len(high_risk_loans):,} ({len(high_risk_loans)/len(df):.1%})")
print(f"Default rate for high-risk loans: {high_risk_loans['loan_status'].mean():.2%}")

# Filter 2: Young borrowers (age < 25)
young_borrowers = df[df['person_age'] < 25]
print(f"\nYoung borrowers (<25): {len(young_borrowers):,} ({len(young_borrowers)/len(df):.1%})")
print(f"Default rate for young borrowers: {young_borrowers['loan_status'].mean():.2%}")



# Grouping analysis by loan grade
print("Analysis by Loan Grade:")
grade_analysis = df.groupby('loan_grade').agg({
    'loan_status': ['count', 'mean'],
    'loan_int_rate': 'mean',
    'loan_amnt': 'mean'
}).round(3)

grade_analysis.columns = ['Total_Loans', 'Default_Rate', 'Avg_Interest_Rate', 'Avg_Loan_Amount']
grade_analysis


# ## 4. Visualization
# 
# Creating plots to visualize key patterns in the data.


# Create visualizations
plt.figure(figsize=(15, 5))

# Plot 1: Loan status distribution
plt.subplot(1, 3, 1)
df['loan_status'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                     colors=['lightgreen', 'lightcoral'],
                                     labels=['No Default', 'Default'])
plt.title('Loan Status Distribution')
plt.ylabel('')

# Plot 2: Default rate by loan grade
plt.subplot(1, 3, 2)
grade_defaults = df.groupby('loan_grade')['loan_status'].mean()
bars = plt.bar(grade_defaults.index, grade_defaults.values, color='skyblue')
plt.title('Default Rate by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Default Rate')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom')

# Plot 3: Interest rate distribution
plt.subplot(1, 3, 3)
plt.hist(df['loan_int_rate'].dropna(), bins=20, alpha=0.7, color='green')
plt.axvline(df['loan_int_rate'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["loan_int_rate"].mean():.2f}%')
plt.title('Interest Rate Distribution')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()


# ## 5. Machine Learning Algorithm Exploration
# 
# Experimenting with Random Forest classifier to predict loan defaults.


# Prepare data for machine learning
print("Preparing data for machine learning...")

# Create a copy for ML processing
df_ml = df.copy()

# Handle missing values
df_ml['person_emp_length'] = df_ml['person_emp_length'].fillna(df_ml['person_emp_length'].median())
df_ml['loan_int_rate'] = df_ml['loan_int_rate'].fillna(df_ml['loan_int_rate'].median())

print(f"Missing values after imputation: {df_ml.isnull().sum().sum()}")


# Encode categorical variables
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
label_encoders = {}

print("Encoding categorical variables:")
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} categories")

# Prepare features and target
X = df_ml.drop('loan_status', axis=1)
y = df_ml['loan_status']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution: {y.value_counts().values}")


# Train Random Forest model
print("Training Random Forest model...")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Train model (simplified for Week 2)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
feature_importance.head()

