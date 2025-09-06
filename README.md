# Credit Risk Analysis - Week 2 Mini Assignment

**Student**: Diwas Puri  
**Course**: Data Engineering Systems | IDS 706.01.Fa25  
**Dataset**: Credit Risk Dataset  
**Primary Algorithm**: Random Forest Classifier  

## 🎯 Project Overview

This project analyzes credit risk using a dataset of 32,581 loan applications to predict loan defaults. The analysis shows fundamental data science techniques including data exploration, visualization, feature engineering, and machine learning model development.

### 📋 Assignment Requirements Covered

✅ **Import Dataset** - Load and verify credit risk data  
✅ **Inspect Data** - Comprehensive exploratory data analysis  
✅ **Basic Filtering and Grouping** - Risk segmentation and statistical analysis  
✅ **Machine Learning Algorithm** - Random Forest classification model  
✅ **Visualization** - Multiple charts showing key patterns and insights  

## 📊 Dataset Information

- **Size**: 32,581 loan applications
- **Features**: 12 columns including demographics, loan details, and credit history
- **Target Variable**: Binary classification (0 = No Default, 1 = Default)
- **Default Rate**: 21.82%
- **Missing Data**: Employment length (895 missing), Interest rate (3,116 missing)

### 🏗️ Key Features

| Feature | Description | Type |
|---------|-------------|------|
| `person_age` | Age of borrower | Numeric |
| `person_income` | Annual income | Numeric |
| `person_home_ownership` | Home ownership status | Categorical |
| `person_emp_length` | Employment length in years | Numeric |
| `loan_intent` | Purpose of loan | Categorical |
| `loan_grade` | Loan grade (A-G) | Categorical |
| `loan_amnt` | Loan amount requested | Numeric |
| `loan_int_rate` | Interest rate (%) | Numeric |
| `loan_status` | **Target**: Default status (0/1) | Binary |
| `loan_percent_income` | Loan as percentage of income | Numeric |
| `cb_person_default_on_file` | Previous default history | Binary |
| `cb_person_cred_hist_length` | Credit history length | Numeric |

## 🛠️ Technical Requirements

### Dependencies
```python
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/androidilicious/IDS706-W2.git
   ```

2. **Ensure dataset is in place**
   ```
   data/
   └── credit_risk_dataset.csv
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook credit_risk_pandas.ipynb
   ```

## 📁 Project Structure

```
credit-risk-analysis/
├── data/
│   └── credit_risk_dataset.csv
├── credit_risk_pandas.ipynb
├── README.md
└── requirements.txt
```

## 🔍 Key Findings

### 📈 Risk Segmentation Analysis

| Loan Grade | Default Rate | Avg Interest Rate | Volume |
|------------|--------------|-------------------|---------|
| **A** | 10.0% | 7.3% | 10,777 loans |
| **B** | 16.3% | 11.0% | 10,451 loans |
| **C** | 20.7% | 13.5% | 6,458 loans |
| **D** | 59.0% | 15.4% | 3,626 loans |
| **E** | 64.4% | 17.0% | 964 loans |
| **F** | 70.5% | 18.6% | 241 loans |
| **G** | **98.4%** | 20.3% | 64 loans |

### 🏆 Model Performance

| Metric | Random Forest | Baseline |
|--------|---------------|----------|
| **Accuracy** | **93.0%** | 78.2% |


### 📊 Top Risk Factors (Feature Importance)

1. **Loan-to-Income Ratio** (22.3%) - Primary risk indicator
2. **Personal Income** (14.8%) - Higher income = lower risk
3. **Interest Rate** (11.6%) - Risk-based pricing validation
4. **Loan Grade** (10.9%) - Credit assessment accuracy
5. **Home Ownership** (10.8%) - Stability indicator

## 📚 Methodology

### Data Preprocessing Pipeline
1. **Missing Value Treatment**: Median imputation for numeric features
2. **Categorical Encoding**: Label encoding for ML compatibility  
3. **Data Validation**: Verification of no infinite/null values
4. **Train/Test Split**: 80/20 stratified split maintaining class balance

### Model Development
- **Algorithm**: Random Forest
- **Rationale**: Handles mixed data types, provides feature importance
- **Evaluation**: Accuracy

## ⚖️ Disclosure

Due to the author's developing knowledge in Machine Learning algorithms, **Claude Sonnet 4.0** was used to generate relevant code snippets and provide guidance on best practices. All analysis, interpretations, and insights were reviewed and validated by the author.


**Ready to dive deeper into credit risk analytics? Check out the full Jupyter notebook!** 🚀

---
*Last Updated: September 2025 | Week 2 Mini Assignment*