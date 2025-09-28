def group_by_loan_grade(df):
    grade_analysis = (
        df.groupby("loan_grade")
          .agg({
              "loan_status": ["count", "mean"],
              "loan_int_rate": "mean",
              "loan_amnt": "mean",
          })
          .round(3)
    )
    grade_analysis.columns = [
        "Total_Loans", "Default_Rate",
        "Avg_Interest_Rate", "Avg_Loan_Amount"
    ]
    return grade_analysis
