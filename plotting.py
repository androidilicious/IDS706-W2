import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Advanced libs
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter


# Import helper from utils (not from main script to avoid circular import)
from analysis_utils import group_by_loan_grade


# ---------------------------
# Utility
# ---------------------------
def ensure_chart_dir(chart_dir="charts"):
    """Ensure charts directory exists."""
    if not os.path.isdir(chart_dir):
        os.makedirs(chart_dir)
    return chart_dir


# ---------------------------
# Core Visualizations
# ---------------------------
def visualize_credit_risk(df, chart_dir="charts"):
    """
    Generates and saves:
      1) Heatmap of default rates by grade & intent
      2) Bubble matrix (volume + default rate by grade & intent)
      3) Risk–Return curve (interest vs. default by grade)
    """
    chart_dir = ensure_chart_dir(chart_dir)

    # 1) Heatmap
    pivot = df.pivot_table(values="loan_status",
                           index="loan_grade",
                           columns="loan_intent",
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds", ax=ax)
    ax.set_title("Default Rate Heatmap by Grade & Intent")
    plt.tight_layout()
    fig.savefig(os.path.join(chart_dir, "heatmap_grade_intent.png"))
    plt.close(fig)

    # 2) Bubble Matrix
    grp = (
        df.groupby(["loan_grade","loan_intent"])
          .agg(Total_Loans=("loan_status","count"),
               Default_Rate=("loan_status","mean"))
          .reset_index()
    )
    max_ct = grp["Total_Loans"].max()
    grp["marker_size"] = grp["Total_Loans"].apply(
        lambda x: 100 + (x/max_ct)*1900
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        x=grp["loan_intent"], y=grp["loan_grade"],
        s=grp["marker_size"], c=grp["Default_Rate"],
        cmap="Reds", alpha=0.7, edgecolors="w", linewidth=1.2
    )
    plt.colorbar(sc, ax=ax, label="Default Rate")
    ax.set_title("Bubble Matrix: Loan Volume & Default Rate by Grade & Intent")
    plt.xticks(rotation=45, ha="right")
    for _, r in grp.iterrows():
        ax.text(r["loan_intent"], r["loan_grade"],
                f"{int(r['Total_Loans'])}\n{r['Default_Rate']:.0%}",
                ha="center", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(chart_dir, "bubble_matrix.png"))
    plt.close(fig)

    # 3) Risk–Return Curve
    grade_summary = group_by_loan_grade(df).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="Avg_Interest_Rate", y="Default_Rate",
                    hue="loan_grade", data=grade_summary, s=200, ax=ax)
    sns.lineplot(x="Avg_Interest_Rate", y="Default_Rate",
                 data=grade_summary, color="gray", linestyle="--", ax=ax)
    ax.set_title("Risk–Return Curve by Loan Grade")
    ax.set_xlabel("Average Interest Rate (%)")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    fig.savefig(os.path.join(chart_dir, "risk_return_curve.png"))
    plt.close(fig)

    print(f"✅ Core charts saved in {chart_dir}/")


# ---------------------------
# Advanced Visualizations
# ---------------------------
def sankey_intent_grade_status(df, chart_dir="charts"):
    """Sankey flow: Intent → Grade → Status"""
    chart_dir = ensure_chart_dir(chart_dir)
    flows = df.groupby(["loan_intent","loan_grade","loan_status"]).size().reset_index(name="count")
    intents = df["loan_intent"].unique().tolist()
    grades = df["loan_grade"].unique().tolist()
    statuses = ["Default","Paid"]

    labels = intents + grades + statuses
    label_index = {l:i for i,l in enumerate(labels)}

    sources, targets, values = [], [], []
    for _, row in flows.iterrows():
        sources.append(label_index[row["loan_intent"]])
        targets.append(label_index[row["loan_grade"]])
        values.append(row["count"])
        sources.append(label_index[row["loan_grade"]])
        targets.append(label_index["Default" if row["loan_status"]==1 else "Paid"])
        values.append(row["count"])

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=20),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text="Loan Flow: Intent → Grade → Status")
    fig.write_html(os.path.join(chart_dir,"sankey_flow.html"))
    print("✅ Sankey diagram saved")


def survival_curve(df, chart_dir="charts"):
    """Kaplan–Meier survival curve (needs loan_duration column)."""
    chart_dir = ensure_chart_dir(chart_dir)
    if "loan_duration" not in df.columns:
        print("⚠️ No 'loan_duration' column found, skipping survival curve.")
        return
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["loan_duration"], event_observed=df["loan_status"])
    ax = kmf.plot_survival_function()
    ax.set_title("Loan Survival Curve")
    ax.set_xlabel("Months")
    ax.set_ylabel("Probability of Survival")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir,"survival_curve.png"))
    plt.close()
    print("✅ Survival curve saved")


def correlation_network(df, chart_dir="charts"):
    """Correlation network of numeric features only."""
    chart_dir = ensure_chart_dir(chart_dir)

    # Select only numeric columns
    df_num = df.select_dtypes(include=["number"])
    corr = df_num.corr()

    import networkx as nx
    plt.figure(figsize=(10,8))
    G = nx.Graph()
    for i in corr.columns:
        for j in corr.columns:
            if i != j and abs(corr.loc[i,j]) > 0.3:
                G.add_edge(i, j, weight=abs(corr.loc[i,j]))

    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=1500,
            node_color="lightblue", edge_color="gray")
    plt.title("Correlation Network of Credit Risk Features")
    plt.savefig(os.path.join(chart_dir,"risk_network.png"))
    plt.close()
    print("✅ Correlation network saved")



def feature_importance_plot(model, X, chart_dir="charts"):
    """Feature importance from Random Forest model."""
    chart_dir = ensure_chart_dir(chart_dir)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(8,6))
    feat_imp.plot(kind="barh", color="teal")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir,"feature_importance.png"))
    plt.close()
    print("✅ Feature importance plot saved")
