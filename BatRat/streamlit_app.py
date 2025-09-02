# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm

st.set_page_config(page_title="Bat vs Rat Forage Analysis", layout="wide")

# ---------------------------
# Load datasets
# ---------------------------
@st.cache_data
def load_data():
    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    return dataset1, dataset2

dataset1, dataset2 = load_data()

st.title("🐭🦇 Bat vs Rat: Forage Behaviour Analysis")
st.markdown("**Investigation A** – Do bats perceive rats as predators?")

# ---------------------------
# Show raw data
# ---------------------------
st.subheader("📂 Dataset Preview")
tab1, tab2 = st.tabs(["Dataset 1 (Bat Landings)", "Dataset 2 (Rat Arrivals)"])

with tab1:
    st.write(dataset1.head())
with tab2:
    st.write(dataset2.head())

# ---------------------------
# Descriptive Statistics
# ---------------------------
st.subheader("📊 Descriptive Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Bat behaviour summary")
    st.write(dataset1.describe(include="all"))

with col2:
    st.write("### Rat activity summary")
    st.write(dataset2.describe(include="all"))

# Plot distribution of risk-taking
st.write("### Risk-taking behaviour (Bat landings)")
fig, ax = plt.subplots()
sns.countplot(data=dataset1, x="risk", hue="season", ax=ax)
ax.set_title("Risk-taking behaviour across seasons")
st.pyplot(fig)

# ---------------------------
# Inferential Analysis
# ---------------------------
st.subheader("📈 Inferential Analysis")

st.write("#### Chi-Square Test: Is risk-taking independent of rat presence?")
# Contingency table: risk vs reward
contingency = pd.crosstab(dataset1['risk'], dataset1['reward'])
chi2, p, dof, expected = chi2_contingency(contingency)

st.write("Contingency Table:")
st.write(contingency)
st.write(f"Chi2 = {chi2:.3f}, p-value = {p:.4f}")

if p < 0.05:
    st.success("✅ Significant association found between risk-taking and reward!")
else:
    st.warning("❌ No significant association found.")

# Logistic regression: Does rat presence/time affect bat risk-taking?
st.write("#### Logistic Regression: Predicting Bat Risk-taking")
# Simple regression with available features
dataset1 = dataset1.dropna(subset=["risk", "seconds_after_rat_arrival", "hours_after_sunset"])
X = dataset1[["seconds_after_rat_arrival", "hours_after_sunset"]]
X = sm.add_constant(X)
y = dataset1["risk"]

model = sm.Logit(y, X).fit(disp=0)
st.write(model.summary())

# ---------------------------
# Interactive Visualizations
# ---------------------------
st.subheader("📉 Interactive Visualizations")

choice = st.selectbox("Select variable to visualize:", dataset1.columns)

fig, ax = plt.subplots()
sns.histplot(dataset1[choice], kde=True, ax=ax)
ax.set_title(f"Distribution of {choice}")
st.pyplot(fig)

st.success("✅ Analysis completed. Use this app to explore relationships and test hypotheses.")
