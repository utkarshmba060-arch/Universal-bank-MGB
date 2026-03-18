# 🏦 Universal Bank — Personal Loan Marketing Intelligence Dashboard

A comprehensive, interactive Streamlit dashboard built for the **Head of Marketing** at Universal Bank to hyper-personalise personal loan campaigns using machine learning and analytics.

---

## 📊 Dashboard Sections

| Section | What it covers |
|---|---|
| 🏠 **Overview** | KPI cards, dataset summary, class distribution, model snapshot |
| 📊 **Descriptive Analytics** | Age, income, education, family, CC spend, mortgage distributions |
| 🔍 **Diagnostic Analytics** | Feature-by-feature breakdown vs loan acceptance, correlation heatmap |
| 🤖 **Predictive Analytics** | Decision Tree, Random Forest & Gradient Boosted Tree — metrics table, single ROC curve, confusion matrices, feature importance |
| 🎯 **Prescriptive Analytics** | Ideal customer profile, radar chart, target segments, campaign action plan |
| 📁 **Predict New Data** | Upload any customer CSV → instant predictions + probability scores + download |

---

## 🚀 Run on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo, set **main file = `app.py`**
4. Click **Deploy** — done!

---

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

---

## 📁 File Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── UniversalBank.csv       # Training dataset (5,000 customers)
├── test_data_sample.csv    # Sample test data for predictions (100 rows)
└── README.md               # This file
```

---

## 🤖 ML Models

All three models are trained on a **70/30 stratified split** with class balancing to handle the 90/10 class imbalance:

- **Decision Tree** — `max_depth=5`, `class_weight='balanced'`
- **Random Forest** — `n_estimators=150`, `class_weight='balanced'`
- **Gradient Boosted Tree** — `n_estimators=150`, `learning_rate=0.08`

---

## 📌 Column Description

| Column | Description |
|---|---|
| Age | Customer age (years) |
| Experience | Professional experience (years) |
| Income | Annual income ($000) |
| Family | Family size (1–4) |
| CCAvg | Avg monthly CC spend ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Mortgage value ($000) |
| Securities Account | Has securities account (0/1) |
| CD Account | Has CD account (0/1) |
| Online | Uses online banking (0/1) |
| CreditCard | Has UniversalBank credit card (0/1) |
| **Personal Loan** | **Target: accepted loan offer (0/1)** |

---

*Built for Universal Bank Marketing Team | Powered by Streamlit + scikit-learn + Plotly*
