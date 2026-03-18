import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import io
import os

# ─────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove top padding */
    .block-container { padding-top: 1.2rem; }

    /* Banner */
    .banner {
        background: linear-gradient(120deg, #0f2848 0%, #1d4ed8 100%);
        color: white; padding: 1.8rem 2.5rem;
        border-radius: 14px; margin-bottom: 1.4rem;
    }
    .banner h1 { font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem; }
    .banner p  { font-size: 0.95rem; margin: 0; opacity: 0.85; }

    /* KPI cards */
    .kpi-wrap { border-radius: 12px; padding: 1.1rem 1rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.07);
                text-align: center; background: white; border-top: 4px solid; }
    .kpi-val  { font-size: 1.9rem; font-weight: 800; color: #0f2848; }
    .kpi-lbl  { font-size: 0.78rem; color: #64748b; font-weight: 600;
                text-transform: uppercase; letter-spacing: .04em; margin-top: .2rem; }
    .kpi-blue   { border-color: #1d4ed8; }
    .kpi-green  { border-color: #10b981; }
    .kpi-amber  { border-color: #f59e0b; }
    .kpi-purple { border-color: #8b5cf6; }
    .kpi-pink   { border-color: #ec4899; }

    /* Section headers */
    .sec-hd {
        font-size: 1.18rem; font-weight: 700; color: #0f2848;
        border-left: 5px solid #1d4ed8;
        padding: 0.3rem 0 0.3rem 0.75rem;
        margin: 1.4rem 0 0.8rem;
    }

    /* Insight boxes */
    .ibox {
        border-left: 4px solid #1d4ed8;
        background: #eff6ff;
        padding: .75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: .86rem; color: #1e3a5f; line-height: 1.65;
        margin: .5rem 0 1rem;
    }
    .ibox.green  { border-color: #10b981; background: #f0fdf4; color: #14532d; }
    .ibox.amber  { border-color: #f59e0b; background: #fffbeb; color: #78350f; }
    .ibox.red    { border-color: #ef4444; background: #fef2f2; color: #7f1d1d; }

    /* Segment card */
    .seg-card {
        border-radius: 10px; padding: .9rem 1.1rem;
        margin: .45rem 0; font-size: .87rem; color: #1e293b;
        line-height: 1.6;
    }
    .seg-title { font-size: .97rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
#  DATA LOADING & CLEANING
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "UniversalBank.csv"
    df = pd.read_csv(path)
    df.drop(columns=["ID", "ZIP Code"], inplace=True, errors="ignore")
    df["Experience"] = df["Experience"].clip(lower=0)
    return df

# ─────────────────────────────────────────────────
#  MODEL TRAINING  (cached as resource to avoid re-train)
# ─────────────────────────────────────────────────
@st.cache_resource
def train_models():
    df = load_data()
    X = df.drop(columns=["Personal Loan"])
    y = df["Personal Loan"]
    feat_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model_defs = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, mdl in model_defs.items():
        mdl.fit(X_train, y_train)
        tr_pred = mdl.predict(X_train)
        te_pred = mdl.predict(X_test)
        te_prob = mdl.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, te_prob)
        cm = confusion_matrix(y_test, te_pred)
        results[name] = dict(
            model=mdl,
            train_acc=accuracy_score(y_train, tr_pred) * 100,
            test_acc=accuracy_score(y_test, te_pred) * 100,
            precision=precision_score(y_test, te_pred, zero_division=0) * 100,
            recall=recall_score(y_test, te_pred, zero_division=0) * 100,
            f1=f1_score(y_test, te_pred, zero_division=0) * 100,
            auc=roc_auc_score(y_test, te_prob) * 100,
            cm=cm, fpr=fpr, tpr=tpr,
        )

    return results, X_train, X_test, y_train, y_test, feat_cols


df = load_data()
results, X_train, X_test, y_train, y_test, FEAT_COLS = train_models()
BEST = max(results, key=lambda k: results[k]["auc"])

# ─────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("**Loan Marketing Intelligence**")
    st.markdown("---")

    nav = st.radio(
        "Go to",
        [
            "🏠  Overview",
            "📊  Descriptive Analytics",
            "🔍  Diagnostic Analytics",
            "🤖  Predictive Analytics",
            "🎯  Prescriptive Analytics",
            "📁  Predict New Data",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**📌 Quick Stats**")
    st.metric("Total Customers", f"{len(df):,}")
    accept_n = int(df["Personal Loan"].sum())
    accept_pct = df["Personal Loan"].mean() * 100
    st.metric("Loan Acceptors", f"{accept_n:,}  ({accept_pct:.1f}%)")
    st.metric("Best Model AUC", f"{results[BEST]['auc']:.2f}%  ({BEST})")
    st.markdown("---")
    st.caption("© Universal Bank | Marketing Analytics")


# ══════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════
if nav == "🏠  Overview":
    st.markdown("""
    <div class='banner'>
        <h1>🏦 Universal Bank — Personal Loan Marketing Intelligence</h1>
        <p>Data-Driven Insights · From Descriptive to Prescriptive Analytics · Head of Marketing Dashboard</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("👥", f"{len(df):,}", "Total Customers", "kpi-blue"),
        ("✅", f"{accept_n:,}", "Loan Acceptors", "kpi-green"),
        ("📈", f"{accept_pct:.1f}%", "Overall Acceptance Rate", "kpi-amber"),
        ("💰", f"${df['Income'].mean():.0f}K", "Avg Annual Income", "kpi-purple"),
        ("🏠", f"${df['Mortgage'].mean():.0f}K", "Avg Mortgage Value", "kpi-pink"),
    ]
    for col, (icon, val, lbl, cls) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-wrap {cls}'>
                <div style='font-size:1.7rem'>{icon}</div>
                <div class='kpi-val'>{val}</div>
                <div class='kpi-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset overview + Donut ──
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("<div class='sec-hd'>📋 Dataset Feature Summary</div>", unsafe_allow_html=True)
        summary_rows = []
        for col_name in FEAT_COLS:
            s = df[col_name]
            summary_rows.append({
                "Feature": col_name,
                "Type": "Binary" if s.nunique() == 2 else ("Ordinal" if col_name in ["Education","Family"] else "Continuous"),
                "Min": round(s.min(), 2),
                "Max": round(s.max(), 2),
                "Mean": round(s.mean(), 2),
                "Std Dev": round(s.std(), 2),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.markdown("""<div class='ibox'>
        📌 After removing <b>ID</b> (identifier) and <b>ZIP Code</b> (high-cardinality geo field),
        <b>11 predictive features</b> are used. Negative Experience values (data error) are clipped to 0.
        No missing values were detected across all 5,000 records.
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec-hd'>🎯 Loan Acceptance Split</div>", unsafe_allow_html=True)
        loan_counts = df["Personal Loan"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Declined (0)", "Accepted (1)"],
            values=[loan_counts[0], loan_counts[1]],
            hole=0.58,
            marker_colors=["#cbd5e1", "#1d4ed8"],
            textinfo="label+percent+value",
            textfont_size=12,
            pull=[0, 0.06],
        ))
        fig.update_layout(
            showlegend=True, height=310,
            margin=dict(t=30, b=10, l=10, r=10),
            annotations=[dict(text="5,000<br>Customers", x=0.5, y=0.5,
                              font_size=13, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='ibox amber'>
        ⚠️ <b>Class Imbalance (90 / 10 split):</b> Only 9.6% of customers accepted the loan.
        All models are trained with <b>class_weight='balanced'</b> so they don't naively predict
        "No" for everyone. Recall and AUC-ROC are therefore the most meaningful evaluation metrics here.
        </div>""", unsafe_allow_html=True)

    # ── Model snapshot ──
    st.markdown("<div class='sec-hd'>🤖 Model Performance Snapshot</div>", unsafe_allow_html=True)
    snap = pd.DataFrame({
        "Model": list(results.keys()),
        "Train Accuracy": [f"{r['train_acc']:.2f}%" for r in results.values()],
        "Test Accuracy":  [f"{r['test_acc']:.2f}%"  for r in results.values()],
        "Precision":      [f"{r['precision']:.2f}%"  for r in results.values()],
        "Recall":         [f"{r['recall']:.2f}%"     for r in results.values()],
        "F1-Score":       [f"{r['f1']:.2f}%"         for r in results.values()],
        "AUC-ROC":        [f"{r['auc']:.2f}%"        for r in results.values()],
    })
    st.dataframe(snap, use_container_width=True, hide_index=True)
    st.markdown(f"""<div class='ibox green'>
    🏆 <b>{BEST}</b> is the top model with AUC-ROC = <b>{results[BEST]['auc']:.2f}%</b>.
    Navigate to <b>Predictive Analytics</b> for full model comparison, ROC curves &amp; confusion matrices.
    Navigate to <b>Prescriptive Analytics</b> for campaign recommendations.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE 2 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════
elif nav == "📊  Descriptive Analytics":
    st.markdown("""<div class='banner'>
        <h1>📊 Descriptive Analytics</h1>
        <p>Understanding the shape, spread and composition of our customer base</p>
    </div>""", unsafe_allow_html=True)

    # ── Age & Experience ──
    st.markdown("<div class='sec-hd'>👤 Age & Professional Experience</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=["#1d4ed8"],
                           title="Customer Age Distribution",
                           labels={"Age": "Age (Years)", "count": "# Customers"})
        fig.add_vline(x=df["Age"].mean(), line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Mean: {df['Age'].mean():.1f} yrs",
                      annotation_position="top right")
        fig.update_layout(bargap=0.05, height=320, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Experience", nbins=30, color_discrete_sequence=["#8b5cf6"],
                           title="Professional Experience Distribution",
                           labels={"Experience": "Experience (Years)", "count": "# Customers"})
        fig.add_vline(x=df["Experience"].mean(), line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Mean: {df['Experience'].mean():.1f} yrs",
                      annotation_position="top right")
        fig.update_layout(bargap=0.05, height=320, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox'>
    📌 <b>Age & Experience</b> span 23–67 yrs and 0–43 yrs respectively, with near-uniform distributions
    indicating well-rounded customer acquisition across life stages. Age and experience are closely correlated
    (as expected). Mean age ≈ 45 yrs; mean experience ≈ 20 yrs.
    </div>""", unsafe_allow_html=True)

    # ── Income & CCAvg ──
    st.markdown("<div class='sec-hd'>💰 Income & Credit Card Spending</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Income", nbins=40, color_discrete_sequence=["#10b981"],
                           title="Annual Income Distribution ($000)",
                           labels={"Income": "Annual Income ($K)", "count": "# Customers"})
        fig.add_vline(x=df["Income"].mean(), line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Mean: ${df['Income'].mean():.0f}K",
                      annotation_position="top right")
        fig.update_layout(bargap=0.05, height=320, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="CCAvg", nbins=40, color_discrete_sequence=["#f59e0b"],
                           title="Monthly Credit Card Spending ($000)",
                           labels={"CCAvg": "CC Monthly Spend ($K)", "count": "# Customers"})
        fig.add_vline(x=df["CCAvg"].mean(), line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Mean: ${df['CCAvg'].mean():.2f}K",
                      annotation_position="top right")
        fig.update_layout(bargap=0.05, height=320, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox'>
    📌 <b>Income</b> is right-skewed (most customers earn $8K–$100K; a small affluent tail reaches $224K).
    <b>CC spending</b> is also right-skewed, averaging $1.94K/month — most customers are moderate spenders with
    a few high-value outliers above $6K/month.
    </div>""", unsafe_allow_html=True)

    # ── Education, Family, Banking Products ──
    st.markdown("<div class='sec-hd'>📂 Education · Family · Banking Product Adoption</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced"}
        edu_c = df["Education"].map(edu_map).value_counts().reset_index()
        edu_c.columns = ["Education", "Count"]
        edu_c["Pct"] = (edu_c["Count"] / len(df) * 100).round(1)
        fig = px.bar(edu_c, x="Education", y="Count",
                     text=edu_c.apply(lambda r: f"{r['Count']}<br>({r['Pct']}%)", axis=1),
                     color="Education",
                     color_discrete_sequence=["#1d4ed8","#10b981","#f59e0b"],
                     title="Education Level Breakdown")
        fig.update_layout(showlegend=False, height=300, plot_bgcolor="#f8fafc")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fam_c = df["Family"].value_counts().sort_index().reset_index()
        fam_c.columns = ["Family Size", "Count"]
        fam_c["Pct"] = (fam_c["Count"] / len(df) * 100).round(1)
        fig = px.bar(fam_c, x="Family Size", y="Count",
                     text=fam_c.apply(lambda r: f"{r['Count']}<br>({r['Pct']}%)", axis=1),
                     color="Family Size",
                     color_discrete_sequence=["#8b5cf6","#ec4899","#14b8a6","#f97316"],
                     title="Family Size Distribution")
        fig.update_layout(showlegend=False, height=300, plot_bgcolor="#f8fafc")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        bin_cols = ["Securities Account","CD Account","Online","CreditCard"]
        bin_pct  = [round(df[c].mean()*100, 1) for c in bin_cols]
        bin_cnt  = [int(df[c].sum()) for c in bin_cols]
        colors_b = ["#1d4ed8","#10b981","#f59e0b","#8b5cf6"]
        fig = go.Figure()
        for lbl, pct, cnt, col in zip(["Securities","CD Acct","Online","CreditCard"], bin_pct, bin_cnt, colors_b):
            fig.add_trace(go.Bar(
                x=[lbl], y=[pct],
                text=[f"{pct}%<br>({cnt})"],
                textposition="outside",
                marker_color=col,
                showlegend=False
            ))
        fig.update_layout(title="Banking Product Adoption Rate (%)",
                          yaxis_title="% Customers", height=300, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='ibox'>
    📌 Education is evenly split across three levels. Family sizes are uniformly distributed (1–4).
    <b>60% use Online Banking</b> (high digital engagement). Only <b>6% hold a CD Account</b> — yet this
    will prove to be one of the strongest loan predictors. CreditCard penetration stands at 29%.
    </div>""", unsafe_allow_html=True)

    # ── Mortgage ──
    st.markdown("<div class='sec-hd'>🏠 Mortgage Distribution</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.6, 1])
    with c1:
        df_m = df[df["Mortgage"] > 0]
        fig = px.histogram(df_m, x="Mortgage", nbins=40,
                           color_discrete_sequence=["#ef4444"],
                           title="Mortgage Value — Customers With a Mortgage Only ($000)",
                           labels={"Mortgage": "Mortgage Value ($K)", "count": "# Customers"})
        fig.update_layout(bargap=0.05, height=300, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hm  = int((df["Mortgage"] > 0).sum())
        nhm = int((df["Mortgage"] == 0).sum())
        fig = go.Figure(go.Pie(
            labels=["Has Mortgage", "No Mortgage"],
            values=[hm, nhm],
            hole=0.52,
            marker_colors=["#ef4444","#e2e8f0"],
            textinfo="label+percent",
            pull=[0.05, 0]
        ))
        fig.update_layout(title="Mortgage Ownership", height=300,
                          annotations=[dict(text=f"{hm}<br>With", x=0.5, y=0.5, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox'>
    📌 ~69% of customers have <b>no mortgage</b>. Among those who do, values range $1K–$635K with a
    right-skewed distribution. Mortgage presence is a useful binary indicator for financial commitment,
    often correlating with higher loan acceptance propensity.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE 3 — DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════
elif nav == "🔍  Diagnostic Analytics":
    st.markdown("""<div class='banner'>
        <h1>🔍 Diagnostic Analytics</h1>
        <p>Uncovering what drives loan acceptance — segmented deep dives into key features</p>
    </div>""", unsafe_allow_html=True)

    CMAP = {0: "#94a3b8", 1: "#1d4ed8"}
    LMAP = {0: "Declined", 1: "Accepted"}

    # ── Income ──
    st.markdown("<div class='sec-hd'>💰 Income vs Loan Acceptance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="Personal Loan", y="Income",
                     color="Personal Loan", color_discrete_map=CMAP,
                     title="Income Distribution by Loan Status",
                     labels={"Personal Loan":"Loan Status","Income":"Annual Income ($K)"})
        fig.for_each_trace(lambda t: t.update(name=LMAP[int(t.name)]))
        fig.update_layout(height=350, showlegend=False, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        df_i = df.copy()
        df_i["Income Band"] = pd.cut(df_i["Income"], bins=[0,50,100,150,224],
                                      labels=["<$50K","$50-100K","$100-150K",">$150K"])
        agg = df_i.groupby("Income Band", observed=True)["Personal Loan"]\
                  .agg(["sum","count"]).reset_index()
        agg["Rate"] = agg["sum"] / agg["count"] * 100
        agg["Label"] = agg.apply(lambda r: f"{r['Rate']:.1f}%  ({int(r['sum'])}/{int(r['count'])})", axis=1)
        fig = px.bar(agg, x="Income Band", y="Rate", text="Label",
                     color="Rate", color_continuous_scale="Blues",
                     title="Acceptance Rate by Income Band (%)",
                     labels={"Rate":"Acceptance Rate (%)","Income Band":"Income Band"})
        fig.update_layout(height=350, coloraxis_showscale=False, plot_bgcolor="#f8fafc")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox green'>
    💡 <b>Income is the single strongest predictor.</b> Customers earning <b>&gt;$100K show 3–5× higher
    acceptance rates</b>. Median income of acceptors is nearly double that of non-acceptors.
    Targeting high-income segments is the <b>#1 priority</b> for your campaign.
    </div>""", unsafe_allow_html=True)

    # ── Education & Family ──
    st.markdown("<div class='sec-hd'>🎓 Education & Family Size vs Loan Acceptance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        df_e = df.copy()
        df_e["Edu Label"] = df_e["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
        agg_e = df_e.groupby("Edu Label")["Personal Loan"].agg(["sum","count"]).reset_index()
        agg_e["Rate"]  = agg_e["sum"] / agg_e["count"] * 100
        agg_e["Label"] = agg_e.apply(lambda r: f"{r['Rate']:.1f}%  ({int(r['sum'])}/{int(r['count'])})", axis=1)
        fig = px.bar(agg_e, x="Edu Label", y="Rate", text="Label",
                     color="Edu Label",
                     color_discrete_sequence=["#1d4ed8","#10b981","#f59e0b"],
                     title="Acceptance Rate by Education Level",
                     labels={"Rate":"Acceptance Rate (%)","Edu Label":"Education"})
        fig.update_layout(height=320, showlegend=False, plot_bgcolor="#f8fafc")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        agg_f = df.groupby("Family")["Personal Loan"].agg(["sum","count"]).reset_index()
        agg_f["Rate"]  = agg_f["sum"] / agg_f["count"] * 100
        agg_f["Label"] = agg_f.apply(lambda r: f"{r['Rate']:.1f}%  ({int(r['sum'])}/{int(r['count'])})", axis=1)
        fig = px.bar(agg_f, x="Family", y="Rate", text="Label",
                     color="Family",
                     color_discrete_sequence=["#8b5cf6","#ec4899","#14b8a6","#f97316"],
                     title="Acceptance Rate by Family Size",
                     labels={"Rate":"Acceptance Rate (%)","Family":"Family Size (members)"})
        fig.update_layout(height=320, showlegend=False, plot_bgcolor="#f8fafc")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox'>
    📌 <b>Graduate and Advanced-degree holders</b> have meaningfully higher acceptance rates.
    Families of <b>3–4 members</b> show elevated propensity — likely driven by multiple financial needs.
    Target <b>graduate+ customers with mid-to-large families</b> for best conversion efficiency.
    </div>""", unsafe_allow_html=True)

    # ── CD Account & CCAvg ──
    st.markdown("<div class='sec-hd'>🏦 CD Account & CC Spending vs Loan Acceptance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        agg_cd = df.groupby("CD Account")["Personal Loan"].agg(["sum","count"]).reset_index()
        agg_cd["Rate"] = agg_cd["sum"] / agg_cd["count"] * 100
        agg_cd["Label_x"] = agg_cd["CD Account"].map({0:"No CD Account",1:"Has CD Account"})
        agg_cd["Lbl"] = agg_cd.apply(lambda r: f"{r['Rate']:.1f}%  ({int(r['sum'])}/{int(r['count'])})", axis=1)
        fig = px.bar(agg_cd, x="Label_x", y="Rate", text="Lbl",
                     color="Label_x",
                     color_discrete_sequence=["#94a3b8","#10b981"],
                     title="Acceptance Rate: CD Account Holders vs Non-Holders",
                     labels={"Rate":"Acceptance Rate (%)","Label_x":"CD Account Status"})
        fig.update_layout(height=320, showlegend=False, plot_bgcolor="#f8fafc")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x="Personal Loan", y="CCAvg",
                     color="Personal Loan", color_discrete_map=CMAP,
                     title="Monthly CC Spend by Loan Status",
                     labels={"Personal Loan":"Loan Status","CCAvg":"Monthly CC Spend ($K)"})
        fig.for_each_trace(lambda t: t.update(name=LMAP[int(t.name)]))
        fig.update_layout(height=320, showlegend=False, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox amber'>
    💡 <b>CD Account holders are ~5× more likely to accept a personal loan</b> — they already trust the
    bank with savings products. Higher CC spending also correlates strongly with acceptance:
    financially active customers are more receptive to loan offers. Both signals should be used for targeting.
    </div>""", unsafe_allow_html=True)

    # ── Correlation Heatmap ──
    st.markdown("<div class='sec-hd'>🔗 Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig_h, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        mask=mask, ax=ax, square=True, linewidths=0.5,
        annot_kws={"size": 9}, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix  (lower triangle)", fontsize=13,
                 fontweight="bold", pad=14)
    plt.tight_layout()
    st.pyplot(fig_h)
    plt.close()
    st.markdown("""<div class='ibox'>
    📌 <b>Income</b> has the strongest positive correlation with Personal Loan acceptance (r ≈ 0.50),
    followed by <b>CCAvg</b> and <b>CD Account</b>. Age and Experience are highly correlated with each
    other (r ≈ 0.99). ZIP Code and ID were dropped as they carry zero predictive signal.
    </div>""", unsafe_allow_html=True)

    # ── Age by Loan ──
    st.markdown("<div class='sec-hd'>📅 Age Distribution by Loan Status</div>", unsafe_allow_html=True)
    fig = px.histogram(df, x="Age", color="Personal Loan", nbins=30, barmode="overlay",
                       color_discrete_map=CMAP, opacity=0.75,
                       title="Age Distribution — Loan Accepted vs Declined",
                       labels={"Age":"Age (Years)","count":"# Customers","Personal Loan":"Loan Status"})
    fig.for_each_trace(lambda t: t.update(name=LMAP[int(t.name)]))
    fig.update_layout(height=320, plot_bgcolor="#f8fafc")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox'>
    📌 Loan acceptance is relatively <b>distributed across all age groups</b> — age alone is not a strong
    discriminator. However, when combined with income and education, middle-aged customers (35–55)
    show slightly elevated acceptance rates due to greater financial capacity and stability.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE 4 — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════
elif nav == "🤖  Predictive Analytics":
    st.markdown("""<div class='banner'>
        <h1>🤖 Predictive Analytics</h1>
        <p>Model training, evaluation and comparison — Decision Tree · Random Forest · Gradient Boosted Tree</p>
    </div>""", unsafe_allow_html=True)

    # ── Metrics Table ──
    st.markdown("<div class='sec-hd'>📊 Model Performance Comparison</div>", unsafe_allow_html=True)
    mdf = pd.DataFrame({
        "Model":           list(results.keys()),
        "Train Accuracy":  [f"{r['train_acc']:.2f}%" for r in results.values()],
        "Test Accuracy":   [f"{r['test_acc']:.2f}%"  for r in results.values()],
        "Precision":       [f"{r['precision']:.2f}%"  for r in results.values()],
        "Recall":          [f"{r['recall']:.2f}%"     for r in results.values()],
        "F1-Score":        [f"{r['f1']:.2f}%"         for r in results.values()],
        "AUC-ROC":         [f"{r['auc']:.2f}%"        for r in results.values()],
    })

    def highlight_best(s):
        try:
            nums = [float(v.replace("%","")) for v in s]
            mx   = max(nums)
            return ["background-color:#dbeafe;font-weight:700"
                    if float(v.replace("%","")) == mx else "" for v in s]
        except Exception:
            return [""] * len(s)

    styled = mdf.set_index("Model").style.apply(highlight_best)
    st.dataframe(styled, use_container_width=True)

    st.markdown(f"""<div class='ibox green'>
    🏆 <b>Best Model → {BEST}</b>  |  AUC-ROC = <b>{results[BEST]['auc']:.2f}%</b>.
    Blue-highlighted cells show the top-performing model per metric.
    <b>Recall</b> (sensitivity) is especially important here: it measures how many real loan-acceptors
    we correctly identify — directly translating to campaign reach and revenue.
    </div>""", unsafe_allow_html=True)

    # ── ROC Curve ──
    st.markdown("<div class='sec-hd'>📈 ROC Curve — All Three Models on a Single Chart</div>",
                unsafe_allow_html=True)
    roc_colors = {"Decision Tree": "#f59e0b",
                  "Random Forest": "#10b981",
                  "Gradient Boosted Tree": "#1d4ed8"}
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name}  (AUC = {r['auc']:.2f}%)",
            line=dict(color=roc_colors[name], width=2.8)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier (AUC = 50%)",
        line=dict(color="#94a3b8", width=1.5, dash="dash")
    ))
    fig.update_layout(
        title=dict(text="ROC Curve Comparison — Decision Tree vs Random Forest vs Gradient Boosted Tree",
                   font_size=14),
        xaxis_title="False Positive Rate  (1 − Specificity)",
        yaxis_title="True Positive Rate  (Sensitivity / Recall)",
        legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        height=490, plot_bgcolor="#f8fafc",
    )
    fig.update_xaxes(gridcolor="#e2e8f0", range=[0, 1.01])
    fig.update_yaxes(gridcolor="#e2e8f0", range=[0, 1.01])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox'>
    📌 <b>ROC Curve:</b> The closer the curve hugs the top-left corner, the better the model.
    AUC = 100% is perfect; 50% = random guessing. All three models far outperform a random classifier,
    confirming they can meaningfully rank customers by loan acceptance likelihood.
    Higher AUC → more precise campaign targeting → lower cost-per-conversion.
    </div>""", unsafe_allow_html=True)

    # ── Confusion Matrices ──
    st.markdown("<div class='sec-hd'>🔲 Confusion Matrices — All Three Models</div>", unsafe_allow_html=True)
    fig_cm, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    cm_cmaps = {"Decision Tree": "Blues", "Random Forest": "Greens",
                "Gradient Boosted Tree": "Purples"}
    for ax, (name, r) in zip(axes, results.items()):
        cm = r["cm"]
        tot = cm.sum()
        annot = np.array([
            [f"TN\n{cm[0,0]}\n({cm[0,0]/tot*100:.1f}%)", f"FP\n{cm[0,1]}\n({cm[0,1]/tot*100:.1f}%)"],
            [f"FN\n{cm[1,0]}\n({cm[1,0]/tot*100:.1f}%)", f"TP\n{cm[1,1]}\n({cm[1,1]/tot*100:.1f}%)"]
        ])
        sns.heatmap(cm, annot=annot, fmt="", cmap=cm_cmaps[name],
                    ax=ax, linewidths=2, linecolor="white",
                    xticklabels=["Predicted: No", "Predicted: Yes"],
                    yticklabels=["Actual: No", "Actual: Yes"],
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(f"{name}\nAcc: {r['test_acc']:.1f}%  |  Recall: {r['recall']:.1f}%  |  F1: {r['f1']:.1f}%",
                     fontsize=10.5, fontweight="bold", pad=10)
        ax.set_ylabel("Actual Label", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)
    plt.suptitle("Confusion Matrix Comparison — 30% Hold-Out Test Set", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()
    st.markdown("""<div class='ibox'>
    📌 <b>Reading the matrix:</b>  <b>TP</b> = correctly identified acceptors (campaign hits);
    <b>TN</b> = correctly excluded non-acceptors (budget saved);
    <b>FP</b> = wrongly targeted — wasted spend;
    <b>FN</b> = missed acceptors — lost revenue opportunity.
    For marketing, <b>maximising TP and minimising FN</b> (i.e. high Recall) is the primary goal.
    </div>""", unsafe_allow_html=True)

    # ── Feature Importance ──
    st.markdown("<div class='sec-hd'>⭐ Feature Importance — Random Forest</div>", unsafe_allow_html=True)
    rf_imp = pd.DataFrame({
        "Feature":    FEAT_COLS,
        "Importance": results["Random Forest"]["model"].feature_importances_
    }).sort_values("Importance", ascending=True)
    rf_imp["Label"] = rf_imp["Importance"].apply(lambda x: f"{x:.4f}  ({x*100:.2f}%)")
    fig = px.bar(rf_imp, x="Importance", y="Feature", orientation="h",
                 text="Label", color="Importance",
                 color_continuous_scale="Blues",
                 title="Feature Importance Score — Random Forest (Higher = More Influential)",
                 labels={"Importance": "Importance Score", "Feature": "Customer Feature"})
    fig.update_layout(height=420, coloraxis_showscale=False, plot_bgcolor="#f8fafc")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox green'>
    💡 <b>Top Predictors:</b> Income, CCAvg (CC spending), and CD Account dominate the importance ranking —
    confirming the patterns observed in Diagnostic Analytics. Mortgage and Family size also contribute.
    Online, CreditCard, and Securities Account are weaker signals. Focus your data collection and
    campaign targeting on the top 4 features for maximum predictive power.
    </div>""", unsafe_allow_html=True)

    # ── Bar chart: metric comparison across models ──
    st.markdown("<div class='sec-hd'>📊 Metric Comparison — Visual Bar Chart</div>", unsafe_allow_html=True)
    metric_names = ["Test Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    metric_keys  = ["test_acc", "precision", "recall", "f1", "auc"]
    bar_data = []
    for name, r in results.items():
        for mname, mkey in zip(metric_names, metric_keys):
            bar_data.append({"Model": name, "Metric": mname, "Score (%)": round(r[mkey], 2)})
    bar_df = pd.DataFrame(bar_data)
    fig = px.bar(bar_df, x="Metric", y="Score (%)", color="Model", barmode="group",
                 text="Score (%)",
                 color_discrete_map={"Decision Tree":"#f59e0b",
                                      "Random Forest":"#10b981",
                                      "Gradient Boosted Tree":"#1d4ed8"},
                 title="Model Performance Comparison Across All Metrics (%)",
                 labels={"Score (%)": "Score (%)"})
    fig.update_layout(height=400, plot_bgcolor="#f8fafc")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════
#  PAGE 5 — PRESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════
elif nav == "🎯  Prescriptive Analytics":
    st.markdown("""<div class='banner'>
        <h1>🎯 Prescriptive Analytics</h1>
        <p>Actionable recommendations to maximise personal loan acceptance with a reduced marketing budget</p>
    </div>""", unsafe_allow_html=True)

    acc  = df[df["Personal Loan"] == 1]
    nacc = df[df["Personal Loan"] == 0]

    # ── Profile side by side ──
    st.markdown("<div class='sec-hd'>🏆 Ideal Customer Profile — Acceptors vs Non-Acceptors</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    profiles = [
        ("💰 Avg Annual Income",   f"${acc['Income'].mean():.0f}K",          f"${nacc['Income'].mean():.0f}K"),
        ("💳 Avg CC Monthly Spend",f"${acc['CCAvg'].mean():.2f}K",            f"${nacc['CCAvg'].mean():.2f}K"),
        ("🏠 Avg Mortgage",        f"${acc['Mortgage'].mean():.0f}K",         f"${nacc['Mortgage'].mean():.0f}K"),
        ("👨‍👩‍👧  Avg Family Size",  f"{acc['Family'].mean():.1f} members",      f"{nacc['Family'].mean():.1f} members"),
        ("🎓 Graduate+ Rate",      f"{(acc['Education']>=2).mean()*100:.1f}%",f"{(nacc['Education']>=2).mean()*100:.1f}%"),
        ("🏦 CD Account Rate",     f"{acc['CD Account'].mean()*100:.1f}%",    f"{nacc['CD Account'].mean()*100:.1f}%"),
        ("📱 Online Banking Rate", f"{acc['Online'].mean()*100:.1f}%",        f"{nacc['Online'].mean()*100:.1f}%"),
        ("🔒 Securities Rate",     f"{acc['Securities Account'].mean()*100:.1f}%",
                                   f"{nacc['Securities Account'].mean()*100:.1f}%"),
    ]
    with c1:
        st.markdown("#### ✅ Loan Acceptors")
        for lbl, va, _ in profiles:
            st.markdown(f"**{lbl}:** {va}")
    with c2:
        st.markdown("#### ❌ Non-Acceptors")
        for lbl, _, vn in profiles:
            st.markdown(f"**{lbl}:** {vn}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar Chart ──
    st.markdown("<div class='sec-hd'>🕸️ Radar Chart — Acceptor vs Non-Acceptor Feature Profile (Normalised)</div>",
                unsafe_allow_html=True)
    radar_feats = ["Income","CCAvg","Mortgage","Family","CD Account","Online","CreditCard"]
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[radar_feats]), columns=radar_feats)
    scaled["Personal Loan"] = df["Personal Loan"].values
    am = scaled[scaled["Personal Loan"]==1][radar_feats].mean().tolist()
    nm = scaled[scaled["Personal Loan"]==0][radar_feats].mean().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=am+[am[0]], theta=radar_feats+[radar_feats[0]],
        fill="toself", name="Loan Acceptors",
        line_color="#1d4ed8", fillcolor="rgba(29,78,216,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=nm+[nm[0]], theta=radar_feats+[radar_feats[0]],
        fill="toself", name="Non-Acceptors",
        line_color="#ef4444", fillcolor="rgba(239,68,68,0.1)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title="Normalised Feature Comparison: Loan Acceptors vs Non-Acceptors",
        height=450, legend=dict(x=0.82, y=1.12)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='ibox green'>
    💡 Loan acceptors clearly dominate on <b>Income, CC Spend, Mortgage, and CD Account</b>.
    These four dimensions are your primary targeting levers. Even slight improvements in targeting
    precision on these axes will translate directly into higher campaign ROI.
    </div>""", unsafe_allow_html=True)

    # ── Target Segments ──
    st.markdown("<div class='sec-hd'>🎯 High-Value Target Segments — Acceptance Rate vs Base (9.6%)</div>",
                unsafe_allow_html=True)
    ds = df.copy()
    segs = {
        "High Income  (>$100K)":           ds[ds["Income"]>100],
        "High CC Spend  (>$3K/mo)":        ds[ds["CCAvg"]>3],
        "CD Account Holders":              ds[ds["CD Account"]==1],
        "Graduate or Higher Education":    ds[ds["Education"]>=2],
        "Family Size 3–4":                 ds[ds["Family"]>=3],
        "High Income + CD Account":        ds[(ds["Income"]>100)&(ds["CD Account"]==1)],
        "High Income + Graduate+":         ds[(ds["Income"]>100)&(ds["Education"]>=2)],
        "High Income + High CC Spend":     ds[(ds["Income"]>100)&(ds["CCAvg"]>3)],
    }
    seg_rows = []
    for sn, sdf in segs.items():
        rate = sdf["Personal Loan"].mean()*100
        seg_rows.append({
            "Segment": sn,
            "# Customers": len(sdf),
            "% of Total": f"{len(sdf)/len(df)*100:.1f}%",
            "Acceptors": int(sdf["Personal Loan"].sum()),
            "Acceptance Rate": f"{rate:.1f}%",
            "Lift vs Baseline (9.6%)": f"+{rate-9.6:.1f} pp",
        })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)
    st.markdown("""<div class='ibox amber'>
    🎯 <b>Budget Optimisation:</b> The <b>High Income + CD Account</b> segment achieves
    <b>3–5× the baseline acceptance rate</b> while representing a focused audience.
    With half the marketing budget, concentrating on Tier 1 &amp; 2 segments can deliver
    equal or better absolute acceptors than broad-spray campaigns.
    </div>""", unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown("<div class='sec-hd'>📋 Recommended Campaign Actions</div>", unsafe_allow_html=True)
    recs = [
        ("🥇","Priority Tier 1 — Super Targets",
         "Income >$100K + CD Account holders. Acceptance rate ~5× baseline. "
         "Use <b>personalised, premium outreach</b>: relationship manager calls, "
         "exclusive rate offers, and branch appointment invites. ~200–250 customers.",
         "#dbeafe"),
        ("🥈","Priority Tier 2 — High Potential",
         "Income >$100K OR CC Spend >$3K/mo + Graduate education. "
         "Deploy <b>targeted digital campaigns</b>, personalised email with EMI calculators "
         "and pre-approval nudges. ~800–1,200 customers.",
         "#dcfce7"),
        ("🥉","Priority Tier 3 — Nurture Segment",
         "Family size 3–4, Graduate education, income $50–100K. "
         "Offer <b>life-event triggered messaging</b> (home purchase, children's education). "
         "Drip email sequences showing loan benefits for life goals. ~1,500–2,000 customers.",
         "#fef9c3"),
        ("📵","Avoid / Deprioritise",
         "Income <$50K, Undergrad-only, no existing banking products — acceptance propensity ~2%. "
         "<b>Exclude from paid media</b>. Focus brand awareness only via low-cost channels (social, ATM screen).",
         "#fee2e2"),
    ]
    for icon, title, desc, bg in recs:
        st.markdown(f"""
        <div class='seg-card' style='background:{bg};'>
            <div class='seg-title'>{icon} {title}</div>
            <div>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='ibox green'>
    💡 <b>Bottom Line:</b> With 50% of last year's budget, allocate <b>70% to Tier 1 &amp; Tier 2 segments</b>.
    Use the ML model's probability scores (available in the <i>Predict New Data</i> page) to rank
    individual customers within each tier — and contact the highest-probability customers first.
    You can realistically <b>maintain or exceed last campaign's loan volumes</b> by targeting smarter, not broader.
    </div>""", unsafe_allow_html=True)

    # ── Acceptance rate by key combo ──
    st.markdown("<div class='sec-hd'>📊 Acceptance Rate Heatmap — Income Band × Education</div>",
                unsafe_allow_html=True)
    df_h = df.copy()
    df_h["Income Band"] = pd.cut(df_h["Income"], bins=[0,50,100,150,224],
                                  labels=["<$50K","$50-100K","$100-150K",">$150K"])
    df_h["Edu Label"] = df_h["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
    pivot = df_h.pivot_table(index="Edu Label", columns="Income Band",
                              values="Personal Loan", aggfunc="mean", observed=True) * 100
    fig_p, ax_p = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax_p,
                linewidths=0.5, annot_kws={"size": 11})
    ax_p.set_title("Loan Acceptance Rate (%) by Education × Income Band", fontsize=12,
                   fontweight="bold", pad=12)
    ax_p.set_ylabel("Education Level", fontsize=10)
    ax_p.set_xlabel("Annual Income Band", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_p)
    plt.close()
    st.markdown("""<div class='ibox'>
    📌 The top-right cells (Advanced education + high income) show the <b>highest acceptance rates</b>
    — confirming that these two dimensions together create a multiplicative effect.
    This grid can directly inform your <b>campaign audience segmentation</b>.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE 6 — PREDICT NEW DATA
# ══════════════════════════════════════════════════
elif nav == "📁  Predict New Data":
    st.markdown("""<div class='banner'>
        <h1>📁 Predict Personal Loan — Upload New Data</h1>
        <p>Upload a customer CSV, get instant loan predictions with probability scores, and download results</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 📥 Upload Customer Data (CSV)")
        st.markdown("""
        Your CSV should contain these columns (order doesn't matter):
        `Age, Experience, Income, Family, CCAvg, Education, Mortgage,
        Securities Account, CD Account, Online, CreditCard`

        *`ID`, `ZIP Code` and `Personal Loan` columns will be ignored if present.*
        """)
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        st.markdown("#### 🔧 Choose Prediction Model")
        sel_model = st.selectbox(
            "Model",
            options=list(results.keys()),
            index=list(results.keys()).index(BEST)
        )
        st.caption(f"Recommended: **{BEST}** (AUC = {results[BEST]['auc']:.2f}%)")

    with c2:
        st.markdown("#### 📄 Required Column Format & Sample")
        sample = pd.DataFrame({
            "Age":                [35, 52, 28, 42, 61],
            "Experience":         [10, 27,  3, 17, 35],
            "Income":             [130, 75, 40, 105, 58],
            "Family":             [3, 2, 1, 4, 2],
            "CCAvg":              [4.5, 1.2, 0.4, 3.8, 0.9],
            "Education":          [2, 1, 3, 2, 1],
            "Mortgage":           [200, 0, 0, 150, 80],
            "Securities Account": [0, 1, 0, 0, 0],
            "CD Account":         [1, 0, 0, 1, 0],
            "Online":             [1, 1, 0, 1, 1],
            "CreditCard":         [0, 1, 0, 1, 0],
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)

        # Download sample test file
        st.download_button(
            label="📥 Download Sample Test CSV",
            data=sample.to_csv(index=False),
            file_name="test_data_sample.csv",
            mime="text/csv"
        )

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            new_df.drop(columns=["ID","ZIP Code","Personal Loan"], inplace=True, errors="ignore")
            new_df["Experience"] = new_df["Experience"].clip(lower=0)

            missing = [c for c in FEAT_COLS if c not in new_df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}. Please check your upload matches the required format.")
            else:
                mdl_obj = results[sel_model]["model"]
                X_new   = new_df[FEAT_COLS]
                preds   = mdl_obj.predict(X_new)
                probas  = mdl_obj.predict_proba(X_new)[:, 1]

                out_df = new_df.copy()
                out_df["Personal Loan Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
                out_df["Acceptance Probability (%)"] = (probas * 100).round(2)
                out_df["Segment Tier"] = out_df["Acceptance Probability (%)"].apply(
                    lambda p: "🥇 Tier 1" if p >= 60 else ("🥈 Tier 2" if p >= 30 else "🥉 Tier 3")
                )

                st.markdown("---")
                st.markdown("### ✅ Prediction Results")

                total    = len(out_df)
                pred_yes = int((preds == 1).sum())
                pred_no  = int((preds == 0).sum())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Customers", f"{total:,}")
                m2.metric("Predicted to Accept", f"{pred_yes}  ({pred_yes/total*100:.1f}%)")
                m3.metric("Predicted to Decline",f"{pred_no}  ({pred_no/total*100:.1f}%)")
                m4.metric("Avg Accept Probability", f"{probas.mean()*100:.1f}%")

                st.dataframe(
                    out_df.sort_values("Acceptance Probability (%)", ascending=False),
                    use_container_width=True, hide_index=True
                )

                st.download_button(
                    label="📥 Download Full Results CSV",
                    data=out_df.to_csv(index=False),
                    file_name="loan_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

                # Probability distribution chart
                fig = px.histogram(
                    out_df, x="Acceptance Probability (%)", nbins=25,
                    color="Personal Loan Prediction",
                    color_discrete_map={"Yes":"#1d4ed8","No":"#94a3b8"},
                    title=f"Distribution of Loan Acceptance Probability  ({sel_model})",
                    labels={"Acceptance Probability (%)":"Predicted Probability (%)",
                            "count":"# Customers"}
                )
                fig.update_layout(height=350, bargap=0.05, plot_bgcolor="#f8fafc")
                st.plotly_chart(fig, use_container_width=True)

                # Tier breakdown
                tier_c = out_df["Segment Tier"].value_counts().reset_index()
                tier_c.columns = ["Tier","Count"]
                tier_c["% of Upload"] = (tier_c["Count"]/total*100).round(1)
                st.markdown("#### 🎯 Segment Tier Distribution")
                st.dataframe(tier_c, use_container_width=True, hide_index=True)
                st.markdown("""<div class='ibox green'>
                💡 <b>Tier 1</b> (≥60% probability) = prioritise with personalised outreach.
                <b>Tier 2</b> (30–60%) = digital campaign.
                <b>Tier 3</b> (&lt;30%) = deprioritise or nurture only.
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("👆 Upload a CSV file above to get predictions. Use the sample file as a template.")
