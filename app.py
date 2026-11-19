
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL DESIGN / CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F4F7FB;
    }

    [data-testid="stSidebar"] {
        background: #101D35;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #14213D;
        margin-bottom: 3px;
    }

    .subtitle {
        color: #718096;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #14213D;
        font-size: 19px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 13px;
        padding: 19px;
        min-height: 115px;
        box-shadow: 0 3px 12px rgba(20, 33, 61, 0.04);
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .kpi-value {
        font-size: 27px;
        font-weight: 800;
        color: #14213D;
        margin-top: 9px;
    }

    .kpi-note {
        color: #8794A8;
        font-size: 11px;
        margin-top: 4px;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 13px;
        padding: 18px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E5EAF2;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #718096;
    }

    .stButton > button {
        background: #2457D6;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 9px 18px;
    }

    .stButton > button:hover {
        background: #173FA8;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E5EAF2;
        border-radius: 10px;
    }

    h1, h2, h3 {
        color: #14213D;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FILES AND MODEL
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "churn_model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
DATA_PATH = BASE_DIR / "telecom_customers.csv"

FEATURES = [
    "gender",
    "senior_citizen",
    "tenure_months",
    "contract",
    "internet_service",
    "payment_method",
    "monthly_charges",
    "support_calls",
    "total_charges"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


try:
    model = load_model()
    data = load_data()
    metrics = load_metrics()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.info(
        "Make sure app.py, churn_model.joblib, metrics.json "
        "and telecom_customers.csv are in the same project folder."
    )
    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================
data.columns = data.columns.str.strip()

required_columns = FEATURES + ["churn"]

missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


def normalize_churn(series):
    """Convert common churn representations into 0 and 1."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    mapping = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "churned": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "retained": 0
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


data["churn"] = normalize_churn(data["churn"])

for col in ["tenure_months", "monthly_charges",
            "total_charges", "support_calls"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["senior_citizen"] = pd.to_numeric(
    data["senior_citizen"], errors="coerce"
).fillna(0).astype(int)

data = data.dropna(subset=FEATURES)

total_customers = len(data)
churned = int(data["churn"].sum())
retained = total_customers - churned

churn_rate = churned / total_customers if total_customers else 0
retention_rate = 1 - churn_rate

average_monthly_charge = data["monthly_charges"].mean()
monthly_revenue = data["monthly_charges"].sum()

accuracy = float(metrics.get("accuracy", 0))
roc_auc = float(metrics.get("roc_auc", 0))
precision = float(metrics.get("precision", 0))
recall = float(metrics.get("recall", 0))
f1 = float(metrics.get("f1", 0))


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("# 📡 TELECOM")
    st.markdown("### CHURN INTELLIGENCE")
    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "Executive Overview",
            "Customer Analytics",
            "Churn Prediction",
            "Batch Scoring",
            "Model Performance"
        ]
    )

    st.markdown("---")
    st.markdown("### PROJECT INFORMATION")
    st.caption("Telecom Customer Churn Prediction")
    st.caption("Machine Learning Analytics System")
    st.caption("Powered by Streamlit")

    st.markdown("---")
    st.caption("© Telecom Intelligence Dashboard")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Telecom Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Customer analytics, churn prediction and business performance monitoring'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# REUSABLE KPI CARD
# =========================================================
def kpi_card(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


# Common chart appearance
CHART_TEMPLATE = "plotly_white"

CHART_LAYOUT = dict(
    template=CHART_TEMPLATE,
    height=350,
    margin=dict(l=20, r=20, t=45, b=20),
    font=dict(family="Inter", color="#42516A"),
    paper_bgcolor="white",
    plot_bgcolor="white"
)


# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":

    section("Business Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Total Customers",
            f"{total_customers:,}",
            "Customers in the dataset"
        )

    with c2:
        kpi_card(
            "Churn Rate",
            f"{churn_rate:.1%}",
            f"{churned:,} customers churned"
        )

    with c3:
        kpi_card(
            "Monthly Revenue",
            f"{monthly_revenue:,.0f}",
            "Sum of monthly customer charges"
        )

    with c4:
        kpi_card(
            "Retention Rate",
            f"{retention_rate:.1%}",
            f"{retained:,} customers retained"
        )

    st.markdown("")
    left, right = st.columns([1.15, 1])

    with left:
        section("Customer Retention Distribution")

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Retained", "Churned"],
                    values=[retained, churned],
                    hole=0.68,
                    marker=dict(
                        colors=["#2457D6", "#F05252"]
                    ),
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value:,}<extra></extra>"
                )
            ]
        )

        fig.update_layout(
            **CHART_LAYOUT,
            height=370,
            showlegend=True,
            annotations=[
                dict(
                    text=f"{churn_rate:.1%}<br>Churn",
                    x=0.5,
                    y=0.5,
                    font_size=21,
                    font_color="#14213D",
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        section("Churn by Contract Type")

        contract_data = (
            data.groupby("contract", as_index=False)
            .agg(
                churn_rate=("churn", "mean"),
                customers=("churn", "count")
            )
        )

        contract_data["churn_rate"] *= 100

        fig = px.bar(
            contract_data,
            x="contract",
            y="churn_rate",
            text=contract_data["churn_rate"].map(
                lambda x: f"{x:.1f}%"
            ),
            color="churn_rate",
            color_continuous_scale=["#B9D0FF", "#2457D6"],
            labels={
                "contract": "Contract Type",
                "churn_rate": "Churn Rate (%)"
            }
        )

        fig.update_layout(
            **CHART_LAYOUT,
            coloraxis_showscale=False,
            xaxis_title="Contract",
            yaxis_title="Churn Rate (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        section("Churn by Internet Service")

        internet_data = (
            data.groupby("internet_service", as_index=False)
            ["churn"].mean()
        )

        internet_data["churn"] *= 100

        fig = px.bar(
            internet_data,
            x="internet_service",
            y="churn",
            color="internet_service",
            text=internet_data["churn"].map(
                lambda x: f"{x:.1f}%"
            ),
            labels={
                "internet_service": "Internet Service",
                "churn": "Churn Rate (%)"
            },
            color_discrete_sequence=[
                "#2457D6", "#13A89E", "#F0A43A"
            ]
        )

        fig.update_layout(
            **CHART_LAYOUT,
            showlegend=False,
            yaxis_title="Churn Rate (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        section("Customer Tenure Distribution")

        fig = px.histogram(
            data,
            x="tenure_months",
            color="churn",
            nbins=20,
            barmode="overlay",
            opacity=0.8,
            labels={
                "tenure_months": "Tenure (Months)",
                "churn": "Churn Status"
            },
            color_discrete_map={
                0: "#2457D6",
                1: "#F05252"
            }
        )

        fig.update_layout(
            **CHART_LAYOUT,
            legend_title="Churn",
            xaxis_title="Tenure (Months)",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE 2: CUSTOMER ANALYTICS
# =========================================================
elif page == "Customer Analytics":

    section("Customer Segmentation and Analysis")

    f1, f2, f3 = st.columns(3)

    with f1:
        contract_filter = st.multiselect(
            "Contract Type",
            sorted(data["contract"].dropna().unique()),
            default=sorted(data["contract"].dropna().unique())
        )

    with f2:
        internet_filter = st.multiselect(
            "Internet Service",
            sorted(data["internet_service"].dropna().unique()),
            default=sorted(data["internet_service"].dropna().unique())
        )

    with f3:
        churn_filter = st.selectbox(
            "Customer Status",
            ["All Customers", "Churned", "Retained"]
        )

    filtered = data[
        data["contract"].isin(contract_filter)
        & data["internet_service"].isin(internet_filter)
    ].copy()

    if churn_filter == "Churned":
        filtered = filtered[filtered["churn"] == 1]
    elif churn_filter == "Retained":
        filtered = filtered[filtered["churn"] == 0]

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        st.stop()

    a, b, c = st.columns(3)

    a.metric("Filtered Customers", f"{len(filtered):,}")
    b.metric("Average Monthly Charge",
             f"{filtered['monthly_charges'].mean():,.2f}")
    c.metric("Average Tenure",
             f"{filtered['tenure_months'].mean():.1f} months")

    left, right = st.columns(2)

    with left:
        section("Customer Distribution by Contract")

        fig = px.histogram(
            filtered,
            x="contract",
            color="churn",
            barmode="group",
            color_discrete_map={
                0: "#2457D6",
                1: "#F05252"
            },
            labels={
                "contract": "Contract",
                "count": "Customers",
                "churn": "Churn"
            }
        )

        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section("Monthly Charges by Churn Status")

        fig = px.box(
            filtered,
            x="churn",
            y="monthly_charges",
            color="churn",
            color_discrete_map={
                0: "#2457D6",
                1: "#F05252"
            },
            labels={
                "churn": "Churn Status",
                "monthly_charges": "Monthly Charges"
            }
        )

        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    section("Customer Records")

    display_columns = [
        col for col in [
            "customer_id", "gender", "tenure_months",
            "contract", "internet_service",
            "monthly_charges", "support_calls", "churn"
        ] if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Customers",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_customers.csv",
        "text/csv"
    )


# =========================================================
# PAGE 3: INDIVIDUAL CHURN PREDICTION
# =========================================================
elif page == "Churn Prediction":

    section("Individual Customer Churn Prediction")

    st.info(
        "Enter customer information below. The trained machine-learning "
        "model will estimate the probability of customer churn."
    )

    with st.form("prediction_form"):

        left, right = st.columns(2)

        with left:
            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            tenure = st.slider(
                "Tenure (Months)", 1, 72, 12
            )

            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with right:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mobile money",
                    "Bank transfer",
                    "Credit card"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=15.0,
                max_value=120.0,
                value=60.0,
                step=1.0
            )

            calls = st.number_input(
                "Support Calls",
                min_value=0,
                max_value=20,
                value=2
            )

        submitted = st.form_submit_button(
            "Analyze Customer Churn Risk",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "senior_citizen": senior,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "monthly_charges": monthly,
            "support_calls": calls,
            "total_charges": monthly * tenure
        }])[FEATURES]

        probability = model.predict_proba(customer)[0, 1]

        st.markdown("---")
        section("Prediction Results")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

            st.progress(float(probability))

        with c2:
            if probability >= 0.70:
                st.error("HIGH RISK: Immediate retention action recommended.")
            elif probability >= 0.50:
                st.warning("MEDIUM RISK: Customer should be monitored.")
            else:
                st.success("LOW RISK: Customer is likely to be retained.")

        st.caption(
            "The probability is a model estimate, not a guarantee "
            "of future customer behavior."
        )

        if probability >= 0.50:
            st.markdown("### Recommended Retention Actions")
            st.write(
                "- Contact the customer and understand their concerns."
            )
            st.write(
                "- Review available contract and pricing options."
            )
            st.write(
                "- Investigate repeated technical support issues."
            )
            st.write(
                "- Offer an appropriate retention incentive if justified."
            )


# =========================================================
# PAGE 4: BATCH CUSTOMER SCORING
# =========================================================
elif page == "Batch Scoring":

    section("Bulk Customer Churn Prediction")

    st.write(
        "Upload a CSV containing the same model input columns as "
        "the training dataset. The system will score every customer."
    )

    st.markdown("**Required columns:**")
    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_data = pd.read_csv(uploaded_file)
            missing = [
                col for col in FEATURES
                if col not in batch_data.columns
            ]

            if missing:
                st.error(
                    f"Your CSV is missing these required columns: {missing}"
                )
            else:
                X = batch_data[FEATURES].copy()

                for col in [
                    "tenure_months", "monthly_charges",
                    "support_calls", "total_charges",
                    "senior_citizen"
                ]:
                    X[col] = pd.to_numeric(
                        X[col], errors="coerce"
                    )

                if X.isnull().any().any():
                    st.error(
                        "Some model input fields contain missing or "
                        "invalid values. Correct the CSV and upload again."
                    )
                else:
                    probabilities = model.predict_proba(X)[:, 1]

                    results = batch_data.copy()
                    results["churn_probability"] = probabilities
                    results["risk_level"] = pd.cut(
                        probabilities,
                        bins=[-0.001, 0.50, 0.70, 1.001],
                        labels=["Low", "Medium", "High"]
                    )

                    results["predicted_churn"] = (
                        probabilities >= 0.50
                    ).astype(int)

                    high_risk = int(
                        (results["risk_level"] == "High").sum()
                    )

                    medium_risk = int(
                        (results["risk_level"] == "Medium").sum()
                    )

                    low_risk = int(
                        (results["risk_level"] == "Low").sum()
                    )

                    a, b, c, d = st.columns(4)

                    a.metric("Scored Customers", f"{len(results):,}")
                    b.metric("High Risk", f"{high_risk:,}")
                    c.metric("Medium Risk", f"{medium_risk:,}")
                    d.metric("Low Risk", f"{low_risk:,}")

                    section("Customer Risk Distribution")

                    risk_chart = pd.DataFrame({
                        "Risk Level": ["High", "Medium", "Low"],
                        "Customers": [
                            high_risk, medium_risk, low_risk
                        ]
                    })

                    fig = px.bar(
                        risk_chart,
                        x="Risk Level",
                        y="Customers",
                        color="Risk Level",
                        text="Customers",
                        color_discrete_map={
                            "High": "#E5484D",
                            "Medium": "#F0A43A",
                            "Low": "#13A89E"
                        }
                    )

                    fig.update_layout(
                        **CHART_LAYOUT,
                        showlegend=False
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    section("Highest-Risk Customers")

                    high_risk_customers = results.sort_values(
                        "churn_probability",
                        ascending=False
                    )

                    st.dataframe(
                        high_risk_customers,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Scored Customer Report",
                        results.to_csv(index=False).encode("utf-8"),
                        "telecom_churn_scored_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as error:
            st.error(f"Unable to process the uploaded CSV: {error}")


# =========================================================
# PAGE 5: MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    section("Machine Learning Model Evaluation")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{accuracy:.1%}")
    c2.metric("Precision", f"{precision:.1%}")
    c3.metric("Recall", f"{recall:.1%}")
    c4.metric("F1 Score", f"{f1:.1%}")
    c5.metric("ROC-AUC", f"{roc_auc:.1%}")

    section("Model Evaluation Metrics")

    model_metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    })

    fig = px.bar(
        model_metrics,
        x="Metric",
        y="Score",
        text=model_metrics["Score"].map(
            lambda x: f"{x:.1%}"
        ),
        color="Score",
        color_continuous_scale=["#B9D0FF", "#2457D6"]
    )

    fig.update_layout(
        **CHART_LAYOUT,
        yaxis=dict(
            title="Score",
            range=[0, 1]
        ),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

    section("Model Information")

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("**Algorithm**")
        st.write("Random Forest Classifier")

        st.markdown("**Prediction Type**")
        st.write("Binary Classification")

        st.markdown("**Target Variable**")
        st.write("Customer Churn")

    with info2:
        st.markdown("**Number of Input Features**")
        st.write(len(FEATURES))

        st.markdown("**Training Dataset Records**")
        st.write(f"{metrics.get('rows', len(data)):,}")

        st.markdown("**Dataset Churn Rate**")
        st.write(f"{metrics.get('churn_rate', churn_rate):.1%}")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Telecom Churn Intelligence | Machine Learning Business Analytics "
    "| Demonstration project using synthetic customer data"
)
