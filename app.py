
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "churn_model.joblib"
DATA_PATH = BASE_DIR / "telecom_customers.csv"
METRICS_PATH = BASE_DIR / "metrics.json"

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

# =========================================================
# 2. PROFESSIONAL STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #101D35;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 30px;
    font-weight: 800;
    color: #14213D;
}

.subtitle {
    color: #718096;
    font-size: 14px;
    margin-bottom: 25px;
}

.kpi {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 12px;
    padding: 18px;
    min-height: 112px;
}

.kpi-label {
    font-size: 12px;
    color: #718096;
    font-weight: 600;
    text-transform: uppercase;
}

.kpi-value {
    color: #14213D;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-note {
    color: #8794A8;
    font-size: 11px;
}

.section {
    color: #14213D;
    font-size: 19px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}

.stButton button {
    background: #2457D6;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton button:hover {
    background: #173FA8;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 14px;
    border: 1px solid #E5EAF2;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD PROJECT FILES
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


try:
    model = load_model()
    data = load_dataset()
    metrics = load_metrics()

except Exception as error:
    st.error("Unable to load the project files.")
    st.code(str(error))
    st.write("Check that these files are in the same folder:")
    st.code("""
app.py
churn_model.joblib
telecom_customers.csv
metrics.json
requirements.txt
""")
    st.stop()


# =========================================================
# 4. DATA PREPARATION
# =========================================================
data.columns = data.columns.str.strip()

missing = [
    col for col in FEATURES + ["churn"]
    if col not in data.columns
]

if missing:
    st.error(f"Missing dataset columns: {missing}")
    st.stop()


def convert_churn(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    mapping = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "false": 0,
        "0": 0
    }

    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


data["churn"] = convert_churn(data["churn"])

for col in [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "senior_citizen"
]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=FEATURES + ["churn"])

total_customers = len(data)
churned = int(data["churn"].sum())
retained = total_customers - churned

churn_rate = churned / total_customers if total_customers else 0
retention_rate = 1 - churn_rate

revenue = data["monthly_charges"].sum()
average_charge = data["monthly_charges"].mean()

accuracy = float(metrics.get("accuracy", 0))
precision = float(metrics.get("precision", 0))
recall = float(metrics.get("recall", 0))
f1 = float(metrics.get("f1", 0))
roc_auc = float(metrics.get("roc_auc", 0))


# =========================================================
# 5. REUSABLE FUNCTIONS
# =========================================================
def heading(title):
    st.markdown(
        f'<div class="section">{title}</div>',
        unsafe_allow_html=True
    )


def kpi(label, value, note):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def make_chart():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def show_chart(fig):
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def prediction_frame(row):
    return pd.DataFrame([row], columns=FEATURES)


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("# 📡 kenny")
    st.markdown("### CHURN  Bon Ange INTELLIGENCE")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Customer Analytics",
            "Churn Prediction",
            "Batch Scoring",
            "Model Performance",
            "Dataset Explorer"
        ]
    )

    st.markdown("---")
    st.markdown("### SYSTEM INFORMATION")
    st.caption("Machine Learning Analytics")
    st.caption("Telecommunications")
    st.caption("Streamlit Dashboard")
    st.markdown("---")
    st.caption("Version 1.0")


# =========================================================
# 7. DASHBOARD HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Telecom Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Customer analytics, business intelligence and predictive modelling'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 8. EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":

    heading("Business Performance Overview")

    a, b, c, d = st.columns(4)

    with a:
        kpi("Total Customers", f"{total_customers:,}", "Dataset records")

    with b:
        kpi("Churn Rate", f"{churn_rate:.1%}", f"{churned:,} churned")

    with c:
        kpi("Monthly Revenue", f"{revenue:,.0f}", "Total monthly charges")

    with d:
        kpi("Retention Rate", f"{retention_rate:.1%}", f"{retained:,} retained")

    left, right = st.columns(2)

    with left:
        heading("Customer Retention Distribution")

        fig, ax = make_chart()

        ax.pie(
            [retained, churned],
            labels=["Retained", "Churned"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#2457D6", "#EF5350"],
            wedgeprops={"width": 0.4}
        )

        ax.set_title("Customer Status")
        show_chart(fig)

    with right:
        heading("Churn Rate by Contract")

        contract = data.groupby("contract")["churn"].mean() * 100

        fig, ax = make_chart()

        bars = ax.bar(
            contract.index.astype(str),
            contract.values,
            color="#2457D6"
        )

        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Contract Type")
        ax.set_title("Contract Performance")

        ax.bar_label(bars, fmt="%.1f%%", padding=3)

        show_chart(fig)

    left, right = st.columns(2)

    with left:
        heading("Churn by Internet Service")

        internet = data.groupby("internet_service")["churn"].mean() * 100

        fig, ax = make_chart()

        bars = ax.bar(
            internet.index.astype(str),
            internet.values,
            color="#13A89E"
        )

        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Internet Service Performance")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)

        show_chart(fig)

    with right:
        heading("Customer Tenure Distribution")

        fig, ax = make_chart()

        ax.hist(
            data.loc[data["churn"] == 0, "tenure_months"],
            bins=15,
            alpha=0.75,
            label="Retained",
            color="#2457D6"
        )

        ax.hist(
            data.loc[data["churn"] == 1, "tenure_months"],
            bins=15,
            alpha=0.75,
            label="Churned",
            color="#EF5350"
        )

        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Customers")
        ax.legend()

        show_chart(fig)


# =========================================================
# 9. CUSTOMER ANALYTICS
# =========================================================
elif page == "Customer Analytics":

    heading("Customer Segmentation")

    col1, col2, col3 = st.columns(3)

    with col1:
        contracts = sorted(data["contract"].dropna().unique())

        selected_contracts = st.multiselect(
            "Contract",
            contracts,
            default=contracts
        )

    with col2:
        services = sorted(data["internet_service"].dropna().unique())

        selected_services = st.multiselect(
            "Internet Service",
            services,
            default=services
        )

    with col3:
        status = st.selectbox(
            "Customer Status",
            ["All", "Churned", "Retained"]
        )

    filtered = data[
        data["contract"].isin(selected_contracts)
        & data["internet_service"].isin(selected_services)
    ].copy()

    if status == "Churned":
        filtered = filtered[filtered["churn"] == 1]

    elif status == "Retained":
        filtered = filtered[filtered["churn"] == 0]

    if filtered.empty:
        st.warning("No records match these filters.")

    else:
        a, b, c = st.columns(3)

        a.metric("Filtered Customers", f"{len(filtered):,}")
        b.metric("Average Charges", f"{filtered['monthly_charges'].mean():.2f}")
        c.metric("Average Tenure", f"{filtered['tenure_months'].mean():.1f} months")

        left, right = st.columns(2)

        with left:
            heading("Customers by Contract")

            fig, ax = make_chart()

            filtered.groupby(
                ["contract", "churn"]
            ).size().unstack(fill_value=0).plot(
                kind="bar",
                ax=ax,
                color=["#2457D6", "#EF5350"]
            )

            ax.set_ylabel("Customers")
            ax.set_xlabel("Contract")
            ax.legend(["Retained", "Churned"])

            show_chart(fig)

        with right:
            heading("Monthly Charges by Customer Status")

            fig, ax = make_chart()

            ax.boxplot([
                filtered.loc[
                    filtered["churn"] == 0, "monthly_charges"
                ].dropna(),
                filtered.loc[
                    filtered["churn"] == 1, "monthly_charges"
                ].dropna()
            ], tick_labels=["Retained", "Churned"])

            ax.set_ylabel("Monthly Charges")

            show_chart(fig)

        heading("Filtered Customer Records")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Customer Data",
            filtered.to_csv(index=False).encode("utf-8"),
            "filtered_customers.csv",
            "text/csv"
        )


# =========================================================
# 10. INDIVIDUAL CHURN PREDICTION
# =========================================================
elif page == "Churn Prediction":

    heading("Individual Customer Risk Assessment")

    st.info(
        "Enter customer details to estimate the probability of churn."
    )

    with st.form("prediction_form"):

        left, right = st.columns(2)

        with left:
            gender = st.selectbox("Gender", ["Male", "Female"])

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            tenure = st.slider("Tenure (Months)", 1, 72, 12)

            contract = st.selectbox(
                "Contract",
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
                value=60.0
            )

            calls = st.number_input(
                "Support Calls",
                min_value=0,
                max_value=20,
                value=2
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        row = {
            "gender": gender,
            "senior_citizen": senior,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "monthly_charges": monthly,
            "support_calls": calls,
            "total_charges": monthly * tenure
        }

        X = prediction_frame(row)

        try:
            probability = float(model.predict_proba(X)[0, 1])

            heading("Prediction Results")

            left, right = st.columns(2)

            with left:
                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )
                st.progress(probability)

            with right:
                if probability >= 0.70:
                    st.error("HIGH RISK")
                elif probability >= 0.50:
                    st.warning("MEDIUM RISK")
                else:
                    st.success("LOW RISK")

            if probability >= 0.50:
                heading("Suggested Retention Actions")

                st.write("- Contact the customer.")
                st.write("- Review contract and pricing options.")
                st.write("- Investigate technical support complaints.")
                st.write("- Consider a suitable retention offer.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# 11. BATCH SCORING
# =========================================================
elif page == "Batch Scoring":

    heading("Bulk Customer Churn Prediction")

    st.write(
        "Upload a customer CSV file containing the nine required "
        "model input columns."
    )

    st.code(", ".join(FEATURES))

    uploaded = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded:

        try:
            batch = pd.read_csv(uploaded)

            missing = [
                col for col in FEATURES if col not in batch.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")

            else:
                X = batch[FEATURES].copy()

                for col in [
                    "tenure_months",
                    "monthly_charges",
                    "support_calls",
                    "total_charges",
                    "senior_citizen"
                ]:
                    X[col] = pd.to_numeric(
                        X[col], errors="coerce"
                    )

                if X.isnull().any().any():
                    st.error("Your uploaded file contains invalid or empty values.")

                else:
                    probabilities = model.predict_proba(X)[:, 1]

                    results = batch.copy()

                    results["churn_probability"] = probabilities

                    results["risk_level"] = [
                        "High" if p >= 0.70
                        else "Medium" if p >= 0.50
                        else "Low"
                        for p in probabilities
                    ]

                    results["predicted_churn"] = (
                        probabilities >= 0.50
                    ).astype(int)

                    high = int((results["risk_level"] == "High").sum())
                    medium = int((results["risk_level"] == "Medium").sum())
                    low = int((results["risk_level"] == "Low").sum())

                    a, b, c, d = st.columns(4)

                    a.metric("Total Scored", f"{len(results):,}")
                    b.metric("High Risk", f"{high:,}")
                    c.metric("Medium Risk", f"{medium:,}")
                    d.metric("Low Risk", f"{low:,}")

                    heading("Customer Risk Distribution")

                    fig, ax = make_chart()

                    ax.bar(
                        ["High", "Medium", "Low"],
                        [high, medium, low],
                        color=["#EF5350", "#F0A43A", "#13A89E"]
                    )

                    ax.set_ylabel("Customers")

                    show_chart(fig)

                    heading("Customer Risk Results")

                    st.dataframe(
                        results.sort_values(
                            "churn_probability",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Scored Report",
                        results.to_csv(index=False).encode("utf-8"),
                        "telecom_churn_results.csv",
                        "text/csv"
                    )

        except Exception as error:
            st.error(f"Batch scoring failed: {error}")


# =========================================================
# 12. KENNY VS Bon Ange
# =========================================================
elif page == "Model Performance":

    heading("Machine Learning Model Evaluation")

    a, b, c, d, e = st.columns(5)

    a.metric("Accuracy", f"{accuracy:.1%}")
    b.metric("Precision", f"{precision:.1%}")
    c.metric("Recall", f"{recall:.1%}")
    d.metric("F1 Score", f"{f1:.1%}")
    e.metric("ROC-AUC", f"{roc_auc:.1%}")

    heading("Performance Comparison")

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    values = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]

    fig, ax = make_chart()

    bars = ax.bar(
        names,
        values,
        color="#2457D6"
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Performance Score")
    ax.set_title("Classification Model Metrics")

    ax.bar_label(
        bars,
        labels=[f"{v:.1%}" for v in values],
        padding=3
    )

    show_chart(fig)

    heading("Model Information")

    left, right = st.columns(2)

    with left:
        st.write("**Algorithm:** Random Forest Classifier")
        st.write("**Problem Type:** Binary Classification")
        st.write("**Target:** Customer Churn")

    with right:
        st.write("**Input Features:**", len(FEATURES))
        st.write("**Training Records:**", metrics.get("rows", total_customers))
        st.write("**Churn Rate:**", f"{metrics.get('churn_rate', churn_rate):.1%}")


# =========================================================
# 13. DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":

    heading("Telecom Customer Dataset Explorer")

    st.write("Explore the dataset used by the churn prediction system.")

    st.metric("Dataset Records", f"{len(data):,}")

    st.metric("Dataset Columns", f"{len(data.columns):,}")

    heading("Dataset Preview")

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True
    )

    heading("Statistical Summary")

    st.dataframe(
        data.describe(include="all").transpose(),
        use_container_width=True
    )

    heading("Missing Data Analysis")

    missing_data = data.isnull().sum()

    missing_data = missing_data[missing_data > 0]

    if missing_data.empty:
        st.success("No missing values found in the cleaned dataset.")
    else:
        st.dataframe(
            missing_data.rename("Missing Values").to_frame(),
            use_container_width=True
        )

    st.download_button(
        "Download Complete Dataset",
        data.to_csv(index=False).encode("utf-8"),
        "telecom_customers.csv",
        "text/csv"
    )


# =========================================================
# 14. FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Telecom Churn Intelligence | Machine Learning Business Analytics "
    "| Demonstration dataset"
)
