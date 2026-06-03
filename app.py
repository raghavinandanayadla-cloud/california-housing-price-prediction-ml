import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    VotingRegressor, BaggingRegressor, RandomForestRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 { font-size: 0.85rem; color: #6c757d; margin: 0; }
    .metric-card p  { font-size: 1.6rem; font-weight: 700; color: #1f77b4; margin: 4px 0 0; }
    .best-badge {
        background: #d4edda; color: #155724;
        border-radius: 6px; padding: 2px 8px;
        font-size: 0.75rem; font-weight: 700;
    }
    .section-header {
        border-left: 4px solid #1f77b4;
        padding-left: 12px;
        margin: 20px 0 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Caching ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"medhousval": "median_house_value"}, inplace=True)
    df["median_house_value"] = df["median_house_value"] * 100_000  # scale to dollars
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)
    return df

@st.cache_resource
def train_models(df):
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Individual models
    lr    = LinearRegression();      lr.fit(X_train_s, y_train)
    lasso = Lasso(alpha=0.1, max_iter=10000); lasso.fit(X_train_s, y_train)
    dt    = DecisionTreeRegressor(random_state=42); dt.fit(X_train_s, y_train)
    knn   = KNeighborsRegressor(n_neighbors=5);     knn.fit(X_train_s, y_train)
    svr   = SVR(kernel='rbf');  svr.fit(X_train_s, y_train)
    rf    = RandomForestRegressor(n_estimators=100, random_state=42); rf.fit(X_train_s, y_train)
    gb    = GradientBoostingRegressor(random_state=42); gb.fit(X_train, y_train)
    ada   = AdaBoostRegressor(random_state=42);     ada.fit(X_train, y_train)
    xgb   = XGBRegressor(random_state=42, verbosity=0); xgb.fit(X_train, y_train)
    bag   = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42)
    bag.fit(X_train, y_train)

    # Voting
    voting = VotingRegressor([('lr', lr), ('lasso', lasso), ('dt', dt)])
    voting.fit(X_train, y_train)

    # Stacking
    stack = StackingRegressor(
        estimators=[('lr', LinearRegression()), ('dt', DecisionTreeRegressor(random_state=42)), ('knn', KNeighborsRegressor())],
        final_estimator=RandomForestRegressor(n_estimators=50, random_state=42)
    )
    stack.fit(X_train, y_train)

    preds = {
        "Linear Regression":     lr.predict(X_test_s),
        "Lasso Regression":      lasso.predict(X_test_s),
        "Decision Tree":         dt.predict(X_test_s),
        "KNN":                   knn.predict(X_test_s),
        "SVR":                   svr.predict(X_test_s),
        "Random Forest":         rf.predict(X_test_s),
        "Gradient Boosting":     gb.predict(X_test),
        "AdaBoost":              ada.predict(X_test),
        "XGBoost":               xgb.predict(X_test),
        "Bagging":               bag.predict(X_test),
        "Voting":                voting.predict(X_test),
        "Stacking":              stack.predict(X_test),
    }

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "scaler": scaler,
        "models": {"lr": lr, "lasso": lasso, "dt": dt, "knn": knn,
                   "svr": svr, "rf": rf, "gb": gb, "ada": ada,
                   "xgb": xgb, "bag": bag, "voting": voting, "stack": stack},
        "preds": preds,
        "feature_names": X.columns.tolist(),
    }

def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MSE":  mse,
        "RMSE": np.sqrt(mse),
        "R²":   r2_score(y_true, y_pred),
    }

# ─── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models…"):
    df = load_data()
    bundle = train_models(df)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Flag_of_California.svg/800px-Flag_of_California.svg.png", use_container_width=True)
    st.title("🏠 CA Housing ML")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "📊 Dataset Overview",
        "📈 EDA",
        "🤖 Individual Models",
        "🎯 Ensemble Models",
        "🔍 Feature Importance",
        "🏆 Model Comparison",
        "🔮 Predict Your Price",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("California Housing Dataset · sklearn")
    st.caption(f"Rows: {len(df):,} · Features: {len(df.columns)-1}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("California Housing Dataset — 20,640 census block groups from the 1990 census.")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Total Rows", "Features", "Missing Values", "Duplicates"],
        [f"{len(df):,}", str(len(df.columns)-1), str(df.isnull().sum().sum()), str(df.duplicated().sum())]
    ):
        col.markdown(f"""<div class='metric-card'><h3>{label}</h3><p>{val}</p></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Raw Data</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.background_gradient(cmap="Blues", subset=["mean","std"]), use_container_width=True)

    st.markdown("<div class='section-header'>Data Types</div>", unsafe_allow_html=True)
    dtype_df = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.values, "Non-Null": df.notnull().sum().values})
    st.dataframe(dtype_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Target Distribution", "Feature Distributions", "Scatter Plots", "Correlation Heatmap"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["median_house_value"], kde=True, color="#1f77b4", ax=ax)
        ax.set_title("Distribution of Median House Value", fontsize=13, fontweight="bold")
        ax.set_xlabel("Median House Value ($)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for i, col in enumerate(num_cols[:9]):
            sns.histplot(df[col], kde=True, ax=axes[i//3][i%3], color="#1f77b4")
            axes[i//3][i%3].set_title(col, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        feat = st.selectbox("X-axis feature", [c for c in df.columns if c != "median_house_value"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[feat], df["median_house_value"], alpha=0.3, s=5, color="#1f77b4")
        ax.set_xlabel(feat);  ax.set_ylabel("Median House Value ($)")
        ax.set_title(f"{feat} vs Median House Value", fontweight="bold")
        st.pyplot(fig); plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mask, ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Heatmap", fontweight="bold")
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Individual Models
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Individual Models":
    st.title("🤖 Individual Regression Models")

    model_choice = st.selectbox("Select a model to explore", [
        "Linear Regression", "Lasso Regression", "Polynomial Regression",
        "Decision Tree", "KNN", "SVR"
    ])

    y_test  = bundle["y_test"]
    X_test  = bundle["X_test"]
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]

    if model_choice == "Polynomial Regression":
        X_poly_raw = df[["medinc"]] if "medinc" in df.columns else df[[df.columns[0]]]
        y_raw = df["median_house_value"]
        poly = PolynomialFeatures(degree=2)
        X_p  = poly.fit_transform(X_poly_raw)
        mdl  = LinearRegression().fit(X_p, y_raw)
        y_pred_poly = mdl.predict(X_p)
        m = compute_metrics(y_raw, y_pred_poly)
        preds_to_show = y_pred_poly[:len(y_test)]
        y_ref = y_raw.values[:len(y_test)]
    else:
        key_map = {
            "Linear Regression": "Linear Regression",
            "Lasso Regression":  "Lasso Regression",
            "Decision Tree":     "Decision Tree",
            "KNN":               "KNN",
            "SVR":               "SVR",
        }
        preds_to_show = bundle["preds"][key_map[model_choice]]
        y_ref = y_test.values
        m = compute_metrics(y_ref, preds_to_show)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4], m.keys(), m.values()):
        col.markdown(f"""<div class='metric-card'><h3>{label}</h3><p>{val:,.0f if label!="R²" else ".4f"}</p></div>""".replace(":,.0f if label!=\"R²\" else \".4f\"", ",.0f" if label != "R²" else ".4f"), unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Actual vs Predicted</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_ref[:500], preds_to_show[:500], alpha=0.5, s=15, color="#1f77b4")
    lims = [min(y_ref.min(), preds_to_show.min()), max(y_ref.max(), preds_to_show.max())]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set_xlabel("Actual Values"); axes[0].set_ylabel("Predicted Values")
    axes[0].set_title(f"{model_choice}: Actual vs Predicted", fontweight="bold")

    residuals = y_ref - preds_to_show
    axes[1].scatter(preds_to_show[:500], residuals[:500], alpha=0.5, s=15, color="#ff7f0e")
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot", fontweight="bold")

    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Ensemble Models
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Ensemble Models":
    st.title("🎯 Ensemble Learning Models")

    ensemble_models = ["Voting", "Bagging", "Random Forest", "Gradient Boosting", "AdaBoost", "XGBoost", "Stacking"]
    selected = st.selectbox("Select Ensemble Model", ensemble_models)

    y_test = bundle["y_test"]
    preds  = bundle["preds"][selected]
    m = compute_metrics(y_test.values, preds)

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1,c2,c3,c4], m.items()):
        fmt = f"{val:.4f}" if label == "R²" else f"{val:,.0f}"
        col.markdown(f"""<div class='metric-card'><h3>{label}</h3><p>{fmt}</p></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Actual vs Predicted</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = min(1000, len(y_test))
    axes[0].scatter(y_test.values[:n], preds[:n], alpha=0.4, s=12, color="#2ca02c")
    lims = [y_test.values[:n].min(), y_test.values[:n].max()]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{selected}: Actual vs Predicted", fontweight="bold")

    residuals = y_test.values - preds
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual")

    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance Analysis")

    rf  = bundle["models"]["rf"]
    xgb = bundle["models"]["xgb"]
    gb  = bundle["models"]["gb"]
    feat_names = bundle["feature_names"]

    tab1, tab2, tab3 = st.tabs(["Random Forest", "XGBoost", "Gradient Boosting"])

    for tab, mdl, label, color in [
        (tab1, rf,  "Random Forest",      "#1f77b4"),
        (tab2, xgb, "XGBoost",            "#ff7f0e"),
        (tab3, gb,  "Gradient Boosting",  "#2ca02c"),
    ]:
        with tab:
            fi = pd.DataFrame({"Feature": feat_names, "Importance": mdl.feature_importances_})
            fi = fi.sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(fi["Feature"], fi["Importance"], color=color, alpha=0.8)
            ax.set_title(f"{label} — Feature Importances", fontweight="bold")
            ax.set_xlabel("Importance Score")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.dataframe(fi.sort_values("Importance", ascending=False).reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")

    y_test = bundle["y_test"]
    rows = []
    for name, preds in bundle["preds"].items():
        m = compute_metrics(y_test.values, preds)
        rows.append({"Model": name, **m})

    results = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)

    best_model = results.iloc[0]["Model"]
    st.success(f"🏆 Best Model: **{best_model}** with R² = {results.iloc[0]['R²']:.4f}")

    def highlight_best(row):
        if row["Model"] == best_model:
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(
        results.style.apply(highlight_best, axis=1).format({"MAE": "{:,.0f}", "MSE": "{:,.0f}", "RMSE": "{:,.0f}", "R²": "{:.4f}"}),
        use_container_width=True
    )

    st.markdown("<div class='section-header'>R² Score Comparison</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if m == best_model else "#1f77b4" for m in results["Model"]]
    bars = ax.barh(results["Model"], results["R²"], color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="{:.3f}", padding=4, fontsize=9)
    ax.set_xlabel("R² Score"); ax.set_title("Model R² Comparison", fontweight="bold")
    ax.set_xlim(0, 1.05); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>RMSE Comparison</div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    results_r = results.sort_values("RMSE")
    ax2.barh(results_r["Model"], results_r["RMSE"], color="#ff7f0e", alpha=0.8)
    ax2.set_xlabel("RMSE ($)"); ax2.set_title("Model RMSE Comparison (lower = better)", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Predict Your Price
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Your Price":
    st.title("🔮 Predict Your House Price")
    st.markdown("Enter census block group statistics to estimate the median house value.")

    col1, col2 = st.columns(2)
    with col1:
        medinc    = st.slider("Median Income (tens of thousands $)", 0.5, 15.0, 3.87, 0.01)
        houseage  = st.slider("House Median Age (years)", 1, 52, 28)
        averooms  = st.slider("Average Rooms per Household", 1.0, 15.0, 5.43, 0.1)
        avebedrooms = st.slider("Average Bedrooms per Household", 0.5, 5.0, 1.10, 0.05)
    with col2:
        population = st.slider("Block Population", 3, 35000, 1425)
        aveoccup   = st.slider("Average Occupancy per Household", 1.0, 10.0, 3.07, 0.1)
        latitude   = st.slider("Latitude", 32.5, 42.0, 35.6, 0.1)
        longitude  = st.slider("Longitude", -124.3, -114.3, -119.6, 0.1)

    feat_names = bundle["feature_names"]
    input_arr  = np.array([[medinc, houseage, averooms, avebedrooms,
                            population, aveoccup, latitude, longitude]])

    # Align columns
    input_df = pd.DataFrame(input_arr, columns=feat_names[:8] if len(feat_names) >= 8 else feat_names)
    # pad missing cols
    for c in feat_names:
        if c not in input_df.columns:
            input_df[c] = 0.0
    input_df = input_df[feat_names]

    input_scaled = bundle["scaler"].transform(input_df)

    model_sel = st.selectbox("Choose prediction model", ["Gradient Boosting", "XGBoost", "Random Forest", "Linear Regression"])
    model_key = {"Gradient Boosting": "gb", "XGBoost": "xgb", "Random Forest": "rf", "Linear Regression": "lr"}[model_sel]
    mdl = bundle["models"][model_key]

    if st.button("🏠 Predict House Value", type="primary"):
        if model_key in ["gb", "xgb", "ada"]:
            pred = mdl.predict(input_df)[0]
        else:
            pred = mdl.predict(input_scaled)[0]
        pred = max(0, pred)

        st.markdown("---")
        st.markdown(f"""
        <div style='text-align:center; background: linear-gradient(135deg,#1f77b4,#0d4f8c);
                    border-radius:16px; padding:32px; color:white;'>
            <h2 style='margin:0; font-size:1rem; opacity:0.8'>Estimated Median House Value</h2>
            <h1 style='margin:8px 0; font-size:3rem; font-weight:800'>${pred:,.0f}</h1>
            <p style='margin:0; opacity:0.7'>Predicted by {model_sel}</p>
        </div>
        """, unsafe_allow_html=True)

        # Comparison with dataset average
        avg = df["median_house_value"].mean()
        pct_diff = (pred - avg) / avg * 100
        direction = "above" if pct_diff > 0 else "below"
        st.info(f"This estimate is **{abs(pct_diff):.1f}%** {direction} the dataset average of **${avg:,.0f}**.")
