import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Stacking Regressor",
    page_icon="🏠",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🏠 California Housing Price Prediction")
st.markdown(
    "### Stacking Regression using Linear Regression, Random Forest and SVR"
)

st.markdown("---")

# =====================================
# LOAD DATASET
# =====================================

data = fetch_california_housing()

X = data.data
y = data.target

df = pd.DataFrame(
    X,
    columns=data.feature_names
)

df["Target"] = y

# =====================================
# LOAD MODEL
# =====================================

with open(
    "stacking_regressor.pkl",
    "rb"
) as f:

    model = pickle.load(f)

# =====================================
# INPUT SECTION
# =====================================

st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:

    MedInc = st.number_input(
        "Median Income",
        value=float(df["MedInc"].mean())
    )

    HouseAge = st.number_input(
        "House Age",
        value=float(df["HouseAge"].mean())
    )

    AveRooms = st.number_input(
        "Average Rooms",
        value=float(df["AveRooms"].mean())
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        value=float(df["AveBedrms"].mean())
    )

with col2:

    Population = st.number_input(
        "Population",
        value=float(df["Population"].mean())
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        value=float(df["AveOccup"].mean())
    )

    Latitude = st.number_input(
        "Latitude",
        value=float(df["Latitude"].mean())
    )

    Longitude = st.number_input(
        "Longitude",
        value=float(df["Longitude"].mean())
    )

# =====================================
# PREDICT BUTTON
# =====================================

st.markdown("")

if st.button(
    "Predict House Price",
    use_container_width=True
):

    input_data = np.array([
        [
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]
    ])

    prediction = model.predict(
        input_data
    )

    st.success(
        f"🏡 Predicted House Price: ${prediction[0]*100000:.2f}"
    )

# =====================================
# DATASET PREVIEW
# =====================================

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(
    df.head()
)

# =====================================
# CORRELATION HEATMAP
# =====================================

st.markdown("---")

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(10,6)
)

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# =====================================
# MODEL COMPARISON
# =====================================

st.markdown("---")

st.subheader("Model Performance Comparison")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
svr = SVR()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
svr_pred = svr.predict(X_test)
stack_pred = model.predict(X_test)

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "SVR",
        "Stacking Regressor"
    ],

    "R² Score": [

        r2_score(y_test, lr_pred),

        r2_score(y_test, rf_pred),

        r2_score(y_test, svr_pred),

        r2_score(y_test, stack_pred)
    ]
})

st.dataframe(
    results,
    use_container_width=True
)

# =====================================
# BAR CHART
# =====================================

fig2, ax2 = plt.subplots(
    figsize=(8,5)
)

ax2.bar(
    results["Model"],
    results["R² Score"]
)

ax2.set_title(
    "Model Comparison"
)

plt.xticks(
    rotation=20
)

st.pyplot(fig2)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.info(
    """
    Stacking Regression

    Base Learners:
    • Linear Regression
    • Random Forest Regressor
    • Support Vector Regressor (SVR)

    Meta Learner:
    • Linear Regression

    Evaluation Metric:
    • R² Score
    """
)